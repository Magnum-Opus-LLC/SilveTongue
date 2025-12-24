"""
Silvertongue Dashboard - Web interface for managing actors and cognitive loops.

Reads configuration from config.json including array_dir path.
"""

import sys
from pathlib import Path
import json
import yaml
from datetime import datetime
import threading
import queue
import traceback
import re
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify, request, Response

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response):
    """Add headers to prevent browser caching."""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# Global state for running loops
active_loops = {}
loop_outputs = {}  # Store output queues for each loop


def load_config():
    """Load configuration from config.json."""
    config_path = project_root / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def get_array_dir():
    """Get the array directory from config.json."""
    config = load_config()
    array_dir = config.get('array_dir', '')
    if array_dir:
        return Path(array_dir)
    return project_root / 'array'


def get_lora_adapters_dir():
    """Get the lora_adapters directory from config.json."""
    config = load_config()
    lora_dir = config.get('lora_adapters_dir', '')
    if lora_dir:
        return Path(lora_dir)
    # Fallback: try to derive from array_dir
    array_dir = get_array_dir()
    return array_dir.parent / 'lora_adapters'


def get_pdf_cache_dir():
    """Get the PDF cache directory."""
    array_dir = get_array_dir()
    return array_dir / 'data' / 'pdf_cache'


def get_pdf_stock_path():
    """Get the path to the PDF stock index file."""
    return get_pdf_cache_dir() / 'pdf_stock.json'


def normalize_title(title: str) -> str:
    """Normalize a title for matching - lowercase, remove punctuation, collapse spaces."""
    if not title:
        return ''
    # Lowercase
    normalized = title.lower()
    # Remove punctuation except hyphens
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    # Collapse multiple spaces/underscores to single space
    normalized = re.sub(r'[\s_]+', ' ', normalized).strip()
    return normalized


def sanitize_title_for_filename(title: str) -> str:
    """Sanitize a paper title to match PDF cache naming convention."""
    # Remove special characters and normalize spaces
    sanitized = re.sub(r'[^\w\s-]', '', title)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    # Limit length
    return sanitized[:100]


def load_pdf_stock() -> dict:
    """Load the PDF stock index from disk."""
    stock_path = get_pdf_stock_path()
    if stock_path.exists():
        try:
            with open(stock_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'pdfs': {}, 'title_index': {}, 'updated_at': None}


def save_pdf_stock(stock: dict):
    """Save the PDF stock index to disk."""
    import datetime
    stock_path = get_pdf_stock_path()
    stock['updated_at'] = datetime.datetime.now().isoformat()
    with open(stock_path, 'w') as f:
        json.dump(stock, f, indent=2)


def rebuild_pdf_stock() -> dict:
    """Rebuild the PDF stock index by scanning the cache directory."""
    import datetime
    pdf_cache_dir = get_pdf_cache_dir()
    
    stock = {
        'pdfs': {},           # filename -> {path, size_mb, original_title, normalized_title, added_at}
        'title_index': {},    # normalized_title -> filename (for quick lookup)
        'updated_at': datetime.datetime.now().isoformat(),
        'total_count': 0,
        'total_size_mb': 0
    }
    
    if not pdf_cache_dir.exists():
        return stock
    
    total_size = 0
    for pdf_file in pdf_cache_dir.glob('*.pdf'):
        filename = pdf_file.name
        stem = pdf_file.stem
        
        # Extract title from filename (format: hash_title.pdf)
        if '_' in stem:
            title_part = stem.split('_', 1)[1]
        else:
            title_part = stem
        
        # Convert underscores back to spaces for the original title
        original_title = title_part.replace('_', ' ')
        normalized = normalize_title(original_title)
        
        try:
            size_mb = round(pdf_file.stat().st_size / (1024 * 1024), 2)
        except:
            size_mb = 0
        
        total_size += size_mb
        
        stock['pdfs'][filename] = {
            'path': str(pdf_file),
            'size_mb': size_mb,
            'original_title': original_title,
            'normalized_title': normalized,
            'references': []  # Papers that matched this PDF
        }
        
        # Index by normalized title for quick lookup
        if normalized:
            stock['title_index'][normalized] = filename
    
    stock['total_count'] = len(stock['pdfs'])
    stock['total_size_mb'] = round(total_size, 2)
    
    # Save to disk
    save_pdf_stock(stock)
    print(f"[PDF Stock] Rebuilt index: {stock['total_count']} PDFs, {stock['total_size_mb']} MB total")
    
    return stock


# In-memory cache for PDF stock
_pdf_stock_cache = {'stock': None, 'loaded_at': 0}

def get_pdf_stock(force_rebuild=False) -> dict:
    """Get the PDF stock, loading from disk or rebuilding if needed."""
    import time
    now = time.time()
    
    # Check if we need to reload (every 5 minutes or if forced)
    if force_rebuild or _pdf_stock_cache['stock'] is None or (now - _pdf_stock_cache['loaded_at']) > 300:
        stock_path = get_pdf_stock_path()
        
        if not force_rebuild and stock_path.exists():
            # Load existing stock
            _pdf_stock_cache['stock'] = load_pdf_stock()
            _pdf_stock_cache['loaded_at'] = now
        else:
            # Rebuild stock
            _pdf_stock_cache['stock'] = rebuild_pdf_stock()
            _pdf_stock_cache['loaded_at'] = now
    
    return _pdf_stock_cache['stock']


def find_cached_pdf_by_path(source_file: str) -> dict:
    """
    Find a cached PDF by its exact source_file path.
    Returns dict with 'found', 'filename', 'path', 'size_mb'.
    """
    if not source_file:
        return {'found': False}
    
    pdf_path = Path(source_file)
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        return {
            'found': True,
            'filename': pdf_path.name,
            'path': str(pdf_path),
            'size_mb': round(size_mb, 2)
        }
    return {'found': False}


def find_pdf_in_stock(title: str, knowledge_bits: list = None) -> dict:
    """
    Find a cached PDF using the stock index.
    Uses multiple matching strategies:
    1. Direct path from knowledge_bits source_file
    2. Exact normalized title match
    3. Fuzzy title match (significant word overlap)
    
    Returns dict with 'found', 'filename', 'path', 'size_mb', 'match_type'.
    """
    # First, try to find via source_file in knowledge_bits
    if knowledge_bits:
        for bit in knowledge_bits:
            source_file = bit.get('source_file', '')
            if source_file:
                result = find_cached_pdf_by_path(source_file)
                if result['found']:
                    result['match_type'] = 'source_file'
                    return result
    
    # Load the stock index
    stock = get_pdf_stock()
    if not stock or not stock.get('pdfs'):
        return {'found': False}
    
    # Normalize the search title
    normalized_search = normalize_title(title)
    if not normalized_search:
        return {'found': False}
    
    # Strategy 2: Exact normalized title match
    if normalized_search in stock['title_index']:
        filename = stock['title_index'][normalized_search]
        pdf_info = stock['pdfs'].get(filename)
        if pdf_info:
            return {
                'found': True,
                'filename': filename,
                'path': pdf_info['path'],
                'size_mb': pdf_info['size_mb'],
                'match_type': 'exact_title'
            }
    
    # Strategy 3: Fuzzy matching - check word overlap
    search_words = set(normalized_search.split())
    # Filter to significant words (length > 3)
    search_words = {w for w in search_words if len(w) > 3}
    
    if len(search_words) >= 2:
        best_match = None
        best_score = 0
        
        for filename, pdf_info in stock['pdfs'].items():
            pdf_normalized = pdf_info.get('normalized_title', '')
            pdf_words = set(pdf_normalized.split())
            pdf_words = {w for w in pdf_words if len(w) > 3}
            
            if not pdf_words:
                continue
            
            # Calculate overlap score
            overlap = search_words & pdf_words
            if len(overlap) >= 2:
                # Score based on how much of both titles match
                score = len(overlap) / max(len(search_words), len(pdf_words))
                
                # Bonus for matching first few significant words
                search_first = [w for w in normalized_search.split()[:3] if len(w) > 3]
                pdf_first = [w for w in pdf_normalized.split()[:3] if len(w) > 3]
                if search_first and pdf_first and search_first[0] == pdf_first[0]:
                    score += 0.3
                
                if score > best_score and score >= 0.5:  # Require at least 50% match
                    best_score = score
                    best_match = (filename, pdf_info)
        
        if best_match:
            filename, pdf_info = best_match
            return {
                'found': True,
                'filename': filename,
                'path': pdf_info['path'],
                'size_mb': pdf_info['size_mb'],
                'match_type': f'fuzzy_{int(best_score*100)}%'
            }
    
    return {'found': False}


# Legacy function name for compatibility
def find_cached_pdf(title: str, knowledge_bits: list = None) -> dict:
    """Find a cached PDF - wrapper for find_pdf_in_stock."""
    return find_pdf_in_stock(title, knowledge_bits)


def load_yaml_file(file_path):
    """Load a YAML file safely."""
    if file_path.exists():
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def get_actor_root(actor_name):
    """Get the root directory for an actor."""
    if actor_name == 'template':
        return project_root / 'actors' / actor_name
    return get_array_dir() / 'actors' / actor_name


def get_available_actors():
    """Get list of available actors."""
    array_dir = get_array_dir()
    actors_dir = array_dir / 'actors'
    
    actors = []
    if actors_dir.exists():
        for item in actors_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if (item / 'behavior.json').exists() or (item / 'characteristic').exists():
                    actors.append(item.name)
    
    # Also check template in project root
    template_dir = project_root / 'actors' / 'template'
    if template_dir.exists() and 'template' not in actors:
        actors.append('template')
    
    return sorted(actors)


def get_actor_details(actor_name):
    """Get detailed information about an actor (lightweight, no heavy imports)."""
    actor_root = get_actor_root(actor_name)
    
    # Define paths
    paths = {
        'principles': actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
        'adjustments': actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml',
        'core_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
        'supporting_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
        'experiences': actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
        'challenges': actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
        'values': actor_root / 'characteristic' / 'value' / 'values.yaml',
        'learning_book': actor_root / 'learning_book.json'
    }
    
    def get_list(key):
        data = load_yaml_file(paths[key])
        return data.get(key, [])
    
    # Load learning book
    learning_book = {}
    if paths['learning_book'].exists():
        with open(paths['learning_book'], 'r') as f:
            learning_book = json.load(f)
    
    # Get archive history
    archive_root = actor_root / 'archive'
    archive_history = []
    if archive_root.exists():
        for key_dir in archive_root.iterdir():
            if key_dir.is_dir():
                for f in sorted(key_dir.glob('*'), reverse=True)[:3]:
                    archive_history.append({
                        'key': key_dir.name,
                        'file': f.name,
                        'path': str(f),
                        'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })
    
    return {
        'name': actor_name,
        'path': str(actor_root),
        'values': get_list('values'),
        'principles': get_list('principles'),
        'experiences': get_list('experiences'),
        'core_knowledge': get_list('core_knowledge'),
        'supporting_knowledge': get_list('supporting_knowledge'),
        'challenges': get_list('challenges'),
        'adjustments': get_list('adjustments'),
        'mindset': learning_book.get('current_mindset', {}),
        'recent_learnings': learning_book.get('learning_history', [])[-10:],
        'archive_history': archive_history[:20]
    }


def add_to_yaml_list(file_path, key, item, source=None):
    """Add an item to a YAML list file."""
    data = load_yaml_file(file_path)
    items = data.get(key, [])
    
    if source:
        item = f"{item} [Source: {source}]"
    
    # Avoid duplicates
    item_lower = item.lower()[:50]
    for existing in items:
        if existing.lower()[:50] == item_lower:
            return False
    
    items.append(item)
    data[key] = items
    data['_meta'] = {
        'last_updated': datetime.now().isoformat(),
        'updated_by': 'dashboard'
    }
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return True


# ==================== Routes ====================

@app.route('/')
def index():
    """Main dashboard page."""
    actors = get_available_actors()
    response = app.make_response(render_template('index.html', actors=actors))
    # Disable caching to ensure fresh JS
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/actors')
def api_actors():
    """Get list of actors."""
    return jsonify(get_available_actors())


@app.route('/api/actor/<actor_name>')
def api_actor_details(actor_name):
    """Get actor details."""
    try:
        details = get_actor_details(actor_name)
        return jsonify(details)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/knowledge/<knowledge_type>')
def api_actor_knowledge(actor_name, knowledge_type):
    """Get specific knowledge type for an actor."""
    try:
        details = get_actor_details(actor_name)
        if knowledge_type in details:
            return jsonify({knowledge_type: details[knowledge_type]})
        return jsonify({'error': 'Invalid knowledge type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/archive/<category>')
def api_actor_archive(actor_name, category):
    """Get archived items for a category with source mappings."""
    actor_root = get_actor_root(actor_name)
    archive_dir = actor_root / 'archive' / category
    
    if not archive_dir.exists():
        return jsonify({'error': f'No archive found for {category}'}), 404
    
    archives = []
    for file in sorted(archive_dir.glob('*.yaml'), reverse=True):
        try:
            with open(file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check for corresponding mappings file
            timestamp = file.stem.replace(f'{category}_', '')
            mappings_file = archive_dir / f'{category}_{timestamp}_mappings.json'
            mappings = None
            if mappings_file.exists():
                with open(mappings_file, 'r') as f:
                    mappings = json.load(f)
            
            archives.append({
                'filename': file.name,
                'timestamp': timestamp,
                'items': data.get(category, []),
                'meta': data.get('_meta', {}),
                'mappings': mappings
            })
        except Exception as e:
            continue
    
    return jsonify({'archives': archives})


@app.route('/api/actor/<actor_name>/archive/<category>/<timestamp>')
def api_actor_archive_detail(actor_name, category, timestamp):
    """Get specific archive with full details and mappings."""
    actor_root = get_actor_root(actor_name)
    archive_dir = actor_root / 'archive' / category
    
    archive_file = archive_dir / f'{category}_{timestamp}.yaml'
    mappings_file = archive_dir / f'{category}_{timestamp}_mappings.json'
    
    if not archive_file.exists():
        return jsonify({'error': 'Archive not found'}), 404
    
    try:
        with open(archive_file, 'r') as f:
            archive_data = yaml.safe_load(f)
        
        result = {
            'category': category,
            'timestamp': timestamp,
            'items': archive_data.get(category, []),
            'meta': archive_data.get('_meta', {}),
            'mappings': None
        }
        
        if mappings_file.exists():
            with open(mappings_file, 'r') as f:
                result['mappings'] = json.load(f)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/knowledge/<knowledge_type>', methods=['POST'])
def api_add_knowledge(actor_name, knowledge_type):
    """Add new knowledge item."""
    actor_root = get_actor_root(actor_name)
    data = request.json
    content = data.get('content', '')
    source = data.get('source', 'dashboard')
    
    paths = {
        'principles': actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
        'experiences': actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
        'core_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
        'supporting_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
        'challenges': actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
        'adjustments': actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml'
    }
    
    if knowledge_type not in paths:
        return jsonify({'error': 'Invalid knowledge type'}), 400
    
    success = add_to_yaml_list(paths[knowledge_type], knowledge_type, content, source)
    return jsonify({'success': success})


@app.route('/api/actor/<actor_name>/knowledge/expand', methods=['POST'])
def api_expand_knowledge(actor_name):
    """Generate AI expansion for a knowledge item."""
    data = request.json
    content = data.get('content', '')
    knowledge_type = data.get('type', '')
    
    if not content:
        return jsonify({'error': 'No content provided'}), 400
    
    try:
        # Import chat utils
        from src.utils.chat_utils import get_chat_completion
        
        type_labels = {
            'values': 'value',
            'principles': 'principle', 
            'experiences': 'experience',
            'core_knowledge': 'core knowledge item',
            'challenges': 'challenge'
        }
        type_label = type_labels.get(knowledge_type, 'knowledge item')
        
        system_prompt = "You are a knowledgeable expert providing detailed expansions of concepts and ideas. Write in a clear, informative style using markdown formatting."
        
        user_prompt = f"""Expand on the following {type_label} with additional context, examples, implications, and practical applications.

{type_label.upper()}: {content}

Provide a comprehensive expansion that includes:
1. **Deeper Explanation**: What does this really mean in practice?
2. **Examples**: Concrete examples or scenarios where this applies
3. **Implications**: What are the broader implications or consequences?
4. **Related Concepts**: How does this connect to other ideas or principles?
5. **Practical Applications**: How can this be applied in real situations?"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        expansion = get_chat_completion(messages, temperature=0.7)
        
        return jsonify({'expansion': expansion})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/knowledge/save-expansion', methods=['POST'])
def api_save_knowledge_expansion(actor_name):
    """Save expansion and notes for a knowledge item."""
    data = request.json
    knowledge_type = data.get('type', '')
    index = data.get('index')
    original = data.get('original', '')
    expansion = data.get('expansion', '')
    notes = data.get('notes', '')
    
    if not knowledge_type or index is None:
        return jsonify({'error': 'Missing type or index'}), 400
    
    try:
        actor_root = get_actor_root(actor_name)
        array_dir = get_array_dir()
        
        # Save expansion to a dedicated expansions file
        expansions_dir = array_dir / 'data' / actor_name / 'knowledge_expansions'
        expansions_dir.mkdir(parents=True, exist_ok=True)
        
        expansions_file = expansions_dir / f'{knowledge_type}_expansions.json'
        
        # Load existing expansions
        expansions = {}
        if expansions_file.exists():
            with open(expansions_file, 'r') as f:
                expansions = json.load(f)
        
        # Save the expansion
        key = f"{index}_{hash(original) % 10000}"
        expansions[key] = {
            'original': original,
            'expansion': expansion,
            'notes': notes,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(expansions_file, 'w') as f:
            json.dump(expansions, f, indent=2)
        
        # If there are notes, also add them as a new experience
        if notes.strip():
            experiences_path = actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml'
            note_content = f"[Note on {knowledge_type}] {notes}"
            add_to_yaml_list(experiences_path, 'experiences', note_content, 'dashboard-expansion')
        
        return jsonify({'success': True})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/knowledge/<knowledge_type>/<int:index>', methods=['DELETE'])
def api_delete_knowledge(actor_name, knowledge_type, index):
    """Delete a knowledge item by index."""
    actor_root = get_actor_root(actor_name)
    
    paths = {
        'values': actor_root / 'characteristic' / 'values.yaml',
        'principles': actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
        'experiences': actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
        'core_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
        'supporting_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
        'challenges': actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
        'adjustments': actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml'
    }
    
    if knowledge_type not in paths:
        return jsonify({'error': 'Invalid knowledge type'}), 400
    
    file_path = paths[knowledge_type]
    
    try:
        if not file_path.exists():
            return jsonify({'error': 'Knowledge file not found'}), 404
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        items = data.get(knowledge_type, [])
        
        if not isinstance(items, list):
            return jsonify({'error': 'Invalid knowledge structure'}), 500
        
        if index < 0 or index >= len(items):
            return jsonify({'error': 'Index out of range'}), 400
        
        # Remove the item
        deleted_item = items.pop(index)
        data[knowledge_type] = items
        
        # Save back
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        return jsonify({'success': True, 'deleted': deleted_item})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/sessions')
def api_actor_sessions(actor_name):
    """Get learning sessions for an actor."""
    try:
        array_dir = get_array_dir()
        sessions_dir = array_dir / 'data' / actor_name / 'learning_sessions'
        
        sessions = []
        if sessions_dir.exists():
            for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
                if session_dir.is_dir():
                    meta_file = session_dir / 'session_meta.json'
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            meta = json.load(f)
                            sessions.append({
                                'session_id': meta.get('session_id', session_dir.name),
                                'topic': meta.get('topic', 'Unknown'),
                                'created_at': meta.get('started_at', ''),
                                'papers_studied': meta.get('total_papers_studied', 0),
                                'qa_pairs': meta.get('qa_pairs_count', 0)
                            })
        
        return jsonify(sessions[:20])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/loops')
def api_actor_loops(actor_name):
    """Get cognitive loops for an actor."""
    try:
        array_dir = get_array_dir()
        loops_dir = array_dir / 'data' / actor_name / 'cognitive_loops'
        
        loops = []
        if loops_dir.exists():
            for loop_file in sorted(loops_dir.glob('loop_*.json'), reverse=True):
                try:
                    with open(loop_file, 'r') as f:
                        data = json.load(f)
                        state = data.get('state', {})
                        loops.append({
                            'loop_id': state.get('loop_id', ''),
                            'topic': state.get('topic', 'Unknown'),
                            'phase': state.get('phase', 'unknown'),
                            'iteration': state.get('iteration', 0),
                            'started_at': state.get('started_at', ''),
                            'papers_studied': state.get('total_papers_studied', 0)
                        })
                except:
                    continue
        
        return jsonify(loops[:20])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/studied-papers')
def api_actor_studied_papers(actor_name):
    """Get all studied papers and their extracted concepts for an actor."""
    import sys
    try:
        array_dir = get_array_dir()
        loops_dir = array_dir / 'data' / actor_name / 'cognitive_loops'
        
        print(f"[studied-papers] Looking for papers in: {loops_dir}", flush=True)
        print(f"[studied-papers] Directory exists: {loops_dir.exists()}", flush=True)
        sys.stdout.flush()
        
        papers = []
        files_checked = 0
        files_with_papers = 0
        
        if loops_dir.exists():
            for loop_file in sorted(loops_dir.glob('loop_*.json'), reverse=True):
                files_checked += 1
                try:
                    with open(loop_file, 'r') as f:
                        data = json.load(f)
                    
                    state = data.get('state', {})
                    loop_topic = state.get('topic', 'Unknown Topic')
                    loop_id = state.get('loop_id', loop_file.stem)
                    
                    studied_papers = state.get('studied_papers', [])
                    if studied_papers:
                        files_with_papers += 1
                        print(f"[studied-papers] {loop_file.name}: {len(studied_papers)} papers")
                    
                    # Extract studied papers with their concepts
                    for paper_entry in studied_papers:
                        paper = paper_entry.get('paper', {}) or {}
                        study_result = paper_entry.get('study_result', {}) or {}
                        
                        # Extract knowledge bits/concepts
                        extraction = study_result.get('extraction', {}) or {}
                        study_notes = study_result.get('study_notes', {}) or {}
                        
                        if extraction:
                            # New format from StudyProcessor
                            knowledge_bits = extraction.get('knowledge_bits', []) or []
                            questions = extraction.get('questions_raised', []) or []
                            summary = study_notes.get('summary', '') or ''
                            key_concepts = study_notes.get('key_concepts', []) or []
                        else:
                            # Old format
                            knowledge_bits = study_result.get('knowledge_bits', []) or []
                            questions = study_result.get('questions', []) or []
                            notes = study_result.get('notes', {}) or {}
                            summary = notes.get('summary', notes.get('paper_summary', '')) if isinstance(notes, dict) else ''
                            key_concepts = notes.get('key_concepts', []) if isinstance(notes, dict) else []
                        
                        title = paper.get('title', 'Unknown Paper')
                        # Check if PDF is cached (pass knowledge_bits for source_file matching)
                        pdf_info = find_cached_pdf(title, knowledge_bits)
                        
                        papers.append({
                            'title': title,
                            'authors': paper.get('authors', paper.get('author', '')),
                            'url': paper.get('link', paper.get('url', '')),
                            'pdf_url': paper.get('pdf_url', paper.get('pdf_link', '')),
                            'abstract': paper.get('abstract', paper.get('summary', '')),
                            'source': paper.get('source', 'unknown'),
                            'studied_at': paper_entry.get('studied_at', ''),
                            'loop_topic': loop_topic,
                            'loop_id': loop_id,
                            'summary': summary,
                            'key_concepts': key_concepts,
                            'knowledge_bits': knowledge_bits,
                            'questions_raised': questions,
                            'pdf_cached': pdf_info.get('found', False),
                            'pdf_filename': pdf_info.get('filename', ''),
                            'pdf_size_mb': pdf_info.get('size_mb', 0),
                            'pdf_match_type': pdf_info.get('match_type', '')
                        })
                except Exception as e:
                    print(f"Error loading {loop_file.name}: {e}")
                    continue
        
        print(f"[studied-papers] Checked {files_checked} files, {files_with_papers} had papers, total {len(papers)} papers found")
        
        # Also check learning_sessions for additional paper data
        sessions_dir = array_dir / 'data' / actor_name / 'learning_sessions'
        if sessions_dir.exists():
            for session_file in sorted(sessions_dir.glob('session_*.json'), reverse=True):
                try:
                    with open(session_file, 'r') as f:
                        session = json.load(f)
                    
                    session_topic = session.get('metadata', {}).get('topic', session.get('topic', 'Unknown'))
                    
                    for note in session.get('study_notes', []):
                        # Avoid duplicates by checking title
                        title = note.get('title', '')
                        if title and not any(p['title'] == title for p in papers):
                            papers.append({
                                'title': title,
                                'authors': '',
                                'url': note.get('url', ''),
                                'pdf_url': '',
                                'abstract': '',
                                'source': note.get('source_type', 'session'),
                                'studied_at': note.get('timestamp', ''),
                                'loop_topic': session_topic,
                                'loop_id': session_file.stem,
                                'summary': note.get('summary', ''),
                                'key_concepts': note.get('key_points', []),
                                'knowledge_bits': [],
                                'questions_raised': note.get('questions', [])
                            })
                except Exception as e:
                    print(f"Error loading session {session_file.name}: {e}")
                    continue
        
        return jsonify({
            'total': len(papers),
            'papers': papers
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# PDF Cache Stock API endpoints - must be BEFORE /api/pdf/<filename> route
@app.route('/api/pdf-cache/stats')
def api_pdf_cache_stats():
    """Get statistics about the PDF cache from the stock index."""
    try:
        stock = get_pdf_stock()
        
        return jsonify({
            'exists': True,
            'path': str(get_pdf_cache_dir()),
            'total_pdfs': stock.get('total_count', 0),
            'total_size_mb': stock.get('total_size_mb', 0),
            'indexed_titles': len(stock.get('title_index', {})),
            'updated_at': stock.get('updated_at')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdf-cache/stock')
def api_pdf_stock():
    """Get the full PDF stock index."""
    try:
        stock = get_pdf_stock()
        return jsonify(stock)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdf-cache/rebuild', methods=['POST'])
def api_pdf_cache_rebuild():
    """Rebuild the PDF stock index by scanning the cache directory."""
    try:
        stock = rebuild_pdf_stock()
        # Clear the in-memory cache to force reload
        _pdf_stock_cache['stock'] = stock
        _pdf_stock_cache['loaded_at'] = __import__('time').time()
        
        return jsonify({
            'success': True,
            'total_pdfs': stock.get('total_count', 0),
            'total_size_mb': stock.get('total_size_mb', 0),
            'indexed_titles': len(stock.get('title_index', {}))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdf-cache/list')
def api_pdf_cache_list():
    """List all PDFs in the cache from the stock index."""
    try:
        stock = get_pdf_stock()
        
        pdfs = []
        for filename, info in stock.get('pdfs', {}).items():
            pdfs.append({
                'filename': filename,
                'title': info.get('original_title', ''),
                'normalized_title': info.get('normalized_title', ''),
                'size_mb': info.get('size_mb', 0),
                'references': info.get('references', [])
            })
        
        # Sort by title
        pdfs.sort(key=lambda x: x.get('title', '').lower())
        
        return jsonify({
            'total': len(pdfs),
            'pdfs': pdfs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdf-cache/search')
def api_pdf_cache_search():
    """Search for a PDF by title in the stock."""
    try:
        title = request.args.get('title', '')
        if not title:
            return jsonify({'error': 'title parameter required'}), 400
        
        result = find_pdf_in_stock(title)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Serve individual PDF files
@app.route('/api/pdf/file/<filename>')
def api_serve_pdf(filename):
    """Serve a cached PDF file."""
    from flask import send_file
    try:
        pdf_cache_dir = get_pdf_cache_dir()
        pdf_path = pdf_cache_dir / filename
        
        if not pdf_path.exists():
            return jsonify({'error': 'PDF not found'}), 404
        
        # Security: ensure the path is within pdf_cache_dir
        if not str(pdf_path.resolve()).startswith(str(pdf_cache_dir.resolve())):
            return jsonify({'error': 'Invalid path'}), 403
        
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/chat', methods=['POST'])
def api_chat(actor_name):
    """Chat with an actor - lightweight cognitive mode for chat only."""
    data = request.json
    message = data.get('message', '')
    images = data.get('images', [])  # Array of {data: base64, type: mime}
    history = data.get('history', [])
    use_cognitive = data.get('use_cognitive', True)
    adapter = data.get('adapter', '')  # LoRA adapter name (empty = base model)
    
    if not message and not images:
        return jsonify({'error': 'No message or images provided'}), 400
    
    # Clean history - only keep role and content (remove extra fields like 'cognitive')
    clean_history = [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in history
        if msg.get("role") and msg.get("content")
    ]
    
    try:
        from src.utils.chat_utils import get_chat_completion
        from src.utils.knowledge_base import KnowledgeBase
        from src.thoughts.subconscious import SilvertongueSubconscious
        from src.thoughts.conscious import SilvertongueConscious
        
        # Initialize knowledge base and thought processors
        kb = KnowledgeBase(actor_name)
        subconscious = SilvertongueSubconscious(kb)
        conscious = SilvertongueConscious(kb)
        
        details = get_actor_details(actor_name)
        
        # Build system prompt from actor's knowledge - increase limits for better context
        values = details.get('values', [])[:15]
        principles = details.get('principles', [])[:20]
        experiences = details.get('experiences', [])[:15]
        core_knowledge = details.get('core_knowledge', [])[:30]
        supporting_knowledge = details.get('supporting_knowledge', [])[:15]
        recent_learnings = details.get('recent_learnings', [])[:10]
        
        if use_cognitive:
            # PROPER COGNITIVE MODE: Use subconscious -> conscious pipeline
            
            # Step 1: Subconscious analysis
            analysis = subconscious.analyze_request(message)
            
            # Step 2: Conscious processing with full context
            conscious_prompt = conscious.construct_conscious_prompt(analysis)
            
            # Step 3: Build final response with full knowledge context
            system_prompt = f"""{conscious_prompt}

You are responding to a chat message. Draw upon:
- Your principles and values
- The relevant experiences and knowledge identified by your subconscious
- Your accumulated learning

User's message: {message}

Provide a thoughtful, helpful response that reflects your knowledge and character."""

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(clean_history[-10:])
            
            # Add user message with images if present
            if images and len(images) > 0:
                user_content = []
                if message:
                    user_content.append({"type": "text", "text": message})
                for img in images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": img['data']}
                    })
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": message})
            
            # Use selected adapter or base model from config
            model_override = adapter if adapter else None
            response = get_chat_completion(messages, temperature=0.7, model_override=model_override)
            
            # Save chat history
            _save_chat_turn(actor_name, message, response)
            
            # Build detailed thought history for the UI
            thought_history = [
                {"step": "Request Analysis", "content": f"Analyzing request in domain: {analysis.get('domain', 'General')}"},
                {"step": "Intent Detection", "content": f"Detected intent: {analysis.get('intent', 'Unknown')} | Complexity: {analysis.get('complexity', 'medium')}"},
            ]
            
            # Add retrieved knowledge step
            knowledge_count = len(analysis.get('relevant_knowledge', []))
            if knowledge_count > 0:
                thought_history.append({
                    "step": "Knowledge Retrieval",
                    "content": f"Retrieved {knowledge_count} relevant knowledge items from memory"
                })
            
            # Add experiences step
            exp_count = len(analysis.get('relevant_experiences', []))
            if exp_count > 0:
                thought_history.append({
                    "step": "Experience Matching",
                    "content": f"Found {exp_count} relevant past experiences to apply"
                })
            
            # Add principles step
            principles_count = len(analysis.get('applicable_principles', []))
            if principles_count > 0:
                thought_history.append({
                    "step": "Principle Application",
                    "content": f"Applying {principles_count} guiding principles to response"
                })
            
            thought_history.append({
                "step": "Response Generation",
                "content": "Synthesizing response using retrieved context and character traits"
            })
            
            return jsonify({
                'response': response,
                'actor': actor_name,
                'cognitive': True,
                'system_prompt': system_prompt,
                'subconscious_analysis': analysis,
                'conscious_prompt': conscious_prompt,
                'analysis': analysis,
                'thought_history': thought_history
            })
        else:
            # Non-cognitive mode: Still use subconscious for context but simpler processing
            analysis = subconscious.analyze_request(message)
            
            system_prompt = f"""You are {actor_name}.

Domain: {analysis.get('domain', 'General')}

Relevant Knowledge:
{chr(10).join(f'- {k}' for k in analysis.get('relevant_knowledge', [])[:10])}

Relevant Experiences:
{chr(10).join(f'- {e}' for e in analysis.get('relevant_experiences', [])[:5])}

Principles:
{chr(10).join(f'- {p}' for p in kb.get_principles()[:10])}

Respond naturally and helpfully."""

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(clean_history[-10:])
            
            # Add user message with images if present
            if images and len(images) > 0:
                user_content = []
                if message:
                    user_content.append({"type": "text", "text": message})
                for img in images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": img['data']}
                    })
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": message})
            
            # Use selected adapter or base model from config
            model_override = adapter if adapter else None
            response = get_chat_completion(messages, temperature=0.7, model_override=model_override)
            
            # Save chat history
            _save_chat_turn(actor_name, message, response)
            
            return jsonify({
                'response': response,
                'actor': actor_name,
                'cognitive': False
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _save_chat_turn(actor_name: str, user_msg: str, assistant_msg: str):
    """Save a chat turn to history file."""
    try:
        array_dir = get_array_dir()
        chat_dir = array_dir / 'data' / actor_name / 'chat_history'
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create today's chat file
        today = datetime.now().strftime('%Y%m%d')
        chat_file = chat_dir / f'chat_{today}.json'
        
        if chat_file.exists():
            with open(chat_file, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'actor_name': actor_name,
                'date': today,
                'turns': []
            }
        
        # Add turn
        data['turns'].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'assistant': assistant_msg
        })
        
        # Save
        with open(chat_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")


@app.route('/api/actor/<actor_name>/chat/history')
def api_chat_history(actor_name):
    """Get chat history for an actor."""
    try:
        array_dir = get_array_dir()
        chat_dir = array_dir / 'data' / actor_name / 'chat_history'
        
        if not chat_dir.exists():
            return jsonify({'history': []})
        
        # Get all chat files
        chat_files = sorted(chat_dir.glob('chat_*.json'), reverse=True)
        
        # Return recent history (last 7 days)
        all_turns = []
        for chat_file in chat_files[:7]:
            with open(chat_file, 'r') as f:
                data = json.load(f)
                all_turns.extend(data.get('turns', []))
        
        return jsonify({
            'actor': actor_name,
            'total_turns': len(all_turns),
            'history': all_turns[-100:]  # Last 100 turns
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/learn', methods=['POST'])
def api_start_learning(actor_name):
    """Start a cognitive learning loop."""
    data = request.json
    topic = data.get('topic', '')
    goal = data.get('goal', '')
    mode = data.get('mode', 'standard')
    adapter = data.get('adapter', '')  # LoRA adapter to use (empty = base model)
    
    # Configuration options
    max_iterations = data.get('max_iterations', 10)
    fetch_pdfs = data.get('fetch_pdfs', True)
    save_session = data.get('save_session', True)
    auto_save = data.get('auto_save', True)
    max_papers_per_search = data.get('max_papers_per_search', 15)
    max_papers_to_study = data.get('max_papers_to_study', 3)
    search_depth = data.get('search_depth', 2)
    min_citations = data.get('min_citations', 10)
    year_from = data.get('year_from', 2018)
    year_to = data.get('year_to', None)
    goal_threshold = data.get('goal_threshold', 85)
    
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400
    
    if not goal:
        goal = f"Develop comprehensive understanding of {topic}"
    
    # Create output queue for this loop
    loop_id = f"{actor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    loop_outputs[loop_id] = queue.Queue()
    
    def run_loop():
        try:
            # Lazy import
            from src.actions.cognitive_loop import CognitiveLoop
            
            loop = CognitiveLoop(actor_name, verbose=True)
            loop.configure(
                fetch_pdfs=fetch_pdfs,
                save_session=save_session,
                auto_save=auto_save,
                max_papers_per_search=max_papers_per_search,
                max_papers_to_study=max_papers_to_study,
                search_depth=search_depth,
                min_citations=min_citations,
                year_from=year_from,
                year_to=year_to,
                model_override=adapter if adapter else None  # LoRA adapter support
            )
            
            # Custom logging to capture output
            original_log = loop._log
            def custom_log(msg):
                loop_outputs[loop_id].put(msg)
                original_log(msg)
            loop._log = custom_log
            
            active_loops[loop_id] = loop
            
            # Adjust iterations based on mode
            if mode == 'nonstop':
                # Nonstop mode: truly unlimited - no goal threshold, just keep going
                result = loop.learn_nonstop(
                    topic, goal, 
                    goal_threshold=100,  # Never stop based on goal (would need 100% which never happens)
                    max_iterations=999999,  # Essentially unlimited
                    checkpoint_interval=1  # Save state after every iteration
                )
            elif mode == 'quick':
                result = loop.learn(topic, goal, max_iterations=1)
            elif mode == 'fast':
                result = loop.learn(topic, goal, max_iterations=min(max_iterations, 3))
            else:
                result = loop.learn(topic, goal, max_iterations)
            
            loop_outputs[loop_id].put(f"\n✅ COMPLETE: {json.dumps(result.get('statistics', {}))}")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            loop_outputs[loop_id].put(f"\n❌ ERROR: {str(e)}")
            loop_outputs[loop_id].put(f"\n📋 Traceback:\n{error_trace}")
        finally:
            if loop_id in active_loops:
                del active_loops[loop_id]
    
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    
    return jsonify({
        'loop_id': loop_id,
        'status': 'started',
        'topic': topic,
        'goal': goal,
        'mode': mode
    })


@app.route('/api/loop/<loop_id>/status')
def api_loop_status(loop_id):
    """Get status of a running loop."""
    if loop_id in active_loops:
        loop = active_loops[loop_id]
        state = loop.get_state()
        return jsonify({
            'running': True,
            'phase': state.phase.value if state else 'unknown',
            'iteration': state.iteration if state else 0,
            'papers_studied': state.total_papers_studied if state else 0,
            'knowledge_bits': state.total_knowledge_bits if state else 0
        })
    return jsonify({'running': False})


@app.route('/api/tasks/active')
def api_active_tasks():
    """Get all active learning tasks/loops with their status."""
    tasks = []
    for loop_id, loop in active_loops.items():
        try:
            state = loop.get_state()
            # Parse loop_id: actorname_timestamp
            parts = loop_id.split('_')
            actor = parts[0] if parts else 'unknown'
            timestamp = '_'.join(parts[1:]) if len(parts) > 1 else ''
            
            tasks.append({
                'loop_id': loop_id,
                'actor': actor,
                'started_at': timestamp,
                'topic': state.topic if state else 'unknown',
                'goal': state.goal if state else 'unknown',
                'phase': state.phase.value if state else 'unknown',
                'iteration': state.iteration if state else 0,
                'papers_studied': state.total_papers_studied if state else 0,
                'knowledge_bits': state.total_knowledge_bits if state else 0,
                'status': 'running'
            })
        except Exception as e:
            tasks.append({
                'loop_id': loop_id,
                'error': str(e),
                'status': 'error'
            })
    
    return jsonify({
        'active_count': len(tasks),
        'tasks': tasks
    })


@app.route('/api/tasks/stop/<loop_id>', methods=['POST'])
def api_stop_task(loop_id):
    """Stop a running task gracefully."""
    if loop_id in active_loops:
        try:
            loop = active_loops[loop_id]
            loop.pause()  # Gracefully pause the loop
            return jsonify({'success': True, 'message': 'Task paused gracefully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False, 'error': 'Task not found'}), 404


@app.route('/api/loop/<loop_id>/output')
def api_loop_output(loop_id):
    """Stream output from a running loop."""
    def generate():
        if loop_id not in loop_outputs:
            yield f"data: {json.dumps({'error': 'Loop not found'})}\n\n"
            return
        
        q = loop_outputs[loop_id]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps({'message': msg})}\n\n"
                if '✅ COMPLETE' in msg or '❌ ERROR' in msg:
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


def _save_chat_turn(actor_name: str, user_msg: str, assistant_msg: str):
    """Save a chat turn to history file."""
    try:
        array_dir = get_array_dir()
        chat_dir = array_dir / 'data' / actor_name / 'chat_history'
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create today's chat file
        today = datetime.now().strftime('%Y%m%d')
        chat_file = chat_dir / f'chat_{today}.json'
        
        if chat_file.exists():
            with open(chat_file, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'actor_name': actor_name,
                'date': today,
                'turns': []
            }
        
        # Add turn
        data['turns'].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'assistant': assistant_msg
        })
        
        # Save
        with open(chat_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")


@app.route('/api/config')
def api_config():
    """Get current configuration."""
    config = load_config()
    array_dir = get_array_dir()
    return jsonify({
        'apiHost': config.get('apiHost', ''),
        'model': config.get('model', ''),
        'array_dir': str(array_dir),
        'array_exists': array_dir.exists()
    })


@app.route('/api/config', methods=['POST'])
def api_update_config():
    """Update configuration."""
    data = request.json
    config = load_config()
    
    # Update allowed fields
    if 'array_dir' in data:
        config['array_dir'] = data['array_dir']
    if 'apiHost' in data:
        config['apiHost'] = data['apiHost']
    if 'model' in data:
        config['model'] = data['model']
    
    # Save config
    config_path = project_root / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return jsonify({'success': True, 'config': config})


@app.route('/api/actor/create', methods=['POST'])
def api_create_actor():
    """Create a new actor from a description using AI generation."""
    data = request.json
    actor_name = data.get('name', '').strip().lower().replace(' ', '_')
    description = data.get('description', '').strip()
    preview_only = data.get('preview', False)
    
    if not actor_name:
        return jsonify({'error': 'Actor name is required'}), 400
    
    if not description:
        return jsonify({'error': 'Actor description is required'}), 400
    
    # Check if actor already exists (unless just previewing)
    array_dir = get_array_dir()
    actor_dir = array_dir / 'actors' / actor_name
    if not preview_only and actor_dir.exists():
        return jsonify({'error': f'Actor "{actor_name}" already exists'}), 400
    
    try:
        from src.utils.actor_generator import generate_actor_profile, save_actor_profile
        
        # Generate profile from description
        profile = generate_actor_profile(description)
        
        if "raw_response" in profile:
            return jsonify({'error': 'Failed to generate actor profile', 'details': profile['raw_response']}), 500
        
        # If only preview, return the profile without saving
        if preview_only:
            return jsonify({
                'success': True,
                'profile': profile,
                'message': 'Profile generated (preview only)'
            })
        
        # Save the actor
        save_actor_profile(profile, actor_name, project_root)
        
        return jsonify({
            'success': True,
            'actor': actor_name,
            'profile': profile,
            'message': f'Actor "{actor_name}" created successfully'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/create/manual', methods=['POST'])
def api_create_actor_manual():
    """Create a new actor manually with provided profile data."""
    data = request.json
    actor_name = data.get('name', '').strip().lower().replace(' ', '_')
    profile = data.get('profile', {})
    
    if not actor_name:
        return jsonify({'error': 'Actor name is required'}), 400
    
    # Check if actor already exists
    array_dir = get_array_dir()
    actor_dir = array_dir / 'actors' / actor_name
    if actor_dir.exists():
        return jsonify({'error': f'Actor "{actor_name}" already exists'}), 400
    
    try:
        from src.utils.actor_generator import save_actor_profile
        
        # Helper to convert string to list (split by newlines)
        def to_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val.strip():
                return [line.strip() for line in val.strip().split('\n') if line.strip()]
            return []
        
        # Ensure profile has required keys with defaults
        default_profile = {
            'values': to_list(profile.get('values', [])),
            'principles': to_list(profile.get('principles', [])),
            'core_knowledge': to_list(profile.get('core_knowledge', [])),
            'supporting_knowledge': to_list(profile.get('supporting_knowledge', [])),
            'experiences': to_list(profile.get('experiences', [])),
            'adjustments': to_list(profile.get('adjustments', [])),
            'challenges': to_list(profile.get('challenges', []))
        }
        
        # Save the actor
        save_actor_profile(default_profile, actor_name, project_root)
        
        return jsonify({
            'success': True,
            'actor': actor_name,
            'message': f'Actor "{actor_name}" created successfully'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/sleep', methods=['POST'])
def api_actor_sleep(actor_name):
    """
    Put an actor to sleep - consolidate memories into LoRA training data.
    
    This processes all of the actor's accumulated knowledge, experiences,
    and learning sessions into training data that can be used to fine-tune
    a model to embody the actor's unique mind.
    
    Request body:
        deep: bool - If True, condense ALL items (deep sleep). 
                     If False (default), only condense NEW items since last sleep.
        adapter: str - LoRA adapter to use for LLM calls (empty = base model)
    """
    try:
        from src.actions.sleep_processor import SleepProcessor
        
        data = request.json or {}
        deep_sleep = data.get('deep', False)  # Default to light sleep (incremental)
        adapter = data.get('adapter', '')  # LoRA adapter to use for LLM calls
        
        # Use verbose=True so we can see consolidation logs
        processor = SleepProcessor(actor_name, verbose=True, model_override=adapter if adapter else None)
        
        # Run sleep cycle - deep or light based on parameter
        if deep_sleep:
            result = processor.deep_sleep()
            mode = "deep"
        else:
            result = processor.sleep()
            mode = "light"
        
        return jsonify({
            'success': True,
            'actor': actor_name,
            'mode': mode,
            'summary': result,
            'message': f'{mode.capitalize()} sleep complete! Generated {result["total_examples"]} training examples.'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/sleep/status', methods=['GET'])
def api_actor_sleep_status(actor_name):
    """Get the status of LoRA training data for an actor."""
    try:
        array_dir = get_array_dir()
        lora_dir = array_dir / 'data' / actor_name / 'lora_training'
        
        if not lora_dir.exists():
            return jsonify({
                'has_data': False,
                'message': 'No sleep data yet. Put the actor to sleep to generate LoRA training data.'
            })
        
        # Find dataset files
        dataset_files = list(lora_dir.glob('*.jsonl'))
        summary_files = list(lora_dir.glob('*.summary.json'))
        
        result = {
            'has_data': len(dataset_files) > 0,
            'datasets': []
        }
        
        for dataset_file in dataset_files:
            # Count lines (examples)
            with open(dataset_file, 'r') as f:
                example_count = sum(1 for _ in f)
            
            # Load summary if exists
            summary_file = dataset_file.with_suffix('.summary.json')
            summary = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            
            result['datasets'].append({
                'filename': dataset_file.name,
                'path': str(dataset_file),
                'examples': example_count,
                'generated_at': summary.get('generated_at', 'Unknown'),
                'by_category': summary.get('by_category', {}),
                'recommended_params': summary.get('recommended_training_params', {})
            })
        
        # Sort datasets by generated_at (newest first), fallback to filename
        result['datasets'].sort(
            key=lambda d: d.get('generated_at', ''),
            reverse=True
        )
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/datasets/merge', methods=['POST'])
def api_merge_datasets(actor_name):
    """
    Merge multiple LoRA datasets from different learning sessions.
    
    Request body:
    {
        "dataset_files": ["file1.jsonl", "file2.jsonl"],  // Filenames to merge
        "output_name": "merged_dataset.jsonl"  // Optional, auto-generates if not provided
    }
    """
    try:
        from src.actions.sleep_processor import merge_datasets
        
        data = request.json or {}
        dataset_files = data.get('dataset_files', [])
        output_name = data.get('output_name', None)
        
        if not dataset_files or len(dataset_files) < 2:
            return jsonify({
                'error': 'Must provide at least 2 datasets to merge'
            }), 400
        
        # Build full paths
        array_dir = get_array_dir()
        lora_dir = array_dir / 'data' / actor_name / 'lora_training'
        dataset_paths = [str(lora_dir / filename) for filename in dataset_files]
        
        # Merge datasets
        result = merge_datasets(actor_name, dataset_paths, output_name)
        
        return jsonify({
            'success': True,
            'actor': actor_name,
            'result': result,
            'message': f'Merged {result["sources_merged"]} datasets into {result["total_examples"]} examples'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Training Server Integration ====================

def get_trainer_config():
    """Get trainer server configuration."""
    config = load_config()
    return config.get('trainer_server', {
        'host': 'localhost',
        'port': 5050,
        'enabled': False
    })


def get_trainer_url():
    """Get the trainer server URL."""
    trainer = get_trainer_config()
    return f"http://{trainer['host']}:{trainer['port']}"


def translate_path_for_server(local_path: str) -> str:
    """
    Translate a local Mac path to the server's view inside the Docker container.
    
    The launch script mounts:
      -v "$LORA_OUTPUT_DIR":/lora_adapters
      -v "$DATASET_DIR":/datasets
    
    Where:
      LORA_OUTPUT_DIR=/mnt/silvertongueArray/lora_adapters
      DATASET_DIR=/mnt/silvertongueArray/array/data
    
    Mac paths need to be translated:
      /Users/ruihe/mnt/silvertongueArray/array/data/... -> /datasets/...
      /Users/ruihe/mnt/silvertongueArray/lora_adapters/... -> /lora_adapters/...
    """
    # Get the local array_dir from config
    config = load_config()
    local_array_dir = config.get('array_dir', '')
    
    # Mac mount points
    mac_array_mount = '/Users/ruihe/mnt/silvertongueArray/array'
    mac_lora_mount = '/Users/ruihe/mnt/silvertongueArray/lora_adapters'
    
    # DGX host paths
    dgx_array_mount = '/mnt/silvertongueArray/array'
    dgx_lora_mount = '/mnt/silvertongueArray/lora_adapters'
    
    # Docker container paths
    docker_data_dir = '/datasets'
    docker_lora_dir = '/lora_adapters'
    
    server_path = local_path
    
    # Handle LoRA adapter paths FIRST (more specific)
    if local_path.startswith(mac_lora_mount):
        # Mac: /Users/ruihe/mnt/silvertongueArray/lora_adapters/steve/v1
        # Docker: /lora_adapters/steve/v1
        server_path = local_path.replace(mac_lora_mount, docker_lora_dir)
        return server_path
    elif local_path.startswith(dgx_lora_mount):
        # DGX: /mnt/silvertongueArray/lora_adapters/steve/v1
        # Docker: /lora_adapters/steve/v1
        server_path = local_path.replace(dgx_lora_mount, docker_lora_dir)
        return server_path
    
    # Handle dataset/array paths
    if local_array_dir and local_path.startswith(local_array_dir):
        server_path = local_path.replace(local_array_dir, dgx_array_mount)
    elif local_path.startswith(mac_array_mount):
        server_path = local_path.replace(mac_array_mount, dgx_array_mount)
    
    # Now convert DGX host path to Docker container path
    # The DATASET_DIR (/mnt/silvertongueArray/array/data) is mounted as /datasets
    dgx_data_dir = '/mnt/silvertongueArray/array/data'
    
    if server_path.startswith(dgx_data_dir):
        server_path = server_path.replace(dgx_data_dir, docker_data_dir)
    
    return server_path


@app.route('/api/trainer/status', methods=['GET'])
def api_trainer_status():
    """Check if the remote trainer server is available."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({
            'enabled': False,
            'message': 'Remote trainer is not enabled in config.json'
        })
    
    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/health", timeout=5)
        if response.ok:
            health_data = response.json()
            
            # Also fetch config to get available models
            available_models = []
            try:
                config_response = requests.get(f"{get_trainer_url()}/api/config", timeout=5)
                if config_response.ok:
                    config_data = config_response.json()
                    available_models = config_data.get('available_models', [])
            except:
                pass  # Non-critical, just won't show model options
            
            return jsonify({
                'enabled': True,
                'connected': True,
                'base_model': health_data.get('base_model', 'Unknown'),
                'available_models': available_models,
                'server': health_data,
                'url': get_trainer_url()
            })
    except Exception as e:
        return jsonify({
            'enabled': True,
            'connected': False,
            'error': str(e),
            'url': get_trainer_url()
        })


@app.route('/api/trainer/config', methods=['GET'])
def api_trainer_config():
    """Get trainer server configuration."""
    return jsonify(get_trainer_config())


@app.route('/api/trainer/config', methods=['PUT'])
def api_update_trainer_config():
    """Update trainer server configuration."""
    data = request.json
    config = load_config()
    
    trainer = config.get('trainer_server', {})
    if 'host' in data:
        trainer['host'] = data['host']
    if 'port' in data:
        trainer['port'] = data['port']
    if 'enabled' in data:
        trainer['enabled'] = data['enabled']
    
    config['trainer_server'] = trainer
    
    config_path = project_root / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return jsonify({'success': True, 'trainer_server': trainer})


@app.route('/api/actor/<actor_name>/train', methods=['POST'])
def api_start_training(actor_name):
    """Submit a LoRA training job to the remote trainer server."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({
            'error': 'Remote trainer is not enabled',
            'hint': 'Enable trainer_server in config.json or set up DGX Spark'
        }), 400
    
    # Check if training data exists - look for various dataset patterns
    array_dir = get_array_dir()
    lora_dir = array_dir / 'data' / actor_name / 'lora_training'
    
    # Find the best dataset file (prefer merged, then latest timestamped, then basic)
    train_file = None
    
    # Check request for specific dataset
    data = request.json or {}
    requested_dataset = data.get('dataset') or data.get('dataset_file')  # Support both field names
    requested_base_model = data.get('base_model')  # Custom base model selection
    
    if requested_dataset:
        train_file = lora_dir / requested_dataset
    else:
        # Auto-find best dataset
        # 1. Look for merged datasets (most comprehensive)
        merged_files = sorted(lora_dir.glob(f'{actor_name}_merged_*.jsonl'), reverse=True)
        if merged_files:
            train_file = merged_files[0]
        else:
            # 2. Look for timestamped datasets
            timestamped = sorted(lora_dir.glob(f'{actor_name}_lora_dataset_*.jsonl'), reverse=True)
            if timestamped:
                train_file = timestamped[0]
            else:
                # 3. Look for basic dataset
                basic = lora_dir / f"{actor_name}_lora_dataset.jsonl"
                if basic.exists():
                    train_file = basic
                else:
                    # 4. Fallback to old naming
                    old_style = lora_dir / f"{actor_name}_lora_train.jsonl"
                    if old_style.exists():
                        train_file = old_style
    
    if not train_file or not train_file.exists():
        return jsonify({
            'error': 'No training data found',
            'hint': 'Run the sleep cycle first to generate training data',
            'searched_dir': str(lora_dir)
        }), 404
    
    # Translate local path to server path
    server_dataset_path = translate_path_for_server(str(train_file))
    
    # Handle base_adapter for LoRA-on-LoRA training
    requested_base_adapter = data.get('base_adapter')
    server_base_adapter = None
    if requested_base_adapter:
        # Translate the adapter path for the server
        server_base_adapter = translate_path_for_server(requested_base_adapter)
        print(f"[TRAIN] LoRA-on-LoRA requested:")
        print(f"  Local adapter path: {requested_base_adapter}")
        print(f"  Server adapter path: {server_base_adapter}")
    
    # Build training config - use server's expected field names (flat structure)
    training_config = {
        'actor_name': actor_name,
        'train_file': server_dataset_path,  # Server expects 'train_file'
        'base_model': requested_base_model,  # Custom base model (None = server default)
        'base_adapter': server_base_adapter,  # LoRA-on-LoRA: existing adapter to build upon
        'lora_r': data.get('lora_r', 32),
        'lora_alpha': data.get('lora_alpha', 64),
        'lora_dropout': data.get('lora_dropout', 0.05),
        'learning_rate': data.get('learning_rate', 2e-4),
        'num_epochs': data.get('num_epochs', 3),
        'batch_size': data.get('batch_size', 4),
        'gradient_accumulation_steps': data.get('gradient_accumulation_steps', 4),
        'max_seq_length': data.get('max_seq_length', 2048),
    }
    
    # Remove None values so server uses its defaults
    training_config = {k: v for k, v in training_config.items() if v is not None}
    
    try:
        import requests
        response = requests.post(
            f"{get_trainer_url()}/api/train",
            json=training_config,
            timeout=30
        )
        
        if response.ok:
            result = response.json()
            return jsonify({
                'success': True,
                'job_id': result.get('job_id'),
                'status': 'started',
                'message': f'Training job submitted for {actor_name}',
                'dataset': str(train_file.name),
                'config': training_config
            })
        else:
            return jsonify({
                'error': 'Failed to submit training job',
                'details': response.text
            }), response.status_code
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to trainer server',
            'hint': f'Make sure the trainer server is running at {get_trainer_url()}'
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Backward compatibility: alias without /api prefix
@app.route('/actor/<actor_name>/train', methods=['POST'])
def start_training_compat(actor_name):
    """Backward-compatible alias for /api/actor/<actor_name>/train."""
    return api_start_training(actor_name)


@app.route('/api/trainer/jobs', methods=['GET'])
def api_trainer_jobs():
    """Get list of training jobs from the remote server."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({'jobs': [], 'enabled': False})
    
    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/api/jobs", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'jobs': [], 'error': response.text})
    except Exception as e:
        return jsonify({'jobs': [], 'error': str(e)})


@app.route('/api/trainer/jobs/<job_id>', methods=['GET'])
def api_trainer_job_status(job_id):
    """Get status of a specific training job."""
    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/api/jobs/{job_id}", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'error': 'Job not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainer/models', methods=['GET'])
def api_trainer_models():
    """Get list of trained models/adapters from the remote server."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({'models': [], 'enabled': False})
    
    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/api/models", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'models': [], 'error': response.text})
    except Exception as e:
        return jsonify({'models': [], 'error': str(e)})


@app.route('/api/adapters', methods=['GET'])
def api_local_adapters():
    """
    Get list of LoRA adapters from the local filesystem (NAS mount).
    
    Structure:
    lora_adapters/
    ├── adapter_book.json       # Nicknames and metadata
    ├── actor_name/
    │   ├── v1/                 # Version folders
    │   │   ├── adapter_config.json
    │   │   └── training_info.json
    │   └── v2/
    │       └── ...
    """
    lora_dir = get_lora_adapters_dir()
    
    if not lora_dir.exists():
        return jsonify({
            'adapters': [],
            'book': {},
            'error': f'LoRA adapters directory not found: {lora_dir}'
        })
    
    # Load adapter book
    book_path = lora_dir / 'adapter_book.json'
    book = {}
    if book_path.exists():
        try:
            with open(book_path, 'r') as f:
                book = json.load(f)
        except Exception:
            book = {}
    
    adapters = []
    
    for actor_dir in lora_dir.iterdir():
        if not actor_dir.is_dir() or actor_dir.name.startswith('.'):
            continue
        if actor_dir.name == 'adapter_book.json':
            continue
            
        actor_name = actor_dir.name
        
        # Check for versioned structure (v1, v2, etc.) or flat structure
        versions = []
        has_versions = False
        
        for item in actor_dir.iterdir():
            if item.is_dir() and item.name.startswith('v'):
                # Versioned structure
                has_versions = True
                adapter_config = item / 'adapter_config.json'
                training_info = item / 'training_info.json'
                
                if adapter_config.exists():
                    version_data = _load_adapter_version(
                        item, actor_name, item.name, book)
                    versions.append(version_data)
        
        if has_versions:
            # Sort versions (v1, v2, etc.)
            versions.sort(key=lambda x: x['version'])
            
            # Get book entry for this actor
            book_entry = book.get(actor_name, {})
            
            adapters.append({
                'name': actor_name,
                'nickname': book_entry.get('nickname', actor_name),
                'description': book_entry.get('description', ''),
                'path': str(actor_dir),
                'versions': versions,
                'latest_version': versions[-1] if versions else None,
                'version_count': len(versions)
            })
        else:
            # Flat structure (legacy) - treat as v1
            adapter_config = actor_dir / 'adapter_config.json'
            if adapter_config.exists():
                version_data = _load_adapter_version(
                    actor_dir, actor_name, 'v1', book)
                book_entry = book.get(actor_name, {})
                
                adapters.append({
                    'name': actor_name,
                    'nickname': book_entry.get('nickname', actor_name),
                    'description': book_entry.get('description', ''),
                    'path': str(actor_dir),
                    'versions': [version_data],
                    'latest_version': version_data,
                    'version_count': 1,
                    'is_legacy': True  # Flat structure, not versioned
                })
    
    # Sort by name
    adapters.sort(key=lambda x: x['name'])
    
    return jsonify({
        'adapters': adapters,
        'book': book,
        'path': str(lora_dir)
    })


def _load_adapter_version(version_dir, actor_name, version, book):
    """Load metadata for a specific adapter version."""
    adapter_config = version_dir / 'adapter_config.json'
    training_info = version_dir / 'training_info.json'
    
    version_data = {
        'version': version,
        'path': str(version_dir),
        'is_valid': adapter_config.exists(),
        'has_training_info': training_info.exists()
    }
    
    # Load training info if available
    if training_info.exists():
        try:
            with open(training_info, 'r') as f:
                info = json.load(f)
                version_data['actor_name'] = info.get(
                    'actor', info.get('actor_name', actor_name))
                version_data['created_at'] = info.get(
                    'trained_at', info.get('created_at'))
                version_data['base_model'] = info.get('base_model')
                version_data['dataset'] = info.get('dataset')
                version_data['examples'] = info.get('examples')
                version_data['config'] = info.get('config', {})
        except Exception:
            version_data['actor_name'] = actor_name
    else:
        version_data['actor_name'] = actor_name
    
    # Get modification time
    try:
        version_data['modified'] = datetime.fromtimestamp(
            version_dir.stat().st_mtime
        ).isoformat()
    except Exception:
        pass
    
    # Check for version-specific nickname in book
    book_entry = book.get(actor_name, {})
    version_nicknames = book_entry.get('version_nicknames', {})
    if version in version_nicknames:
        version_data['nickname'] = version_nicknames[version]
    
    return version_data


@app.route('/api/adapters/book', methods=['GET'])
def api_get_adapter_book():
    """Get the adapter book with all nicknames and metadata."""
    lora_dir = get_lora_adapters_dir()
    book_path = lora_dir / 'adapter_book.json'
    
    if not book_path.exists():
        return jsonify({'book': {}})
    
    try:
        with open(book_path, 'r') as f:
            book = json.load(f)
        return jsonify({'book': book})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/adapters/book', methods=['PUT'])
def api_update_adapter_book():
    """Update the adapter book (nicknames, descriptions, etc.)."""
    lora_dir = get_lora_adapters_dir()
    book_path = lora_dir / 'adapter_book.json'
    
    data = request.json
    
    # Load existing book
    book = {}
    if book_path.exists():
        try:
            with open(book_path, 'r') as f:
                book = json.load(f)
        except Exception:
            book = {}
    
    # Update with new data
    adapter_name = data.get('adapter_name')
    if adapter_name:
        if adapter_name not in book:
            book[adapter_name] = {}
        
        if 'nickname' in data:
            book[adapter_name]['nickname'] = data['nickname']
        if 'description' in data:
            book[adapter_name]['description'] = data['description']
        if 'version_nickname' in data:
            version = data.get('version', 'v1')
            if 'version_nicknames' not in book[adapter_name]:
                book[adapter_name]['version_nicknames'] = {}
            book[adapter_name]['version_nicknames'][version] = data['version_nickname']
        
        book[adapter_name]['updated_at'] = datetime.now().isoformat()
    
    # Save book
    try:
        with open(book_path, 'w') as f:
            json.dump(book, f, indent=2)
        return jsonify({'success': True, 'book': book})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/adapters/<adapter_name>/rename', methods=['POST'])
def api_rename_adapter(adapter_name):
    """Rename a LoRA adapter in the local filesystem."""
    import re
    import shutil
    
    lora_dir = get_lora_adapters_dir()
    adapter_path = lora_dir / adapter_name
    
    if not adapter_path.exists():
        return jsonify({'error': f'Adapter not found: {adapter_name}'}), 404
    
    data = request.json
    new_name = data.get('new_name', '').strip()
    
    if not new_name:
        return jsonify({'error': 'new_name is required'}), 400
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', new_name):
        return jsonify({
            'error': 'Invalid name. Use only letters, numbers, '
                     'underscores, and hyphens.'
        }), 400
    
    new_path = lora_dir / new_name
    if new_path.exists():
        return jsonify({'error': f'Adapter already exists: {new_name}'}), 400
    
    try:
        shutil.move(str(adapter_path), str(new_path))
        
        # Update the book to reflect the rename
        book_path = lora_dir / 'adapter_book.json'
        if book_path.exists():
            with open(book_path, 'r') as f:
                book = json.load(f)
            if adapter_name in book:
                book[new_name] = book.pop(adapter_name)
                book[new_name]['renamed_from'] = adapter_name
                book[new_name]['renamed_at'] = datetime.now().isoformat()
                with open(book_path, 'w') as f:
                    json.dump(book, f, indent=2)
        
        return jsonify({
            'success': True,
            'old_name': adapter_name,
            'new_name': new_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/adapters/<adapter_name>/<version>', methods=['DELETE'])
def api_delete_adapter_version(adapter_name, version):
    """Delete a specific version of a LoRA adapter."""
    import shutil
    
    lora_dir = get_lora_adapters_dir()
    version_path = lora_dir / adapter_name / version
    
    if not version_path.exists():
        return jsonify({
            'error': f'Version not found: {adapter_name}/{version}'
        }), 404
    
    try:
        shutil.rmtree(str(version_path))
        
        # Check if adapter dir is now empty, if so remove it
        adapter_path = lora_dir / adapter_name
        remaining = list(adapter_path.iterdir())
        if not remaining:
            adapter_path.rmdir()
        
        return jsonify({
            'success': True,
            'deleted': f'{adapter_name}/{version}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/adapters/<adapter_name>', methods=['DELETE'])
def api_delete_adapter(adapter_name):
    """Delete a LoRA adapter (all versions) from the local filesystem."""
    import shutil
    
    lora_dir = get_lora_adapters_dir()
    adapter_path = lora_dir / adapter_name
    
    if not adapter_path.exists():
        return jsonify({'error': f'Adapter not found: {adapter_name}'}), 404
    
    try:
        shutil.rmtree(str(adapter_path))
        
        # Remove from book if present
        book_path = lora_dir / 'adapter_book.json'
        if book_path.exists():
            with open(book_path, 'r') as f:
                book = json.load(f)
            if adapter_name in book:
                del book[adapter_name]
                with open(book_path, 'w') as f:
                    json.dump(book, f, indent=2)
        
        return jsonify({
            'success': True,
            'deleted': adapter_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainer/models/<model_name>/rename', methods=['POST'])
def api_trainer_rename_model(model_name):
    """Rename a trained LoRA adapter on the remote server."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({'error': 'Trainer not enabled'}), 400
    
    data = request.json
    new_name = data.get('new_name', '').strip()
    
    if not new_name:
        return jsonify({'error': 'new_name is required'}), 400
    
    # Validate new name (alphanumeric, underscore, hyphen only)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', new_name):
        return jsonify({'error': 'Invalid name. Use only letters, numbers, underscores, and hyphens.'}), 400
    
    try:
        import requests
        response = requests.post(
            f"{get_trainer_url()}/api/models/{model_name}/rename",
            json={'new_name': new_name},
            timeout=30
        )
        if response.ok:
            return jsonify(response.json())
        return jsonify({'error': response.json().get('error', 'Rename failed')}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainer/models/<model_name>', methods=['DELETE'])
def api_trainer_delete_model(model_name):
    """Delete a trained LoRA adapter on the remote server."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({'error': 'Trainer not enabled'}), 400
    
    try:
        import requests
        response = requests.delete(
            f"{get_trainer_url()}/api/models/{model_name}",
            timeout=30
        )
        if response.ok:
            return jsonify(response.json())
        return jsonify({'error': response.json().get('error', 'Delete failed')}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5050, threaded=True)
