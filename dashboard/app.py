"""
Lingkui Dashboard - Web interface for managing actors and cognitive loops.

Reads configuration from config.json including array_dir path.
"""

import sys
from pathlib import Path
import json
import yaml
from datetime import datetime
import threading
import queue

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify, request, Response

app = Flask(__name__)

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
    return render_template('index.html', actors=actors)


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


@app.route('/api/actor/<actor_name>/chat', methods=['POST'])
def api_chat(actor_name):
    """Chat with an actor - lightweight cognitive mode for chat only."""
    data = request.json
    message = data.get('message', '')
    history = data.get('history', [])
    use_cognitive = data.get('use_cognitive', True)
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Clean history - only keep role and content (remove extra fields like 'cognitive')
    clean_history = [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in history
        if msg.get("role") and msg.get("content")
    ]
    
    try:
        from src.utils.chat_utils import get_chat_completion
        
        details = get_actor_details(actor_name)
        
        # Build system prompt from actor's knowledge (limited for chat efficiency)
        values = details.get('values', [])[:10]
        principles = details.get('principles', [])[:10]
        experiences = details.get('experiences', [])[:10]
        core_knowledge = details.get('core_knowledge', [])[:10]
        
        if use_cognitive:
            # Lightweight cognitive mode - quick routing + selective knowledge
            # Step 1: Quick route to understand the request
            route_prompt = """Analyze this request and return JSON:
{"domain": "topic area", "intent": "what user wants", "needs_knowledge": true/false, "complexity": "simple/moderate/complex"}
Return ONLY valid JSON."""
            
            route_messages = [
                {"role": "system", "content": "You are a fast request router. Return only JSON."},
                {"role": "user", "content": f"{route_prompt}\n\nRequest: {message}"}
            ]
            
            try:
                route_response = get_chat_completion(route_messages, temperature=0.1)
                import json
                routing = json.loads(route_response.replace("```json", "").replace("```", "").strip())
            except:
                routing = {"domain": "general", "intent": "chat", "needs_knowledge": False, "complexity": "simple"}
            
            # Step 2: Build context-aware prompt
            system_prompt = f"""You are {actor_name}, an AI assistant.

DOMAIN: {routing.get('domain', 'General')}

VALUES:
{chr(10).join(f'- {v}' for v in values) if values else '- Be helpful'}

PRINCIPLES:
{chr(10).join(f'- {p}' for p in principles) if principles else '- Think clearly'}"""

            # Only add knowledge if routing says it's needed
            if routing.get('needs_knowledge', False) and core_knowledge:
                # Simple keyword search for relevant knowledge
                relevant = []
                msg_lower = message.lower()
                for k in core_knowledge:
                    if any(word in k.lower() for word in msg_lower.split() if len(word) > 3):
                        relevant.append(k[:200])
                if relevant:
                    system_prompt += f"\n\nRELEVANT KNOWLEDGE:\n" + "\n".join(f"- {k}" for k in relevant[:5])

            if experiences:
                system_prompt += f"\n\nEXPERIENCES:\n" + "\n".join(f'- {e[:150]}' for e in experiences[:5])

            system_prompt += "\n\nRespond helpfully while staying true to your values and principles."

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(clean_history[-10:])
            messages.append({"role": "user", "content": message})
            
            response = get_chat_completion(messages, temperature=0.7)
            
            return jsonify({
                'response': response,
                'actor': actor_name,
                'cognitive': True,
                'analysis': routing,
                'thought_history': [{"step": "Quick Route", "content": f"Domain: {routing.get('domain')}, Intent: {routing.get('intent')}"}]
            })
        else:
            # Simple direct chat (no cognitive processing)
            system_prompt = f"""You are {actor_name}, an AI assistant with the following characteristics:

VALUES:
{chr(10).join(f'- {v}' for v in values[:5]) if values else '- Be helpful and accurate'}

PRINCIPLES:
{chr(10).join(f'- {p}' for p in principles[:5]) if principles else '- Think step by step'}

Respond naturally while staying true to these characteristics."""

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(clean_history[-10:])
            messages.append({"role": "user", "content": message})
            
            response = get_chat_completion(messages, temperature=0.7)
            
            return jsonify({
                'response': response,
                'actor': actor_name,
                'cognitive': False
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/learn', methods=['POST'])
def api_start_learning(actor_name):
    """Start a cognitive learning loop."""
    data = request.json
    topic = data.get('topic', '')
    goal = data.get('goal', '')
    mode = data.get('mode', 'standard')
    
    # Configuration options
    max_iterations = data.get('max_iterations', 2)
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
                year_to=year_to
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
                # Nonstop mode: run until goal achieved or max iterations
                result = loop.learn_nonstop(
                    topic, goal, 
                    goal_threshold=goal_threshold,
                    max_iterations=max_iterations
                )
            elif mode == 'quick':
                result = loop.learn(topic, goal, max_iterations=1)
            elif mode == 'fast':
                result = loop.learn(topic, goal, max_iterations=min(max_iterations, 3))
            else:
                result = loop.learn(topic, goal, max_iterations)
            
            loop_outputs[loop_id].put(f"\n✅ COMPLETE: {json.dumps(result.get('statistics', {}))}")
            
        except Exception as e:
            loop_outputs[loop_id].put(f"\n❌ ERROR: {str(e)}")
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


if __name__ == '__main__':
    app.run(debug=True, port=5050, threaded=True)
