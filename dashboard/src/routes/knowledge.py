"""
Knowledge management routes for the Silvertongue Dashboard.

Handles: knowledge CRUD, archives, expansions
"""

import json
import yaml
from datetime import datetime
from flask import Blueprint, jsonify, request

from dashboard.utils import get_array_dir, add_to_yaml_list, import_project_module
from dashboard.actors import (
    get_actor_root, get_actor_details, get_actor_knowledge_paths
)

knowledge_bp = Blueprint('knowledge', __name__)


@knowledge_bp.route('/api/actor/<actor_name>/knowledge/<knowledge_type>')
def api_actor_knowledge(actor_name, knowledge_type):
    """Get specific knowledge type for an actor."""
    try:
        details = get_actor_details(actor_name)
        if knowledge_type in details:
            return jsonify({knowledge_type: details[knowledge_type]})
        return jsonify({'error': 'Invalid knowledge type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@knowledge_bp.route('/api/actor/<actor_name>/knowledge/<knowledge_type>', methods=['POST'])
def api_add_knowledge(actor_name, knowledge_type):
    """Add new knowledge item."""
    paths = get_actor_knowledge_paths(actor_name)

    if knowledge_type not in paths:
        return jsonify({'success': False, 'error': f'Invalid knowledge type: {knowledge_type}'}), 400

    # Handle both JSON and form data
    if request.is_json:
        data = request.json or {}
    else:
        data = request.form.to_dict() if request.form else {}

    content = data.get('content', '')
    if not content:
        return jsonify({'success': False, 'error': 'No content provided'}), 400

    source = data.get('source', 'dashboard')

    try:
        success = add_to_yaml_list(
            paths[knowledge_type], knowledge_type, content, source)
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Item already exists (duplicate)'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@knowledge_bp.route('/api/actor/<actor_name>/knowledge/<knowledge_type>/<int:index>', methods=['DELETE'])
def api_delete_knowledge(actor_name, knowledge_type, index):
    """Delete a knowledge item by index."""
    paths = get_actor_knowledge_paths(actor_name)

    if knowledge_type not in paths:
        return jsonify({'success': False, 'error': f'Invalid knowledge type: {knowledge_type}'}), 400

    file_path = paths[knowledge_type]

    try:
        if not file_path.exists():
            return jsonify({'success': False, 'error': 'Knowledge file not found'}), 404

        with open(file_path, 'r') as f:
            data = yaml.safe_load(f) or {}

        items = data.get(knowledge_type, [])

        if not isinstance(items, list):
            return jsonify({'success': False, 'error': 'Invalid knowledge structure'}), 500

        if index < 0 or index >= len(items):
            return jsonify({'success': False, 'error': f'Index {index} out of range (0-{len(items)-1})'}), 400

        deleted_item = items.pop(index)
        data[knowledge_type] = items

        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return jsonify({'success': True, 'deleted': deleted_item})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@knowledge_bp.route('/api/actor/<actor_name>/knowledge/expand', methods=['POST'])
def api_expand_knowledge(actor_name):
    """Generate AI expansion for a knowledge item."""
    data = request.json
    content = data.get('content', '')
    knowledge_type = data.get('type', '')

    if not content:
        return jsonify({'error': 'No content provided'}), 400

    try:
        chat_utils = import_project_module('src.utils.chat_utils')
        get_chat_completion = chat_utils.get_chat_completion

        type_labels = {
            'values': 'value',
            'principles': 'principle',
            'experiences': 'experience',
            'core_knowledge': 'core knowledge item',
            'challenges': 'challenge'
        }
        type_label = type_labels.get(knowledge_type, 'knowledge item')

        system_prompt = "You are a knowledgeable expert providing " + \
            "detailed expansions of concepts and ideas. Write in a clear, " + \
            "informative style using markdown formatting."

        user_prompt = f"""
Expand on the following {type_label} with additional context,
examples, implications, and practical applications.

{type_label.upper()}: {content}

Provide a comprehensive expansion that includes:
1. Deeper Explanation: What does this really mean in practice?
2. Examples: Concrete examples or scenarios where this applies
3. Implications: What are the broader implications or consequences?
4. Related Concepts: How does this connect to other ideas or principles?
5. Practical Applications: How can this be applied in real situations?
"""

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


@knowledge_bp.route(
    '/api/actor/<actor_name>/knowledge/save-expansion',
    methods=['POST']
)
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

        expansions_dir = array_dir / 'data'
        expansions_dir = expansions_dir / actor_name / 'knowledge_expansions'
        expansions_dir.mkdir(parents=True, exist_ok=True)

        expansions_file = expansions_dir / f'{knowledge_type}_expansions.json'

        expansions = {}
        if expansions_file.exists():
            with open(expansions_file, 'r') as f:
                expansions = json.load(f)

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
            experiences_path = actor_root / 'expertise' / \
                'domain' / 'experience' / 'experiences.yaml'
            note_content = f"[Note on {knowledge_type}] {notes}"
            add_to_yaml_list(experiences_path, 'experiences',
                             note_content, 'dashboard-expansion')

        return jsonify({'success': True})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@knowledge_bp.route('/api/actor/<actor_name>/archive/<category>')
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

            timestamp = file.stem.replace(f'{category}_', '')
            mappings_file = archive_dir / \
                f'{category}_{timestamp}_mappings.json'
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
            print(e)
            continue

    return jsonify({'archives': archives})


@knowledge_bp.route('/api/actor/<actor_name>/archive/<category>/<timestamp>')
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
