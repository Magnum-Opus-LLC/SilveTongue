"""
LoRA adapter routes for Silvertongue Dashboard.
"""

import json
import re
import shutil
from datetime import datetime
from flask import Blueprint, jsonify, request

from dashboard.utils import get_lora_adapters_dir

adapters_bp = Blueprint('adapters', __name__)


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


@adapters_bp.route('/api/adapters', methods=['GET'])
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


@adapters_bp.route('/api/adapters/book', methods=['GET'])
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


@adapters_bp.route('/api/adapters/book', methods=['PUT'])
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


@adapters_bp.route('/api/adapters/<adapter_name>/rename', methods=['POST'])
def api_rename_adapter(adapter_name):
    """Rename a LoRA adapter in the local filesystem."""
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


@adapters_bp.route('/api/adapters/<adapter_name>/<version>', methods=['DELETE'])
def api_delete_adapter_version(adapter_name, version):
    """Delete a specific version of a LoRA adapter."""
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


@adapters_bp.route('/api/adapters/<adapter_name>', methods=['DELETE'])
def api_delete_adapter(adapter_name):
    """Delete a LoRA adapter (all versions) from the local filesystem."""
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
