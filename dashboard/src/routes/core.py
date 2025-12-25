"""
Core routes for the Silvertongue Dashboard.

Handles: /, /api/actors, /api/actor/<name>, /api/config
"""

from flask import Blueprint, render_template, jsonify, request

from ..utils import load_config, save_config, get_array_dir
from ..actors import get_available_actors, get_actor_details

core_bp = Blueprint('core', __name__)


@core_bp.route('/')
def index():
    """Main dashboard page."""
    actors = get_available_actors()
    return render_template('index.html', actors=actors)


@core_bp.route('/api/actors')
def api_actors():
    """Get list of actors."""
    return jsonify(get_available_actors())


@core_bp.route('/api/actor/<actor_name>')
def api_actor_details(actor_name):
    """Get actor details."""
    try:
        details = get_actor_details(actor_name)
        return jsonify(details)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@core_bp.route('/api/config')
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


@core_bp.route('/api/config', methods=['POST'])
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

    save_config(config)

    return jsonify({'success': True, 'config': config})
