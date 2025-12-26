"""
Actor creation routes for Silvertongue Dashboard.
"""

from flask import Blueprint, jsonify, request

from ..utils import get_array_dir, PROJECT_ROOT, import_project_module

actor_creation_bp = Blueprint('actor_creation', __name__)


@actor_creation_bp.route('/api/actor/create', methods=['POST'])
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
        actor_generator = import_project_module('src.utils.actor_generator')
        generate_actor_profile = actor_generator.generate_actor_profile
        save_actor_profile = actor_generator.save_actor_profile

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
        save_actor_profile(profile, actor_name, PROJECT_ROOT)

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


@actor_creation_bp.route('/api/actor/create/manual', methods=['POST'])
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
        actor_generator = import_project_module('src.utils.actor_generator')
        save_actor_profile = actor_generator.save_actor_profile

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
        save_actor_profile(default_profile, actor_name, PROJECT_ROOT)

        return jsonify({
            'success': True,
            'actor': actor_name,
            'message': f'Actor "{actor_name}" created successfully'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
