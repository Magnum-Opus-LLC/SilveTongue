"""
Actor Management for the Silvertongue Dashboard.

Functions for managing actors, their knowledge, and settings.
"""

import json
from pathlib import Path
from datetime import datetime

from .utils import (
    PROJECT_ROOT, get_array_dir, load_yaml_file
)


def get_actor_root(actor_name: str) -> Path:
    """Get the root directory for an actor."""
    if actor_name == 'template':
        return PROJECT_ROOT / 'actors' / actor_name
    return get_array_dir() / 'actors' / actor_name


def get_available_actors() -> list:
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
    template_dir = PROJECT_ROOT / 'actors' / 'template'
    if template_dir.exists() and 'template' not in actors:
        actors.append('template')

    return sorted(actors)


def get_actor_details(actor_name: str) -> dict:
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


def get_actor_knowledge_paths(actor_name: str) -> dict:
    """Get paths to all knowledge files for an actor."""
    actor_root = get_actor_root(actor_name)

    return {
        'values': actor_root / 'characteristic' / 'value' / 'values.yaml',
        'principles': actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
        'experiences': actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
        'core_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
        'supporting_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
        'challenges': actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
        'adjustments': actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml'
    }
