#!/usr/bin/env python3
"""
Cleanup Script: Remove [Item X] and [Items X,Y,Z] references from knowledge files.

These references were added during condensation but are meaningless in the final
knowledge base since the original items are archived.

Usage:
    python scripts/cleanup_item_references.py <actor_name>
    python scripts/cleanup_item_references.py --all
"""

import re
import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.chat_utils import load_config


def get_array_dir() -> Path:
    """Get the array_dir from config.json."""
    try:
        config = load_config()
        return Path(config.get('array_dir', ''))
    except Exception:
        return project_root


def clean_item_references(text: str) -> str:
    """Remove [Item X] or [Items X,Y,Z] prefixes from text."""
    return re.sub(r'^\[Items?\s*[\d,\s]+\]\s*', '', text)


def cleanup_knowledge_file(file_path: Path, dry_run: bool = False) -> dict:
    """Clean up a single knowledge YAML file."""
    if not file_path.exists():
        return {'status': 'not_found', 'file': str(file_path)}
    
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    # Find the main list key (not _meta)
    list_key = None
    for key in data:
        if key != '_meta' and isinstance(data[key], list):
            list_key = key
            break
    
    if not list_key:
        return {'status': 'no_list_found', 'file': str(file_path)}
    
    items = data[list_key]
    cleaned_items = []
    changes_made = 0
    
    for item in items:
        if isinstance(item, str):
            cleaned = clean_item_references(item)
            if cleaned != item:
                changes_made += 1
            cleaned_items.append(cleaned)
        else:
            cleaned_items.append(item)
    
    if changes_made > 0 and not dry_run:
        data[list_key] = cleaned_items
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return {
        'status': 'cleaned' if changes_made > 0 else 'no_changes',
        'file': str(file_path),
        'items_cleaned': changes_made,
        'total_items': len(items)
    }


def cleanup_actor(actor_name: str, dry_run: bool = False) -> list:
    """Clean up all knowledge files for an actor."""
    array_dir = get_array_dir()
    
    if actor_name == 'template':
        actor_root = project_root / 'actors' / actor_name
    else:
        actor_root = array_dir / 'actors' / actor_name
    
    if not actor_root.exists():
        print(f"❌ Actor '{actor_name}' not found at {actor_root}")
        return []
    
    # Define knowledge file paths
    knowledge_files = [
        actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
        actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
        actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
        actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
        actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
        actor_root / 'characteristic' / 'value' / 'values.yaml',
    ]
    
    results = []
    for file_path in knowledge_files:
        result = cleanup_knowledge_file(file_path, dry_run)
        results.append(result)
    
    return results


def get_all_actors() -> list:
    """Get list of all actors."""
    array_dir = get_array_dir()
    actors_dir = array_dir / 'actors'
    
    actors = []
    if actors_dir.exists():
        for item in actors_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                actors.append(item.name)
    
    # Also include template
    template_dir = project_root / 'actors' / 'template'
    if template_dir.exists() and 'template' not in actors:
        actors.append('template')
    
    return sorted(actors)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean up [Item X] references from knowledge files'
    )
    parser.add_argument('actor', nargs='?', help='Actor name to clean up')
    parser.add_argument('--all', action='store_true', help='Clean up all actors')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    
    args = parser.parse_args()
    
    if args.all:
        actors = get_all_actors()
        print(f"🧹 Cleaning up {len(actors)} actors...")
    elif args.actor:
        actors = [args.actor]
    else:
        parser.print_help()
        return
    
    if args.dry_run:
        print("🔍 DRY RUN - no changes will be made\n")
    
    total_cleaned = 0
    total_files = 0
    
    for actor in actors:
        print(f"\n📁 Actor: {actor}")
        results = cleanup_actor(actor, args.dry_run)
        
        for result in results:
            total_files += 1
            if result['status'] == 'cleaned':
                total_cleaned += result['items_cleaned']
                print(f"   ✓ {Path(result['file']).name}: {result['items_cleaned']}/{result['total_items']} items cleaned")
            elif result['status'] == 'no_changes':
                print(f"   ○ {Path(result['file']).name}: no references found")
            elif result['status'] == 'not_found':
                print(f"   - {Path(result['file']).name}: file not found")
    
    print(f"\n{'=' * 50}")
    print(f"Summary: {total_cleaned} item references removed from {total_files} files")
    if args.dry_run:
        print("(dry run - no actual changes made)")


if __name__ == '__main__':
    main()
