import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime


def get_array_dir() -> Path:
    """Get the array_dir from config.json."""
    try:
        from src.utils.chat_utils import load_config
        config = load_config()
        return Path(config.get('array_dir', ''))
    except Exception:
        # Fallback to project root
        return Path(__file__).parent.parent.parent


class KnowledgeBase:
    def __init__(self, actor_name='template'):
        self.actor_name = actor_name
        self.project_root = Path(__file__).parent.parent.parent
        
        # Template stays in core, all other actors go to array_dir
        if actor_name == 'template':
            self.actor_root = self.project_root / 'actors' / actor_name
        else:
            array_dir = get_array_dir()
            self.actor_root = array_dir / 'actors' / actor_name
            
            # If actor doesn't exist in array_dir, copy from template
            if not self.actor_root.exists():
                self._init_actor_from_template()
        
        self.archive_root = self.actor_root / 'archive'
        self.paths = {
            'principles': self.actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
            'adjustments': self.actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml',
            'core_knowledge': self.actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
            'supporting_knowledge': self.actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
            'experiences': self.actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
            'challenges': self.actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
            'values': self.actor_root / 'characteristic' / 'value' / 'values.yaml',
            'learning_book': self.actor_root / 'learning_book.json'
        }
        self._cache = {}
        
        # Ensure archive directory exists
        self.archive_root.mkdir(parents=True, exist_ok=True)
    
    def _init_actor_from_template(self):
        """Initialize a new actor by copying from template."""
        template_root = self.project_root / 'actors' / 'template'
        if template_root.exists():
            shutil.copytree(template_root, self.actor_root)
            print(f"📁 Created new actor '{self.actor_name}' from template at {self.actor_root}")

    def _load_yaml(self, key):
        """Load YAML file for a given key if not already cached."""
        if key not in self._cache:
            file_path = self.paths.get(key)
            if not file_path or not file_path.exists():
                # Return empty list or dict structure if file doesn't exist,
                # but here we expect them to exist as we created them.
                # Returning empty structure is safer.
                return {}

            with open(file_path, 'r') as f:
                self._cache[key] = yaml.safe_load(f) or {}
        return self._cache[key]

    def _track_usage(self, key: str, item_index: int = None):
        """Track when knowledge items are accessed for usage-based condensation."""
        usage_file = self.actor_root / 'usage_stats.json'
        
        # Load existing usage stats
        if usage_file.exists():
            with open(usage_file, 'r') as f:
                usage = json.load(f)
        else:
            usage = {}
        
        # Initialize category if needed
        if key not in usage:
            usage[key] = {'access_count': 0, 'last_accessed': None, 'items': {}}
        
        # Update access stats
        usage[key]['access_count'] += 1
        usage[key]['last_accessed'] = datetime.now().isoformat()
        
        # Track individual item if specified
        if item_index is not None:
            item_key = str(item_index)
            if item_key not in usage[key]['items']:
                usage[key]['items'][item_key] = {'count': 0, 'last_accessed': None}
            usage[key]['items'][item_key]['count'] += 1
            usage[key]['items'][item_key]['last_accessed'] = datetime.now().isoformat()
        
        # Save updated usage
        with open(usage_file, 'w') as f:
            json.dump(usage, f, indent=2)
    
    def get_usage_stats(self, key: str = None):
        """Get usage statistics for knowledge items."""
        usage_file = self.actor_root / 'usage_stats.json'
        if not usage_file.exists():
            return {} if key is None else {'access_count': 0, 'items': {}}
        
        with open(usage_file, 'r') as f:
            usage = json.load(f)
        
        return usage.get(key, {'access_count': 0, 'items': {}}) if key else usage

    def get_principles(self, track_usage=True):
        data = self._load_yaml('principles')
        if track_usage:
            self._track_usage('principles')
        return data.get('principles', [])

    def get_adjustments(self, track_usage=True):
        data = self._load_yaml('adjustments')
        if track_usage:
            self._track_usage('adjustments')
        return data.get('adjustments', [])

    def get_core_knowledge(self, track_usage=True):
        data = self._load_yaml('core_knowledge')
        if track_usage:
            self._track_usage('core_knowledge')
        return data.get('core_knowledge', [])

    def get_supporting_knowledge(self, track_usage=True):
        data = self._load_yaml('supporting_knowledge')
        if track_usage:
            self._track_usage('supporting_knowledge')
        return data.get('supporting_knowledge', [])

    def get_experiences(self, track_usage=True):
        data = self._load_yaml('experiences')
        if track_usage:
            self._track_usage('experiences')
        return data.get('experiences', [])

    def get_challenges(self, track_usage=True):
        data = self._load_yaml('challenges')
        if track_usage:
            self._track_usage('challenges')
        return data.get('challenges', [])

    def get_values(self, track_usage=True):
        data = self._load_yaml('values')
        if track_usage:
            self._track_usage('values')
        return data.get('values', [])

    def refresh(self):
        """Clear cache to reload from disk."""
        self._cache = {}

    # ==================== Archive & Update Methods ====================
    
    def _archive_file(self, key: str, reason: str = "update") -> Path:
        """
        Archive a file before making changes.
        
        Args:
            key: The key name (e.g., 'principles', 'experiences')
            reason: Reason for archiving (e.g., 'update', 'cognitive_loop', 'manual')
            
        Returns:
            Path to the archived file
        """
        file_path = self.paths.get(key)
        if not file_path or not file_path.exists():
            return None
        
        # Create timestamped archive directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = self.archive_root / key
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Archive filename: principles_20251208_143022_update.yaml
        archive_name = f"{file_path.stem}_{timestamp}_{reason}{file_path.suffix}"
        archive_path = archive_dir / archive_name
        
        # Copy current file to archive
        shutil.copy2(file_path, archive_path)
        
        return archive_path
    
    def _save_yaml(self, key: str, data: dict, archive: bool = True, reason: str = "update"):
        """
        Save data to a YAML file, optionally archiving the previous version.
        
        Args:
            key: The key name (e.g., 'principles', 'experiences')
            data: The data dict to save
            archive: Whether to archive the previous version
            reason: Reason for the change
        """
        file_path = self.paths.get(key)
        if not file_path:
            raise ValueError(f"Unknown key: {key}")
        
        # Archive if requested and file exists
        if archive and file_path.exists():
            self._archive_file(key, reason)
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        data['_meta'] = {
            'last_updated': datetime.now().isoformat(),
            'updated_by': 'cognitive_loop' if reason == 'cognitive_loop' else 'knowledge_base',
            'reason': reason
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Update cache
        self._cache[key] = data
    
    def add_principle(self, principle: str, source: str = None, archive: bool = True):
        """
        Add a new principle to the actor's knowledge base.
        
        Args:
            principle: The principle text
            source: Where this principle came from (e.g., paper name)
            archive: Whether to archive the previous file
        """
        data = self._load_yaml('principles')
        principles = data.get('principles', [])
        
        # Add with source annotation if provided
        if source:
            principle = f"{principle} [Source: {source}]"
        
        # Avoid duplicates (check if similar principle exists)
        principle_lower = principle.lower()
        for existing in principles:
            if existing.lower()[:50] == principle_lower[:50]:
                return False  # Already exists
        
        principles.append(principle)
        data['principles'] = principles
        
        self._save_yaml('principles', data, archive=archive, reason='add_principle')
        self.record_knowledge_evolution('principles_added', principle[:100])
        
        return True
    
    def update_principle(self, old_principle: str, new_principle: str, reason: str = "refined"):
        """
        Update/refine an existing principle.
        
        Args:
            old_principle: The original principle (or substring to match)
            new_principle: The updated principle
            reason: Reason for the update
        """
        data = self._load_yaml('principles')
        principles = data.get('principles', [])
        
        # Find and replace
        old_lower = old_principle.lower()
        for i, p in enumerate(principles):
            if old_lower in p.lower():
                principles[i] = f"{new_principle} [Refined: {datetime.now().strftime('%Y-%m-%d')}]"
                data['principles'] = principles
                self._save_yaml('principles', data, archive=True, reason=f'refine_principle_{reason}')
                self.record_knowledge_evolution('principles_refined', f"{old_principle[:50]} -> {new_principle[:50]}")
                return True
        
        return False
    
    def add_experience(self, experience: str, source: str = None, archive: bool = True):
        """
        Add a new experience to the actor's knowledge base.
        
        Args:
            experience: The experience text
            source: Where this experience came from
            archive: Whether to archive the previous file
        """
        data = self._load_yaml('experiences')
        experiences = data.get('experiences', [])
        
        if source:
            experience = f"{experience} [Source: {source}]"
        
        # Avoid duplicates
        exp_lower = experience.lower()[:50]
        for existing in experiences:
            if existing.lower()[:50] == exp_lower:
                return False
        
        experiences.append(experience)
        data['experiences'] = experiences
        
        self._save_yaml('experiences', data, archive=archive, reason='add_experience')
        self.record_knowledge_evolution('experiences_learned', experience[:100])
        
        return True
    
    def add_core_knowledge(self, knowledge: str, source: str = None, archive: bool = True):
        """Add new core knowledge."""
        data = self._load_yaml('core_knowledge')
        items = data.get('core_knowledge', [])
        
        if source:
            knowledge = f"{knowledge} [Source: {source}]"
        
        # Avoid duplicates
        k_lower = knowledge.lower()[:50]
        for existing in items:
            if existing.lower()[:50] == k_lower:
                return False
        
        items.append(knowledge)
        data['core_knowledge'] = items
        
        self._save_yaml('core_knowledge', data, archive=archive, reason='add_knowledge')
        
        return True
    
    def add_supporting_knowledge(self, knowledge: str, source: str = None, archive: bool = True):
        """Add new supporting knowledge."""
        data = self._load_yaml('supporting_knowledge')
        items = data.get('supporting_knowledge', [])
        
        if source:
            knowledge = f"{knowledge} [Source: {source}]"
        
        # Avoid duplicates
        k_lower = knowledge.lower()[:50]
        for existing in items:
            if existing.lower()[:50] == k_lower:
                return False
        
        items.append(knowledge)
        data['supporting_knowledge'] = items
        
        self._save_yaml('supporting_knowledge', data, archive=archive, reason='add_supporting')
        
        return True
    
    def add_challenge(self, challenge: str, source: str = None, archive: bool = True):
        """Add a new challenge encountered."""
        data = self._load_yaml('challenges')
        challenges = data.get('challenges', [])
        
        if source:
            challenge = f"{challenge} [Source: {source}]"
        
        challenges.append(f"[{datetime.now().strftime('%Y-%m-%d')}] {challenge}")
        data['challenges'] = challenges
        
        self._save_yaml('challenges', data, archive=archive, reason='add_challenge')
        self.record_knowledge_evolution('challenges_encountered', challenge[:100])
        
        return True
    
    def resolve_challenge(self, challenge_substring: str, resolution: str):
        """Mark a challenge as resolved."""
        data = self._load_yaml('challenges')
        challenges = data.get('challenges', [])
        
        challenge_lower = challenge_substring.lower()
        for i, c in enumerate(challenges):
            if challenge_lower in c.lower():
                challenges[i] = f"{c} [RESOLVED: {resolution}]"
                data['challenges'] = challenges
                self._save_yaml('challenges', data, archive=True, reason='resolve_challenge')
                self.record_knowledge_evolution('challenges_resolved', f"{challenge_substring[:50]}: {resolution[:50]}")
                return True
        
        return False
    
    def add_adjustment(self, adjustment: str, source: str = None, archive: bool = True):
        """Add a behavioral/methodological adjustment."""
        data = self._load_yaml('adjustments')
        adjustments = data.get('adjustments', [])
        
        if source:
            adjustment = f"{adjustment} [Source: {source}]"
        
        adjustments.append(f"[{datetime.now().strftime('%Y-%m-%d')}] {adjustment}")
        data['adjustments'] = adjustments
        
        self._save_yaml('adjustments', data, archive=archive, reason='add_adjustment')
        
        return True
    
    def bulk_update(self, updates: dict, reason: str = "cognitive_loop"):
        """
        Apply multiple updates at once.
        
        Args:
            updates: Dict with keys like 'principles', 'experiences', etc.
                     Each value is a list of items to add.
            reason: Reason for the bulk update
        """
        results = {
            'principles_added': 0,
            'experiences_added': 0,
            'knowledge_added': 0,
            'challenges_added': 0,
            'adjustments_added': 0
        }
        
        source = f"bulk_update_{datetime.now().strftime('%Y%m%d')}"
        
        for principle in updates.get('principles', []):
            if self.add_principle(principle, source=source):
                results['principles_added'] += 1
        
        for experience in updates.get('experiences', []):
            if self.add_experience(experience, source=source):
                results['experiences_added'] += 1
        
        for knowledge in updates.get('core_knowledge', []):
            if self.add_core_knowledge(knowledge, source=source):
                results['knowledge_added'] += 1
        
        for knowledge in updates.get('supporting_knowledge', []):
            if self.add_supporting_knowledge(knowledge, source=source):
                results['knowledge_added'] += 1
        
        for challenge in updates.get('challenges', []):
            if self.add_challenge(challenge, source=source):
                results['challenges_added'] += 1
        
        for adjustment in updates.get('adjustments', []):
            if self.add_adjustment(adjustment, source=source):
                results['adjustments_added'] += 1
        
        return results
    
    def get_archive_history(self, key: str = None) -> list:
        """
        Get archive history for a key or all keys.
        
        Args:
            key: Specific key to get history for, or None for all
            
        Returns:
            List of archived files with metadata
        """
        history = []
        
        if key:
            archive_dir = self.archive_root / key
            if archive_dir.exists():
                for f in sorted(archive_dir.glob('*'), reverse=True):
                    history.append({
                        'key': key,
                        'file': f.name,
                        'path': str(f),
                        'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })
        else:
            for k in self.paths.keys():
                if k == 'learning_book':
                    continue
                archive_dir = self.archive_root / k
                if archive_dir.exists():
                    for f in sorted(archive_dir.glob('*'), reverse=True)[:5]:
                        history.append({
                            'key': k,
                            'file': f.name,
                            'path': str(f),
                            'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                        })
        
        return history
    
    def restore_from_archive(self, key: str, archive_filename: str):
        """
        Restore a previous version from archive.
        
        Args:
            key: The key to restore (e.g., 'principles')
            archive_filename: The archive file to restore from
        """
        archive_path = self.archive_root / key / archive_filename
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        
        target_path = self.paths.get(key)
        
        # Archive current before restore
        self._archive_file(key, reason='pre_restore')
        
        # Copy archive to target
        shutil.copy2(archive_path, target_path)
        
        # Clear cache
        if key in self._cache:
            del self._cache[key]
        
        return True

    # ==================== Learning Book Methods ====================
    
    def get_learning_book(self):
        """Load the learning book for this actor."""
        book_path = self.paths['learning_book']
        if not book_path.exists():
            return self._create_empty_learning_book()
        
        with open(book_path, 'r') as f:
            return json.load(f)
    
    def _create_empty_learning_book(self):
        """Create a new empty learning book structure."""
        return {
            "meta": {
                "actor_name": self.actor_name,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": 1
            },
            "current_mindset": {
                "summary": "",
                "dominant_values": [],
                "active_principles": [],
                "focus_areas": [],
                "confidence_level": "low"
            },
            "learning_history": [],
            "knowledge_evolution": {
                "principles_added": [],
                "principles_refined": [],
                "experiences_learned": [],
                "challenges_encountered": [],
                "challenges_resolved": []
            },
            "reflection_notes": []
        }
    
    def save_learning_book(self, book):
        """Save the learning book to disk."""
        book_path = self.paths['learning_book']
        book['meta']['last_updated'] = datetime.now().isoformat()
        book['meta']['version'] = book['meta'].get('version', 0) + 1
        
        with open(book_path, 'w') as f:
            json.dump(book, f, indent=2)
    
    def record_learning_event(self, event_type, description, query=None, insights=None, mindset_changes=None):
        """
        Record a learning event in the actor's book.
        
        Args:
            event_type: Type of event (e.g., 'research_task', 'user_interaction', 'self_reflection')
            description: Brief description of what happened
            query: The user query that triggered this event (optional)
            insights: List of insights gained (optional)
            mindset_changes: List of mindset changes (optional)
        """
        book = self.get_learning_book()
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "description": description,
            "insights_gained": insights or [],
            "mindset_changes": mindset_changes or []
        }
        
        if query:
            event["query"] = query
        
        book['learning_history'].append(event)
        self.save_learning_book(book)
        
        return event
    
    def update_mindset(self, summary=None, dominant_values=None, active_principles=None, 
                       focus_areas=None, confidence_level=None):
        """Update the current mindset of the actor."""
        book = self.get_learning_book()
        
        if summary is not None:
            book['current_mindset']['summary'] = summary
        if dominant_values is not None:
            book['current_mindset']['dominant_values'] = dominant_values
        if active_principles is not None:
            book['current_mindset']['active_principles'] = active_principles
        if focus_areas is not None:
            book['current_mindset']['focus_areas'] = focus_areas
        if confidence_level is not None:
            book['current_mindset']['confidence_level'] = confidence_level
        
        self.save_learning_book(book)
    
    def add_reflection_note(self, note, action_items=None):
        """Add a reflection note to the learning book."""
        book = self.get_learning_book()
        
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "note": note,
            "action_items": action_items or []
        }
        
        book['reflection_notes'].append(reflection)
        self.save_learning_book(book)
    
    def record_knowledge_evolution(self, category, item):
        """
        Record a knowledge evolution event.
        
        Args:
            category: One of 'principles_added', 'principles_refined', 'experiences_learned',
                     'challenges_encountered', 'challenges_resolved'
            item: Description of the knowledge item
        """
        book = self.get_learning_book()
        
        if category in book['knowledge_evolution']:
            dated_item = f"{datetime.now().strftime('%Y-%m-%d')}: {item}"
            book['knowledge_evolution'][category].append(dated_item)
            self.save_learning_book(book)
    
    def get_current_mindset(self):
        """Get the current mindset summary."""
        book = self.get_learning_book()
        return book.get('current_mindset', {})
    
    def get_recent_learnings(self, count=5):
        """Get the most recent learning events."""
        book = self.get_learning_book()
        history = book.get('learning_history', [])
        return history[-count:] if history else []


# Global instance for easy access
kb = KnowledgeBase()
