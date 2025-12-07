import yaml
import json
from pathlib import Path
from datetime import datetime


class KnowledgeBase:
    def __init__(self, actor_name='template'):
        self.actor_name = actor_name
        self.project_root = Path(__file__).parent.parent
        self.actor_root = self.project_root / 'actors' / actor_name
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

    def get_principles(self):
        data = self._load_yaml('principles')
        return data.get('principles', [])

    def get_adjustments(self):
        data = self._load_yaml('adjustments')
        return data.get('adjustments', [])

    def get_core_knowledge(self):
        data = self._load_yaml('core_knowledge')
        return data.get('core_knowledge', [])

    def get_supporting_knowledge(self):
        data = self._load_yaml('supporting_knowledge')
        return data.get('supporting_knowledge', [])

    def get_experiences(self):
        data = self._load_yaml('experiences')
        return data.get('experiences', [])

    def get_challenges(self):
        data = self._load_yaml('challenges')
        return data.get('challenges', [])

    def get_values(self):
        data = self._load_yaml('values')
        return data.get('values', [])

    def refresh(self):
        """Clear cache to reload from disk."""
        self._cache = {}

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
