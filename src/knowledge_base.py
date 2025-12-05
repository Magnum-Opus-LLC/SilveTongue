import yaml
from pathlib import Path


class KnowledgeBase:
    def __init__(self, actor_name='template'):
        self.project_root = Path(__file__).parent.parent
        self.actor_root = self.project_root / 'actors' / actor_name
        self.paths = {
            'principles': self.actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
            'adjustments': self.actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml',
            'core_knowledge': self.actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
            'supporting_knowledge': self.actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
            'experiences': self.actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
            'challenges': self.actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
            'values': self.actor_root / 'characteristic' / 'value' / 'values.yaml'
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


# Global instance for easy access
kb = KnowledgeBase()
