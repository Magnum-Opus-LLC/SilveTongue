import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from actor_generator import generate_actor_profile, save_actor_profile


class TestActorGenerator(unittest.TestCase):
    
    @patch('actor_generator.get_chat_completion')
    def test_generate_actor_profile_parses_yaml(self, mock_chat):
        mock_chat.return_value = """```yaml
values:
  - "Value 1"
  - "Value 2"
principles:
  - "Principle 1"
core_knowledge:
  - "Knowledge 1"
supporting_knowledge:
  - "Support 1"
experiences:
  - "Experience 1"
adjustments:
  - "Adjustment 1"
challenges:
  - "Challenge 1"
```"""
        
        profile = generate_actor_profile("Test actor")
        
        self.assertIn("values", profile)
        self.assertIn("principles", profile)
        self.assertEqual(len(profile["values"]), 2)
        self.assertEqual(profile["values"][0], "Value 1")
    
    @patch('actor_generator.get_chat_completion')
    def test_generate_actor_profile_handles_no_code_blocks(self, mock_chat):
        mock_chat.return_value = """values:
  - "Direct YAML"
principles:
  - "P1"
core_knowledge: []
supporting_knowledge: []
experiences: []
adjustments: []
challenges: []
"""
        
        profile = generate_actor_profile("Test actor")
        
        self.assertEqual(profile["values"][0], "Direct YAML")
    
    def test_save_actor_profile_creates_directories(self):
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal template structure
            template_dir = Path(tmpdir) / 'actors' / 'template'
            template_dir.mkdir(parents=True)
            
            # Write a minimal behavior.json
            with open(template_dir / 'behavior.json', 'w') as f:
                f.write('{}')
            
            profile = {
                "values": ["Test Value"],
                "principles": ["Test Principle"],
                "core_knowledge": ["Test Knowledge"],
                "supporting_knowledge": [],
                "experiences": [],
                "adjustments": [],
                "challenges": []
            }
            
            save_actor_profile(profile, "test_actor", project_root=Path(tmpdir))
            
            # Check files were created
            actor_dir = Path(tmpdir) / 'actors' / 'test_actor'
            self.assertTrue((actor_dir / 'characteristic' / 'value' / 'values.yaml').exists())
            self.assertTrue((actor_dir / 'expertise' / 'domain' / 'principal' / 'principles.yaml').exists())


if __name__ == '__main__':
    unittest.main()
