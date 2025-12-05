from knowledge_base import KnowledgeBase
import unittest
from unittest.mock import patch, mock_open
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(actor_name='template')

    @patch('builtins.open', new_callable=mock_open, read_data="principles:\n  - Be kind\n")
    @patch('knowledge_base.Path.exists')
    def test_get_principles(self, mock_exists, mock_file):
        mock_exists.return_value = True
        principles = self.kb.get_principles()
        self.assertEqual(principles, ['Be kind'])
        # Verify it tries to open the correct file
        expected_path = self.kb.paths['principles']
        mock_file.assert_called_with(expected_path, 'r')

    @patch('builtins.open', new_callable=mock_open, read_data="adjustments:\n  - Rule 1\n")
    @patch('knowledge_base.Path.exists')
    def test_get_adjustments(self, mock_exists, mock_file):
        mock_exists.return_value = True
        adjustments = self.kb.get_adjustments()
        self.assertEqual(adjustments, ['Rule 1'])

    @patch('builtins.open', new_callable=mock_open, read_data="core_knowledge:\n  - Fact A\n")
    @patch('knowledge_base.Path.exists')
    def test_get_core_knowledge(self, mock_exists, mock_file):
        mock_exists.return_value = True
        data = self.kb.get_core_knowledge()
        self.assertEqual(data, ['Fact A'])

    @patch('builtins.open', new_callable=mock_open, read_data="supporting_knowledge:\n  - Case 1\n")
    @patch('knowledge_base.Path.exists')
    def test_get_supporting_knowledge(self, mock_exists, mock_file):
        mock_exists.return_value = True
        data = self.kb.get_supporting_knowledge()
        self.assertEqual(data, ['Case 1'])

    @patch('builtins.open', new_callable=mock_open, read_data="experiences:\n  - Action 1\n")
    @patch('knowledge_base.Path.exists')
    def test_get_experiences(self, mock_exists, mock_file):
        mock_exists.return_value = True
        data = self.kb.get_experiences()
        self.assertEqual(data, ['Action 1'])

    @patch('builtins.open', new_callable=mock_open, read_data="challenges:\n  - Challenge 1\n")
    @patch('knowledge_base.Path.exists')
    def test_get_challenges(self, mock_exists, mock_file):
        mock_exists.return_value = True
        data = self.kb.get_challenges()
        self.assertEqual(data, ['Challenge 1'])

    @patch('builtins.open', new_callable=mock_open, read_data="values:\n  - Value 1\n")
    @patch('knowledge_base.Path.exists')
    def test_get_values(self, mock_exists, mock_file):
        mock_exists.return_value = True
        data = self.kb.get_values()
        self.assertEqual(data, ['Value 1'])

    def test_missing_file(self):
        with patch('knowledge_base.Path.exists') as mock_exists:
            mock_exists.return_value = False
            # Should return empty list/dict if file missing
            self.assertEqual(self.kb.get_principles(), [])


if __name__ == '__main__':
    unittest.main()
