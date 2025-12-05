import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from subconscious import LingkuiSubconscious
from conscious import LingkuiConscious

class TestThoughtProcess(unittest.TestCase):
    def setUp(self):
        self.mock_kb = MagicMock()
        self.subconscious = LingkuiSubconscious(self.mock_kb)
        self.conscious = LingkuiConscious(self.mock_kb)

    def test_generate_subconscious_prompt(self):
        self.mock_kb.get_values.return_value = ["Value1"]
        self.mock_kb.get_principles.return_value = ["Principle1"]
        self.mock_kb.get_adjustments.return_value = ["Adj1"]
        self.mock_kb.get_core_knowledge.return_value = ["Core1"]
        
        prompt = self.subconscious.generate_subconscious_prompt()
        
        self.assertIn("Value1", prompt)
        self.assertIn("Principle1", prompt)
        self.assertIn("Adj1", prompt)
        self.assertIn("Core1", prompt)

    @patch('subconscious.get_chat_completion')
    def test_analyze_request(self, mock_chat):
        self.mock_kb.get_experiences.return_value = ["Exp1"]
        self.mock_kb.get_supporting_knowledge.return_value = ["Supp1"]
        
        mock_chat.return_value = '{"domain": "Test", "relevant_experiences": ["Exp1"], "relevant_knowledge": []}'
        
        result = self.subconscious.analyze_request("test input")
        
        self.assertEqual(result['domain'], "Test")
        self.assertEqual(result['relevant_experiences'], ["Exp1"])

    def test_construct_conscious_prompt(self):
        self.mock_kb.get_principles.return_value = ["P1"]
        
        analysis = {
            "domain": "Science",
            "relevant_experiences": ["Exp A"],
            "relevant_knowledge": ["Know B"]
        }
        
        prompt = self.conscious.construct_conscious_prompt(analysis)
        
        self.assertIn("Science", prompt)
        self.assertIn("P1", prompt)
        self.assertIn("Exp A", prompt)
        self.assertIn("Know B", prompt)

    @patch('conscious.get_chat_completion')
    def test_perform_task_planning(self, mock_chat):
        mock_chat.return_value = "Plan A"
        result = self.conscious.perform_task_planning("input", "prompt")
        self.assertEqual(result, "Plan A")

    @patch('conscious.open')
    @patch('conscious.Path.mkdir')
    def test_log_interaction(self, mock_mkdir, mock_open):
        self.mock_kb.project_root = Path("/test/root")
        self.mock_kb.actor_name = "test_actor"
        
        # Mock open context manager
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        log_path = self.conscious.log_interaction(
            "user input", 
            "conscious prompt", 
            "response", 
            {"domain": "test"}
        )
        
        self.assertIn("test_actor", log_path)
        # Should open 4 files (prompt, query, response, analysis)
        self.assertEqual(mock_open.call_count, 4)

if __name__ == '__main__':
    unittest.main()
