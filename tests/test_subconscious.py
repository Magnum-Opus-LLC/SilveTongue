import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from thoughts.subconscious import SilvertongueSubconscious


class TestSilvertongueSubconscious(unittest.TestCase):
    def setUp(self):
        # Mock the knowledge base passed to the agent
        self.mock_kb = MagicMock()
        self.agent = SilvertongueSubconscious(knowledge_base=self.mock_kb)

    def test_get_character_prompt(self):
        self.mock_kb.get_values.return_value = ["Integrity", "Curiosity"]
        prompt = self.agent._get_character_prompt()
        self.assertIn("Integrity", prompt)
        self.assertIn("Curiosity", prompt)
        self.assertIn("core values", prompt)

    @patch('thoughts.subconscious.get_chat_completion')
    def test_classify_information(self, mock_chat):
        self.mock_kb.get_experiences.return_value = ["Experience 1"]
        mock_chat.return_value = "Chemistry"
        
        domain = self.agent.classify_information("H2O is water.")
        
        self.assertEqual(domain, "Chemistry")
        # Verify chat completion was called with experience in prompt
        args, _ = mock_chat.call_args
        messages = args[0]
        self.assertIn("Experience 1", messages[0]['content'])

    def test_generate_dynamic_system_prompt(self):
        self.mock_kb.get_values.return_value = ["Value A"]
        self.mock_kb.get_experiences.return_value = ["Exp B"]
        
        prompt = self.agent.generate_dynamic_system_prompt("Physics")
        
        self.assertIn("Value A", prompt)
        self.assertIn("Exp B", prompt)
        self.assertIn("Physics", prompt)

    @patch('thoughts.subconscious.get_chat_completion')
    @patch('thoughts.subconscious.SilvertongueSubconscious.classify_information')
    def test_process_content(self, mock_classify, mock_chat):
        mock_classify.return_value = "Biology"
        mock_chat.return_value = "Analysis Result"

        result = self.agent.process_content("Cells divide.", "Explain this.")

        self.assertEqual(result['domain'], "Biology")
        self.assertEqual(result['response'], "Analysis Result")
        self.assertIn("Biology", result['system_prompt_used'])


if __name__ == '__main__':
    unittest.main()
