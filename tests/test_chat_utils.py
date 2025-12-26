from utils.chat_utils import get_chat_completion, load_config
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestChatUtils(unittest.TestCase):

    def test_load_config(self):
        # Test loading the actual config file
        config = load_config()
        self.assertIn('apiHost', config)
        # We can't assert specific values since they are user defined, but we can check structure.
        if 'apiKey' in config:
            self.assertIsInstance(config['apiKey'], str)

    @patch('utils.chat_utils.OpenAI')
    def test_get_chat_completion(self, mock_openai):
        # Get actual config to verify against
        config = load_config()
        expected_host = config['apiHost']
        expected_key = config.get('apiKey', 'EMPTY')
        expected_model = config.get('model')

        # Handle expected base_url logic
        expected_base_url = expected_host
        if not expected_base_url.endswith('/v1'):
            expected_base_url = f"{expected_base_url.rstrip('/')}/v1"

        # Setup mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response

        # Run function
        messages = [{"role": "user", "content": "Hello"}]
        response = get_chat_completion(messages)

        # Assertions
        self.assertEqual(response, "Test response")

        # Verify OpenAI client initialization with actual config values
        mock_openai.assert_called_with(
            base_url=expected_base_url,
            api_key=expected_key
        )

        # Verify completion call with actual config values
        mock_client.chat.completions.create.assert_called_with(
            model=expected_model,
            messages=messages,
            temperature=0.7
        )

    @patch('utils.chat_utils.load_config')
    @patch('utils.chat_utils.OpenAI')
    def test_get_chat_completion_no_v1_suffix(self, mock_openai, mock_load_config):
        # Test that /v1 is appended if missing
        mock_load_config.return_value = {
            "apiHost": "http://test-host",
            "apiKey": "test-key",
            "model": "test-model"
        }

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Response"

        get_chat_completion([{"role": "user", "content": "Hi"}])

        mock_openai.assert_called_with(
            base_url="http://test-host/v1",
            api_key="test-key"
        )

    @patch('utils.chat_utils.load_config')
    @patch('utils.chat_utils.OpenAI')
    def test_get_chat_completion_with_v1_suffix(self, mock_openai, mock_load_config):
        # Test that /v1 is NOT appended if already present
        mock_load_config.return_value = {
            "apiHost": "http://test-host/v1",
            "apiKey": "test-key",
            "model": "test-model"
        }

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Response"

        get_chat_completion([{"role": "user", "content": "Hi"}])

        mock_openai.assert_called_with(
            base_url="http://test-host/v1",
            api_key="test-key"
        )


if __name__ == '__main__':
    unittest.main()
