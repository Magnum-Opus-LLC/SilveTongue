import json
import os
from pathlib import Path
from openai import OpenAI


def load_config():
    """Load configuration from config.json in the project root."""
    # Get the project root directory (assuming this file is in src/)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    config_path = project_root / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def get_chat_completion(messages, temperature=0.7):
    """
    Perform chat completion using the configured host and model.

    Args:
        messages (list): List of message dictionaries (role, content).
        temperature (float): Sampling temperature.

    Returns:
        str: The content of the response message.
    """
    config = load_config()

    api_host = config.get('apiHost')
    api_key = config.get('apiKey', 'EMPTY')  # Default to EMPTY if not provided
    model = config.get('model')

    if not api_host:
        raise ValueError("apiHost not defined in config.json")

    # Handle base_url construction
    base_url = api_host
    if not base_url.endswith('/v1'):
        base_url = f"{base_url.rstrip('/')}/v1"

    # Initialize OpenAI client with custom base_url
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    # Some local servers might not need /v1 if it's already in the host,
    # but standard convention for base_url in openai python lib is usually the root.
    # If the user provided "http://...:8355", it often needs "/v1".
    # I will assume the user provided the base and I might need to adjust or just pass it.
    # Let's stick to passing it as is, but often `v1` is required.
    # I'll check if the user's host ends in /v1.

    # Actually, let's just use the apiHost as base_url.
    # If it fails, the user can adjust the config.

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Simple test
    try:
        test_messages = [
            {"role": "user", "content": "Hello, are you working?"}]
        print("Sending test message...")
        response = get_chat_completion(test_messages)
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")
