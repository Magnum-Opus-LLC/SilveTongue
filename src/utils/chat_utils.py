import json
import os
from pathlib import Path
from openai import OpenAI

# Cache for model info
_model_info_cache = {}


def load_config():
    """Load configuration from config.json in the project root."""
    # Get the project root directory (this file is in src/utils/)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # utils -> src -> project_root
    config_path = project_root / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def get_chat_completion(messages, temperature=0.7, model_override=None, max_tokens=None):
    """
    Perform chat completion using the configured host and model.

    Args:
        messages (list): List of message dictionaries (role, content).
        temperature (float): Sampling temperature.
        model_override (str): Optional model name to use.
        max_tokens (int): Maximum tokens to generate. If None, dynamically calculated.

    Returns:
        str: The content of the response message.
    """
    config = load_config()

    api_host = config.get('apiHost')
    api_key = config.get('apiKey', 'EMPTY')  # Default to EMPTY if not provided
    model = model_override or config.get('model')
    context_window = config.get('max_tokens', 32768)
    
    # Calculate input tokens (rough estimate - be conservative, use 3 chars per token)
    input_text = "".join(m.get('content', '') for m in messages)
    input_tokens = len(input_text) // 3  # More conservative than // 4
    
    # Dynamic max_tokens: use remaining context window (with larger buffer for safety)
    if max_tokens is None:
        max_tokens = context_window - input_tokens - 500  # 500 token buffer for safety
        max_tokens = max(1024, min(max_tokens, context_window - 1000))  # Cap to leave room

    if not api_host:
        raise ValueError("apiHost not defined in config.json")

    # Handle base_url construction
    base_url = api_host
    if not base_url.endswith('/v1'):
        base_url = f"{base_url.rstrip('/')}/v1"

    # Initialize OpenAI client with custom base_url
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to get chat completion: {e}")


def get_model_context_window(model_name: str = None) -> int:
    """
    Get the context window size (max tokens) for the current model.
    
    Args:
        model_name: Optional model name. If None, uses config model.
    
    Returns:
        Context window size in tokens (defaults to 8192 if cannot determine)
    """
    config = load_config()
    model = model_name or config.get('model', 'unknown')
    
    # Check if manually configured in config.json
    if 'max_tokens' in config or 'context_window' in config:
        manual_limit = config.get('max_tokens') or config.get('context_window')
        _model_info_cache[model] = manual_limit
        return manual_limit
    
    # Check cache first
    if model in _model_info_cache:
        return _model_info_cache[model]
    
    try:
        api_host = config.get('apiHost')
        api_key = config.get('apiKey', 'EMPTY')
        
        base_url = api_host
        if not base_url.endswith('/v1'):
            base_url = f"{base_url.rstrip('/')}/v1"
        
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        # Try to get model info from /v1/models endpoint
        models = client.models.list()
        
        for model_info in models.data:
            if model_info.id == model or model in model_info.id:
                # Check for context_length field
                if hasattr(model_info, 'context_length'):
                    context_window = model_info.context_length
                    _model_info_cache[model] = context_window
                    return context_window
                # Some servers use max_model_len
                elif hasattr(model_info, 'max_model_len'):
                    context_window = model_info.max_model_len
                    _model_info_cache[model] = context_window
                    return context_window
        
        # Fallback: try to infer from model name
        context_window = _infer_context_from_name(model)
        _model_info_cache[model] = context_window
        return context_window
        
    except Exception as e:
        # Fallback to reasonable default
        print(f"Could not determine context window: {e}")
        return 8192  # Conservative default


def _infer_context_from_name(model_name: str) -> int:
    """Infer context window from common model naming patterns."""
    model_lower = model_name.lower()
    
    # Common patterns
    if '32k' in model_lower or '32000' in model_lower:
        return 32000
    elif '16k' in model_lower or '16000' in model_lower:
        return 16000
    elif '128k' in model_lower or '128000' in model_lower:
        return 128000
    elif '8k' in model_lower or '8000' in model_lower:
        return 8000
    elif '4k' in model_lower or '4096' in model_lower:
        return 4096
    
    # Known models
    if 'llama-3' in model_lower or 'llama3' in model_lower:
        return 8192
    elif 'llama-2' in model_lower or 'llama2' in model_lower:
        return 4096
    elif 'mistral' in model_lower:
        return 32000 if '0.3' in model_lower else 8192
    elif 'qwen' in model_lower:
        return 32000
    elif 'deepseek' in model_lower:
        return 32000
    
    # Default fallback
    return 8192


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count (1 token ≈ 4 characters for English).
    For accurate counting, use tiktoken library.
    
    Args:
        text: Text to estimate
    
    Returns:
        Estimated token count
    """
    # Rough heuristic: 1 token ≈ 4 chars for English text
    return len(text) // 4


def get_safe_chunk_size(context_size: int = 0, overhead: int = 2000) -> int:
    """
    Calculate safe chunk size given context and overhead.
    
    Args:
        context_size: Size of context to include (in tokens)
        overhead: Reserve for prompt + response (in tokens)
    
    Returns:
        Safe chunk size in tokens
    """
    max_tokens = get_model_context_window()
    safe_size = max_tokens - context_size - overhead
    return max(1000, safe_size)  # At least 1000 tokens


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
