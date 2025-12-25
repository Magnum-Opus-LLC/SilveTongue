"""
Configuration and utility functions for the Silvertongue Dashboard.
"""

import json
import yaml
import queue
from pathlib import Path
from datetime import datetime

# Project paths
DASHBOARD_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = DASHBOARD_ROOT.parent

# Global state for running loops
active_loops = {}
loop_outputs = {}  # Store output queues for each loop


def load_config() -> dict:
    """Load configuration from config.json."""
    config_path = PROJECT_ROOT / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: dict):
    """Save configuration to config.json."""
    config_path = PROJECT_ROOT / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def get_array_dir() -> Path:
    """Get the array directory from config.json."""
    config = load_config()
    array_dir = config.get('array_dir', '')
    if array_dir:
        return Path(array_dir)
    return PROJECT_ROOT / 'array'


def get_lora_adapters_dir() -> Path:
    """Get the lora_adapters directory from config.json."""
    config = load_config()
    lora_dir = config.get('lora_adapters_dir', '')
    if lora_dir:
        return Path(lora_dir)
    # Fallback: try to derive from array_dir
    array_dir = get_array_dir()
    return array_dir.parent / 'lora_adapters'


def get_pdf_cache_dir() -> Path:
    """Get the PDF cache directory."""
    array_dir = get_array_dir()
    return array_dir / 'data' / 'pdf_cache'


def get_pdf_stock_path() -> Path:
    """Get the path to the PDF stock index file."""
    return get_pdf_cache_dir() / 'pdf_stock.json'


def load_yaml_file(file_path: Path) -> dict:
    """Load a YAML file safely."""
    if file_path.exists():
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_yaml_file(file_path: Path, data: dict):
    """Save data to a YAML file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


def add_to_yaml_list(file_path: Path, key: str, item: str, source: str = None) -> bool:
    """Add an item to a YAML list file."""
    data = load_yaml_file(file_path)
    items = data.get(key, [])

    if source:
        item = f"{item} [Source: {source}]"

    # Avoid duplicates
    item_lower = item.lower()[:50]
    for existing in items:
        if existing.lower()[:50] == item_lower:
            return False

    items.append(item)
    data[key] = items
    data['_meta'] = {
        'last_updated': datetime.now().isoformat(),
        'updated_by': 'dashboard'
    }

    save_yaml_file(file_path, data)
    return True


# ==================== Trainer Server Config ====================

def get_trainer_config() -> dict:
    """Get trainer server configuration."""
    config = load_config()
    return config.get('trainer_server', {
        'host': 'localhost',
        'port': 5050,
        'enabled': False
    })


def get_trainer_url() -> str:
    """Get the trainer server URL."""
    trainer = get_trainer_config()
    return f"http://{trainer['host']}:{trainer['port']}"


def translate_path_for_server(local_path: str) -> str:
    """
    Translate a local Mac path to the server's view inside the Docker container.

    The launch script mounts:
      -v "$LORA_OUTPUT_DIR":/lora_adapters
      -v "$DATASET_DIR":/datasets

    Where:
      LORA_OUTPUT_DIR=/mnt/silvertongueArray/lora_adapters
      DATASET_DIR=/mnt/silvertongueArray/array/data

    Mac paths need to be translated:
      /Users/ruihe/mnt/silvertongueArray/array/data/... -> /datasets/...
      /Users/ruihe/mnt/silvertongueArray/lora_adapters/... -> /lora_adapters/...
    """
    config = load_config()
    local_array_dir = config.get('array_dir', '')

    # Mac mount points
    mac_array_mount = '/Users/ruihe/mnt/silvertongueArray/array'
    mac_lora_mount = '/Users/ruihe/mnt/silvertongueArray/lora_adapters'

    # DGX host paths
    dgx_array_mount = '/mnt/silvertongueArray/array'
    dgx_lora_mount = '/mnt/silvertongueArray/lora_adapters'

    # Docker container paths
    docker_data_dir = '/datasets'
    docker_lora_dir = '/lora_adapters'

    server_path = local_path

    # Handle LoRA adapter paths FIRST (more specific)
    if local_path.startswith(mac_lora_mount):
        server_path = local_path.replace(mac_lora_mount, docker_lora_dir)
        return server_path
    elif local_path.startswith(dgx_lora_mount):
        server_path = local_path.replace(dgx_lora_mount, docker_lora_dir)
        return server_path

    # Handle dataset/array paths
    if local_array_dir and local_path.startswith(local_array_dir):
        server_path = local_path.replace(local_array_dir, dgx_array_mount)
    elif local_path.startswith(mac_array_mount):
        server_path = local_path.replace(mac_array_mount, dgx_array_mount)

    # Convert DGX host path to Docker container path
    dgx_data_dir = '/mnt/silvertongueArray/array/data'

    if server_path.startswith(dgx_data_dir):
        server_path = server_path.replace(dgx_data_dir, docker_data_dir)

    return server_path
