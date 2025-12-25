"""
Trainer server routes for Silvertongue Dashboard.
"""

import json
from flask import Blueprint, jsonify, request

from ..utils import (
    load_config, get_array_dir, get_trainer_config, get_trainer_url,
    translate_path_for_server, PROJECT_ROOT
)

trainer_bp = Blueprint('trainer', __name__)


@trainer_bp.route('/api/trainer/status', methods=['GET'])
def api_trainer_status():
    """Check if the remote trainer server is available."""
    trainer = get_trainer_config()

    if not trainer.get('enabled', False):
        return jsonify({
            'enabled': False,
            'message': 'Remote trainer is not enabled in config.json'
        })

    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/health", timeout=5)
        if response.ok:
            health_data = response.json()

            # Also fetch config to get available models
            available_models = []
            try:
                config_response = requests.get(
                    f"{get_trainer_url()}/api/config", timeout=5)
                if config_response.ok:
                    config_data = config_response.json()
                    available_models = config_data.get('available_models', [])
            except:
                pass  # Non-critical, just won't show model options

            return jsonify({
                'enabled': True,
                'connected': True,
                'base_model': health_data.get('base_model', 'Unknown'),
                'available_models': available_models,
                'server': health_data,
                'url': get_trainer_url()
            })
    except Exception as e:
        return jsonify({
            'enabled': True,
            'connected': False,
            'error': str(e),
            'url': get_trainer_url()
        })


@trainer_bp.route('/api/trainer/config', methods=['GET'])
def api_trainer_config():
    """Get trainer server configuration."""
    return jsonify(get_trainer_config())


@trainer_bp.route('/api/trainer/config', methods=['PUT'])
def api_update_trainer_config():
    """Update trainer server configuration."""
    data = request.json
    config = load_config()

    trainer = config.get('trainer_server', {})
    if 'host' in data:
        trainer['host'] = data['host']
    if 'port' in data:
        trainer['port'] = data['port']
    if 'enabled' in data:
        trainer['enabled'] = data['enabled']

    config['trainer_server'] = trainer

    config_path = PROJECT_ROOT / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return jsonify({'success': True, 'trainer_server': trainer})


@trainer_bp.route('/api/actor/<actor_name>/train', methods=['POST'])
def api_start_training(actor_name):
    """Submit a LoRA training job to the remote trainer server."""
    trainer = get_trainer_config()

    if not trainer.get('enabled', False):
        return jsonify({
            'error': 'Remote trainer is not enabled',
            'hint': 'Enable trainer_server in config.json or set up DGX Spark'
        }), 400

    # Check if training data exists - look for various dataset patterns
    array_dir = get_array_dir()
    lora_dir = array_dir / 'data' / actor_name / 'lora_training'

    # Find the best dataset file (prefer merged, then latest timestamped, then basic)
    train_file = None

    # Check request for specific dataset
    data = request.json or {}
    requested_dataset = data.get('dataset') or data.get(
        'dataset_file')  # Support both field names
    # Custom base model selection
    requested_base_model = data.get('base_model')

    if requested_dataset:
        train_file = lora_dir / requested_dataset
    else:
        # Auto-find best dataset
        # 1. Look for merged datasets (most comprehensive)
        merged_files = sorted(lora_dir.glob(
            f'{actor_name}_merged_*.jsonl'), reverse=True)
        if merged_files:
            train_file = merged_files[0]
        else:
            # 2. Look for timestamped datasets
            timestamped = sorted(lora_dir.glob(
                f'{actor_name}_lora_dataset_*.jsonl'), reverse=True)
            if timestamped:
                train_file = timestamped[0]
            else:
                # 3. Look for basic dataset
                basic = lora_dir / f"{actor_name}_lora_dataset.jsonl"
                if basic.exists():
                    train_file = basic
                else:
                    # 4. Fallback to old naming
                    old_style = lora_dir / f"{actor_name}_lora_train.jsonl"
                    if old_style.exists():
                        train_file = old_style

    if not train_file or not train_file.exists():
        return jsonify({
            'error': 'No training data found',
            'hint': 'Run the sleep cycle first to generate training data',
            'searched_dir': str(lora_dir)
        }), 404

    # Translate local path to server path
    server_dataset_path = translate_path_for_server(str(train_file))

    # Handle base_adapter for LoRA-on-LoRA training
    requested_base_adapter = data.get('base_adapter')
    server_base_adapter = None
    if requested_base_adapter:
        # Translate the adapter path for the server
        server_base_adapter = translate_path_for_server(requested_base_adapter)
        print(f"[TRAIN] LoRA-on-LoRA requested:")
        print(f"  Local adapter path: {requested_base_adapter}")
        print(f"  Server adapter path: {server_base_adapter}")

    # Build training config - use server's expected field names (flat structure)
    training_config = {
        'actor_name': actor_name,
        'train_file': server_dataset_path,  # Server expects 'train_file'
        # Custom base model (None = server default)
        'base_model': requested_base_model,
        # LoRA-on-LoRA: existing adapter to build upon
        'base_adapter': server_base_adapter,
        'lora_r': data.get('lora_r', 32),
        'lora_alpha': data.get('lora_alpha', 64),
        'lora_dropout': data.get('lora_dropout', 0.05),
        'learning_rate': data.get('learning_rate', 2e-4),
        'num_epochs': data.get('num_epochs', 3),
        'batch_size': data.get('batch_size', 4),
        'gradient_accumulation_steps': data.get('gradient_accumulation_steps', 4),
        'max_seq_length': data.get('max_seq_length', 2048),
    }

    # Remove None values so server uses its defaults
    training_config = {k: v for k,
                       v in training_config.items() if v is not None}

    try:
        import requests
        response = requests.post(
            f"{get_trainer_url()}/api/train",
            json=training_config,
            timeout=30
        )

        if response.ok:
            result = response.json()
            return jsonify({
                'success': True,
                'job_id': result.get('job_id'),
                'status': 'started',
                'message': f'Training job submitted for {actor_name}',
                'dataset': str(train_file.name),
                'config': training_config
            })
        else:
            return jsonify({
                'error': 'Failed to submit training job',
                'details': response.text
            }), response.status_code

    except Exception as e:
        if 'ConnectionError' in type(e).__name__:
            return jsonify({
                'error': 'Cannot connect to trainer server',
                'hint': f'Make sure the trainer server is running at {get_trainer_url()}'
            }), 503
        return jsonify({'error': str(e)}), 500


# Backward compatibility: alias without /api prefix
@trainer_bp.route('/actor/<actor_name>/train', methods=['POST'])
def start_training_compat(actor_name):
    """Backward-compatible alias for /api/actor/<actor_name>/train."""
    return api_start_training(actor_name)


@trainer_bp.route('/api/trainer/jobs', methods=['GET'])
def api_trainer_jobs():
    """Get list of training jobs from the remote server."""
    trainer = get_trainer_config()

    if not trainer.get('enabled', False):
        return jsonify({'jobs': [], 'enabled': False})

    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/api/jobs", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'jobs': [], 'error': response.text})
    except Exception as e:
        return jsonify({'jobs': [], 'error': str(e)})


@trainer_bp.route('/api/trainer/jobs/<job_id>', methods=['GET'])
def api_trainer_job_status(job_id):
    """Get status of a specific training job."""
    try:
        import requests
        response = requests.get(
            f"{get_trainer_url()}/api/jobs/{job_id}", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'error': 'Job not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@trainer_bp.route('/api/trainer/models', methods=['GET'])
def api_trainer_models():
    """Get list of trained models/adapters from the remote server."""
    trainer = get_trainer_config()

    if not trainer.get('enabled', False):
        return jsonify({'models': [], 'enabled': False})

    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/api/models", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'models': [], 'error': response.text})
    except Exception as e:
        return jsonify({'models': [], 'error': str(e)})


@trainer_bp.route('/api/trainer/models/<model_name>/rename', methods=['POST'])
def api_trainer_rename_model(model_name):
    """Rename a trained LoRA adapter on the remote server."""
    import re

    trainer = get_trainer_config()

    if not trainer.get('enabled', False):
        return jsonify({'error': 'Trainer not enabled'}), 400

    data = request.json
    new_name = data.get('new_name', '').strip()

    if not new_name:
        return jsonify({'error': 'new_name is required'}), 400

    # Validate new name (alphanumeric, underscore, hyphen only)
    if not re.match(r'^[a-zA-Z0-9_-]+$', new_name):
        return jsonify({'error': 'Invalid name. Use only letters, numbers, underscores, and hyphens.'}), 400

    try:
        import requests
        response = requests.post(
            f"{get_trainer_url()}/api/models/{model_name}/rename",
            json={'new_name': new_name},
            timeout=30
        )
        if response.ok:
            return jsonify(response.json())
        return jsonify({'error': response.json().get('error', 'Rename failed')}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@trainer_bp.route('/api/trainer/models/<model_name>', methods=['DELETE'])
def api_trainer_delete_model(model_name):
    """Delete a trained LoRA adapter on the remote server."""
    trainer = get_trainer_config()

    if not trainer.get('enabled', False):
        return jsonify({'error': 'Trainer not enabled'}), 400

    try:
        import requests
        response = requests.delete(
            f"{get_trainer_url()}/api/models/{model_name}",
            timeout=30
        )
        if response.ok:
            return jsonify(response.json())
        return jsonify({'error': response.json().get('error', 'Delete failed')}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500
