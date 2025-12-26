"""
Sleep & training data routes for Silvertongue Dashboard.
"""

import json
from flask import Blueprint, jsonify, request

from ..utils import get_array_dir, import_project_module

sleep_bp = Blueprint('sleep', __name__)


@sleep_bp.route('/api/actor/<actor_name>/sleep', methods=['POST'])
def api_actor_sleep(actor_name):
    """
    Put an actor to sleep - consolidate memories into LoRA training data.

    This processes all of the actor's accumulated knowledge, experiences,
    and learning sessions into training data that can be used to fine-tune
    a model to embody the actor's unique mind.

    Request body:
        deep: bool - If True, condense ALL items (deep sleep). 
                     If False (default), only condense NEW items since last sleep.
        adapter: str - LoRA adapter to use for LLM calls (empty = base model)
    """
    try:
        sleep_processor = import_project_module('src.actions.sleep_processor')
        SleepProcessor = sleep_processor.SleepProcessor

        data = request.json or {}
        # Default to light sleep (incremental)
        deep_sleep = data.get('deep', False)
        adapter = data.get('adapter', '')  # LoRA adapter to use for LLM calls

        # Use verbose=True so we can see consolidation logs
        processor = SleepProcessor(
            actor_name, verbose=True, model_override=adapter if adapter else None)

        # Run sleep cycle - deep or light based on parameter
        if deep_sleep:
            result = processor.deep_sleep()
            mode = "deep"
        else:
            result = processor.sleep()
            mode = "light"

        return jsonify({
            'success': True,
            'actor': actor_name,
            'mode': mode,
            'summary': result,
            'message': f'{mode.capitalize()} sleep complete! Generated {result["total_examples"]} training examples.'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@sleep_bp.route('/api/actor/<actor_name>/sleep/status', methods=['GET'])
def api_actor_sleep_status(actor_name):
    """Get the status of LoRA training data for an actor."""
    try:
        array_dir = get_array_dir()
        lora_dir = array_dir / 'data' / actor_name / 'lora_training'

        if not lora_dir.exists():
            return jsonify({
                'has_data': False,
                'message': 'No sleep data yet. Put the actor to sleep to generate LoRA training data.'
            })

        # Find dataset files
        dataset_files = list(lora_dir.glob('*.jsonl'))
        summary_files = list(lora_dir.glob('*.summary.json'))

        result = {
            'has_data': len(dataset_files) > 0,
            'datasets': []
        }

        for dataset_file in dataset_files:
            # Count lines (examples)
            with open(dataset_file, 'r') as f:
                example_count = sum(1 for _ in f)

            # Load summary if exists
            summary_file = dataset_file.with_suffix('.summary.json')
            summary = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)

            result['datasets'].append({
                'filename': dataset_file.name,
                'path': str(dataset_file),
                'examples': example_count,
                'generated_at': summary.get('generated_at', 'Unknown'),
                'by_category': summary.get('by_category', {}),
                'recommended_params': summary.get('recommended_training_params', {})
            })

        # Sort datasets by generated_at (newest first), fallback to filename
        result['datasets'].sort(
            key=lambda d: d.get('generated_at', ''),
            reverse=True
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@sleep_bp.route('/api/actor/<actor_name>/datasets/merge', methods=['POST'])
def api_merge_datasets(actor_name):
    """
    Merge multiple LoRA datasets from different learning sessions.

    Request body:
    {
        "dataset_files": ["file1.jsonl", "file2.jsonl"],  // Filenames to merge
        "output_name": "merged_dataset.jsonl"  // Optional, auto-generates if not provided
    }
    """
    try:
        sleep_processor = import_project_module('src.actions.sleep_processor')
        merge_datasets = sleep_processor.merge_datasets

        data = request.json or {}
        dataset_files = data.get('dataset_files', [])
        output_name = data.get('output_name', None)

        if not dataset_files or len(dataset_files) < 2:
            return jsonify({
                'error': 'Must provide at least 2 datasets to merge'
            }), 400

        # Build full paths
        array_dir = get_array_dir()
        lora_dir = array_dir / 'data' / actor_name / 'lora_training'
        dataset_paths = [str(lora_dir / filename)
                         for filename in dataset_files]

        # Merge datasets
        result = merge_datasets(actor_name, dataset_paths, output_name)

        return jsonify({
            'success': True,
            'actor': actor_name,
            'result': result,
            'message': f'Merged {result["sources_merged"]} datasets into {result["total_examples"]} examples'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
