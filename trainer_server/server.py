#!/usr/bin/env python3
"""
Silvertongue LoRA Training Server
=================================
A Flask server that runs on a GPU machine (e.g., DGX Spark) to handle
LoRA fine-tuning requests from the main Silvertongue application.

Deploy this on your DGX Spark and configure the main app to send training
jobs to this server.

Usage:
    python server.py --host 0.0.0.0 --port 5050 --data-dir /path/to/shared/nas
"""

import os
import json
import uuid
import threading
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global state for tracking jobs
training_jobs: Dict[str, 'TrainingJob'] = {}
current_job: Optional[str] = None
job_lock = threading.Lock()


@dataclass
class TrainingJob:
    """Represents a LoRA training job."""
    job_id: str
    actor_name: str
    status: str = "queued"  # queued, running, completed, failed
    progress: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    # Training config
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    train_file: str = ""
    val_file: str = ""
    output_dir: str = ""
    
    # Hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    
    # Results
    final_loss: Optional[float] = None
    adapter_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingConfig:
    """Server configuration."""
    def __init__(self):
        self.data_dir = os.environ.get('SILVERTONGUE_DATA_DIR', '/mnt/nas/silvertongue/array')
        self.output_base = os.environ.get('SILVERTONGUE_OUTPUT_DIR', '/mnt/nas/silvertongue/models')
        self.default_base_model = os.environ.get('SILVERTONGUE_BASE_MODEL', 'meta-llama/Llama-3.2-3B-Instruct')
        self.max_concurrent_jobs = 1  # GPU memory constraint
        
config = TrainingConfig()


def run_training(job: TrainingJob):
    """Execute the actual LoRA training."""
    global current_job
    
    try:
        job.status = "running"
        job.started_at = datetime.now().isoformat()
        
        # Create output directory
        output_dir = Path(config.output_base) / job.actor_name / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = str(output_dir)
        
        # Generate training script
        training_script = generate_training_script(job)
        script_path = output_dir / "train.py"
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        # Generate config file
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(job.to_dict(), f, indent=2)
        
        # Run training
        print(f"[{job.job_id}] Starting training for actor: {job.actor_name}")
        
        result = subprocess.run(
            ['python', str(script_path)],
            cwd=str(output_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            job.status = "completed"
            job.progress = 100.0
            job.adapter_path = str(output_dir / "adapter")
            print(f"[{job.job_id}] Training completed successfully")
        else:
            job.status = "failed"
            job.error = result.stderr[-2000:] if result.stderr else "Unknown error"
            print(f"[{job.job_id}] Training failed: {job.error}")
            
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        print(f"[{job.job_id}] Training error: {e}")
        
    finally:
        job.completed_at = datetime.now().isoformat()
        with job_lock:
            current_job = None
        
        # Check for queued jobs
        process_queue()


def process_queue():
    """Process the next job in queue if GPU is available."""
    global current_job
    
    with job_lock:
        if current_job is not None:
            return
            
        # Find next queued job
        for job_id, job in training_jobs.items():
            if job.status == "queued":
                current_job = job_id
                thread = threading.Thread(target=run_training, args=(job,))
                thread.daemon = True
                thread.start()
                return


def generate_training_script(job: TrainingJob) -> str:
    """Generate a training script for the job."""
    return f'''#!/usr/bin/env python3
"""
Auto-generated LoRA Training Script for {job.actor_name}
Generated: {datetime.now().isoformat()}
Job ID: {job.job_id}
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
BASE_MODEL = "{job.base_model}"
TRAIN_FILE = "{job.train_file}"
VAL_FILE = "{job.val_file}"
OUTPUT_DIR = "./adapter"

# LoRA Configuration
LORA_R = {job.lora_r}
LORA_ALPHA = {job.lora_alpha}
LORA_DROPOUT = {job.lora_dropout}

# Training Configuration
LEARNING_RATE = {job.learning_rate}
NUM_EPOCHS = {job.num_epochs}
BATCH_SIZE = {job.batch_size}
GRADIENT_ACCUMULATION = {job.gradient_accumulation_steps}
MAX_SEQ_LENGTH = {job.max_seq_length}


def format_instruction(example):
    """Format training example into instruction format."""
    if example.get('input'):
        text = f"### Instruction:\\n{{example['instruction']}}\\n\\n### Input:\\n{{example['input']}}\\n\\n### Response:\\n{{example['output']}}"
    else:
        text = f"### Instruction:\\n{{example['instruction']}}\\n\\n### Response:\\n{{example['output']}}"
    return {{"text": text}}


def main():
    print(f"Loading base model: {{BASE_MODEL}}")
    
    # Quantization config for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"Loading training data from: {{TRAIN_FILE}}")
    dataset = load_dataset("json", data_files={{"train": TRAIN_FILE, "validation": VAL_FILE}})
    
    # Format dataset
    dataset = dataset.map(format_instruction)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        report_to="none",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    print(f"Saving adapter to: {{OUTPUT_DIR}}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training info
    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump({{
            "base_model": BASE_MODEL,
            "actor_name": "{job.actor_name}",
            "job_id": "{job.job_id}",
            "lora_config": {{
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
            }},
            "training_config": {{
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            }},
        }}, f, indent=2)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
'''


# ==================== API Routes ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "server": "silvertongue-trainer",
        "version": "1.0.0",
        "gpu_available": torch_gpu_available(),
        "current_job": current_job,
        "queued_jobs": len([j for j in training_jobs.values() if j.status == "queued"]),
    })


@app.route('/api/train', methods=['POST'])
def submit_training_job():
    """Submit a new LoRA training job."""
    data = request.json
    
    if not data.get('actor_name'):
        return jsonify({"error": "actor_name is required"}), 400
    
    actor_name = data['actor_name']
    
    # Locate training data
    train_file = data.get('train_file')
    val_file = data.get('val_file')
    
    if not train_file:
        # Try to find in standard location
        data_path = Path(config.data_dir) / "data" / actor_name / "lora_training"
        train_file = str(data_path / f"{actor_name}_lora_train.jsonl")
        val_file = str(data_path / f"{actor_name}_lora_val.jsonl")
    
    if not Path(train_file).exists():
        return jsonify({
            "error": f"Training file not found: {train_file}",
            "hint": "Run the sleep cycle first to generate training data"
        }), 404
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = TrainingJob(
        job_id=job_id,
        actor_name=actor_name,
        train_file=train_file,
        val_file=val_file if Path(val_file).exists() else train_file,
        base_model=data.get('base_model', config.default_base_model),
        lora_r=data.get('lora_r', 16),
        lora_alpha=data.get('lora_alpha', 32),
        lora_dropout=data.get('lora_dropout', 0.05),
        learning_rate=data.get('learning_rate', 2e-4),
        num_epochs=data.get('num_epochs', 3),
        batch_size=data.get('batch_size', 4),
        gradient_accumulation_steps=data.get('gradient_accumulation_steps', 4),
        max_seq_length=data.get('max_seq_length', 2048),
    )
    
    training_jobs[job_id] = job
    process_queue()
    
    return jsonify({
        "job_id": job_id,
        "status": job.status,
        "message": f"Training job queued for actor: {actor_name}",
        "config": job.to_dict()
    })


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all training jobs."""
    return jsonify({
        "jobs": [job.to_dict() for job in training_jobs.values()],
        "current_job": current_job,
    })


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get status of a specific job."""
    if job_id not in training_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(training_jobs[job_id].to_dict())


@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Cancel a queued job."""
    if job_id not in training_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = training_jobs[job_id]
    
    if job.status == "running":
        return jsonify({"error": "Cannot cancel running job"}), 400
    
    if job.status in ["completed", "failed"]:
        return jsonify({"error": "Job already finished"}), 400
    
    job.status = "cancelled"
    return jsonify({"message": "Job cancelled", "job": job.to_dict()})


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available trained models/adapters."""
    models = []
    output_base = Path(config.output_base)
    
    if output_base.exists():
        for actor_dir in output_base.iterdir():
            if actor_dir.is_dir():
                for job_dir in actor_dir.iterdir():
                    adapter_path = job_dir / "adapter"
                    if adapter_path.exists():
                        info_file = adapter_path / "training_info.json"
                        info = {}
                        if info_file.exists():
                            with open(info_file) as f:
                                info = json.load(f)
                        
                        models.append({
                            "actor_name": actor_dir.name,
                            "job_id": job_dir.name,
                            "adapter_path": str(adapter_path),
                            "info": info
                        })
    
    return jsonify({"models": models})


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get server configuration."""
    return jsonify({
        "data_dir": config.data_dir,
        "output_base": config.output_base,
        "default_base_model": config.default_base_model,
        "max_concurrent_jobs": config.max_concurrent_jobs,
    })


@app.route('/api/config', methods=['PUT'])
def update_config():
    """Update server configuration."""
    data = request.json
    
    if 'data_dir' in data:
        config.data_dir = data['data_dir']
    if 'output_base' in data:
        config.output_base = data['output_base']
    if 'default_base_model' in data:
        config.default_base_model = data['default_base_model']
    
    return jsonify({"message": "Configuration updated", "config": get_config().json})


def torch_gpu_available() -> bool:
    """Check if PyTorch GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Silvertongue LoRA Training Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5050, help='Port to listen on')
    parser.add_argument('--data-dir', help='Path to shared data directory (NAS)')
    parser.add_argument('--output-dir', help='Path to save trained models')
    parser.add_argument('--base-model', help='Default base model for fine-tuning')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_base = args.output_dir
    if args.base_model:
        config.default_base_model = args.base_model
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Silvertongue LoRA Training Server                  ║
╠══════════════════════════════════════════════════════════════╣
║  Data Directory:   {config.data_dir:<40} ║
║  Output Directory: {config.output_base:<40} ║
║  Base Model:       {config.default_base_model:<40} ║
║  GPU Available:    {str(torch_gpu_available()):<40} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
