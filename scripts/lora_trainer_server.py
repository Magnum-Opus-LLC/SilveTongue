#!/usr/bin/env python3
"""
LoRA Training Server for Silvertongue
Optimized for NVIDIA NIM Unsloth container on Blackwell GPUs.

Reference: https://build.nvidia.com/spark/unsloth/instructions

Usage (inside NIM container):
    python lora_trainer_server.py --port 5050

The server provides REST APIs for:
    - Starting LoRA training jobs
    - Monitoring training progress
    - Managing trained adapters
"""

import os
import json
import threading
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration (can be overridden via environment or CLI args)
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LORA_OUTPUT_DIR = Path(os.environ.get(
    "LORA_OUTPUT_DIR", "/mnt/silvertongueArray/lora_adapters"))
DATASET_DIR = Path(os.environ.get(
    "DATASET_DIR", "/mnt/silvertongueArray/array/data"))

# Training state
active_jobs = {}
completed_jobs = {}


def get_default_lora_config():
    """Default LoRA hyperparameters optimized for Qwen2.5-VL on B200."""
    return {
        "r": 32,                      # LoRA rank
        "lora_alpha": 64,             # LoRA alpha (usually 2x rank)
        "lora_dropout": 0.05,
        "target_modules": [           # Qwen2.5 target modules
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        # Training params
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "max_seq_length": 2048,
        "fp16": False,
        "bf16": True,                 # Use BF16 on B200
    }


def train_lora_adapter(job_id: str, actor_name: str, dataset_path: Path, config: dict):
    """
    Train a LoRA adapter for an actor.
    Runs in a background thread.
    """
    try:
        active_jobs[job_id]["status"] = "loading"
        active_jobs[job_id]["progress"] = 0
        
        # Try to use Unsloth first (much faster), fall back to PEFT
        try:
            from unsloth import FastLanguageModel
            use_unsloth = True
            print(f"[{job_id}] Using Unsloth for fast training")
        except ImportError:
            use_unsloth = False
            print(f"[{job_id}] Unsloth not available, using PEFT")
        
        if use_unsloth:
            train_with_unsloth(job_id, actor_name, dataset_path, config)
        else:
            train_with_peft(job_id, actor_name, dataset_path, config)
            
    except Exception as e:
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)
        active_jobs[job_id]["traceback"] = traceback.format_exc()
        print(f"[{job_id}] Training failed: {e}")
        traceback.print_exc()


def train_with_unsloth(job_id: str, actor_name: str, dataset_path: Path, config: dict):
    """
    Train using Unsloth (2-5x faster than standard PEFT).
    Optimized for NVIDIA NIM container on Blackwell GPUs.
    Reference: https://build.nvidia.com/spark/unsloth
    """
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    active_jobs[job_id]["status"] = "loading_model"
    
    # Load model with Unsloth optimizations
    # 4bit loading uses less VRAM and is faster on Blackwell
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=config["max_seq_length"],
        dtype=None,  # Auto-detect (will use bfloat16 on Blackwell)
        load_in_4bit=True,
    )
    
    # Apply chat template for instruction following
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",  # Qwen uses ChatML format
    )
    
    active_jobs[job_id]["progress"] = 10
    active_jobs[job_id]["status"] = "configuring_lora"
    
    # Add LoRA adapters with Unsloth optimizations
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=42,
        use_rslora=True,  # Rank-stabilized LoRA for better convergence
        loftq_config=None,
    )
    
    active_jobs[job_id]["progress"] = 20
    active_jobs[job_id]["status"] = "loading_dataset"
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    
    # Format for training using ChatML (Qwen format)
    def format_for_training(example):
        # Convert Alpaca format to ChatML messages
        messages = [{"role": "user", "content": example['instruction']}]
        if example.get("input"):
            messages[0]["content"] += f"\n\n{example['input']}"
        messages.append({"role": "assistant", "content": example['output']})
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    dataset = dataset.map(format_for_training)
    
    active_jobs[job_id]["progress"] = 30
    active_jobs[job_id]["status"] = "training"
    active_jobs[job_id]["total_steps"] = len(dataset) * config["num_epochs"] // (config["batch_size"] * config["gradient_accumulation_steps"])
    
    output_dir = LORA_OUTPUT_DIR / actor_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments optimized for Blackwell
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        fp16=not config["bf16"],  # Use FP16 if BF16 disabled
        bf16=config["bf16"],      # BF16 preferred on Blackwell
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",       # 8-bit Adam for memory efficiency
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",         # Disable wandb/tensorboard
    )
    
    # Custom callback to update progress
    class ProgressCallback:
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0:
                progress = 30 + int(60 * state.global_step / state.max_steps)
                active_jobs[job_id]["progress"] = min(progress, 90)
                active_jobs[job_id]["current_step"] = state.global_step
                active_jobs[job_id]["loss"] = state.log_history[-1].get("loss") if state.log_history else None
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    active_jobs[job_id]["progress"] = 95
    active_jobs[job_id]["status"] = "saving"
    
    # Save the LoRA adapter
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="lora",
    )
    
    # Save training info
    info = {
        "actor": actor_name,
        "base_model": BASE_MODEL,
        "dataset": str(dataset_path),
        "config": config,
        "trained_at": datetime.now().isoformat(),
        "examples": len(dataset),
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    active_jobs[job_id]["status"] = "completed"
    active_jobs[job_id]["progress"] = 100
    active_jobs[job_id]["output_dir"] = str(output_dir)
    
    # Move to completed
    completed_jobs[job_id] = active_jobs.pop(job_id)
    print(f"[{job_id}] Training completed! Adapter saved to {output_dir}")


def train_with_peft(job_id: str, actor_name: str, dataset_path: Path, config: dict):
    """Train using standard PEFT (fallback if Unsloth unavailable)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    
    active_jobs[job_id]["status"] = "loading_model"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    active_jobs[job_id]["progress"] = 10
    active_jobs[job_id]["status"] = "configuring_lora"
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias=config["bias"],
        task_type=config["task_type"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    active_jobs[job_id]["progress"] = 20
    active_jobs[job_id]["status"] = "loading_dataset"
    
    # Load and preprocess dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    
    def tokenize_function(example):
        if example.get("input"):
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=config["max_seq_length"],
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    
    active_jobs[job_id]["progress"] = 30
    active_jobs[job_id]["status"] = "training"
    
    output_dir = LORA_OUTPUT_DIR / actor_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        bf16=config["bf16"],
        logging_steps=10,
        save_strategy="epoch",
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    active_jobs[job_id]["progress"] = 95
    active_jobs[job_id]["status"] = "saving"
    
    # Save adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save info
    info = {
        "actor": actor_name,
        "base_model": BASE_MODEL,
        "dataset": str(dataset_path),
        "config": config,
        "trained_at": datetime.now().isoformat(),
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    active_jobs[job_id]["status"] = "completed"
    active_jobs[job_id]["progress"] = 100
    active_jobs[job_id]["output_dir"] = str(output_dir)
    
    completed_jobs[job_id] = active_jobs.pop(job_id)
    print(f"[{job_id}] Training completed! Adapter saved to {output_dir}")


# ==================== API Routes ====================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "base_model": BASE_MODEL,
        "lora_output_dir": str(LORA_OUTPUT_DIR),
        "active_jobs": len(active_jobs),
    })


@app.route("/api/train", methods=["POST"])
def start_training():
    """
    Start a LoRA training job.
    
    Request body:
    {
        "actor_name": "steve",
        "dataset_path": "/path/to/dataset.jsonl",  # or relative to DATASET_DIR
        "config": { ... }  # optional, uses defaults if not provided
    }
    """
    data = request.json
    actor_name = data.get("actor_name")
    dataset_path = data.get("dataset_path")
    config = data.get("config", {})
    
    if not actor_name:
        return jsonify({"error": "actor_name is required"}), 400
    
    if not dataset_path:
        return jsonify({"error": "dataset_path is required"}), 400
    
    # Resolve dataset path
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = DATASET_DIR / actor_name / "lora_training" / dataset_path
    
    if not dataset_path.exists():
        return jsonify({"error": f"Dataset not found: {dataset_path}"}), 404
    
    # Merge with defaults
    full_config = get_default_lora_config()
    full_config.update(config)
    
    # Create job
    job_id = f"{actor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    active_jobs[job_id] = {
        "job_id": job_id,
        "actor_name": actor_name,
        "dataset_path": str(dataset_path),
        "config": full_config,
        "status": "queued",
        "progress": 0,
        "started_at": datetime.now().isoformat(),
    }
    
    # Start training in background thread
    thread = threading.Thread(
        target=train_lora_adapter,
        args=(job_id, actor_name, dataset_path, full_config),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": f"Training started for {actor_name}",
    })


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """List all training jobs."""
    return jsonify({
        "active": list(active_jobs.values()),
        "completed": list(completed_jobs.values()),
    })


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    """Get status of a specific job."""
    if job_id in active_jobs:
        return jsonify(active_jobs[job_id])
    elif job_id in completed_jobs:
        return jsonify(completed_jobs[job_id])
    else:
        return jsonify({"error": "Job not found"}), 404


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Cancel a training job (best effort)."""
    if job_id in active_jobs:
        active_jobs[job_id]["status"] = "cancelling"
        # Note: Actually interrupting training requires more complex logic
        return jsonify({"success": True, "message": "Cancel requested"})
    return jsonify({"error": "Job not found or already completed"}), 404


@app.route("/api/adapters", methods=["GET"])
def list_adapters():
    """List all trained LoRA adapters."""
    adapters = []
    
    if LORA_OUTPUT_DIR.exists():
        for adapter_dir in LORA_OUTPUT_DIR.iterdir():
            if adapter_dir.is_dir():
                info_file = adapter_dir / "training_info.json"
                adapter_config = adapter_dir / "adapter_config.json"
                
                if adapter_config.exists() or info_file.exists():
                    adapter_info = {"name": adapter_dir.name, "path": str(adapter_dir)}
                    
                    if info_file.exists():
                        with open(info_file) as f:
                            adapter_info["training_info"] = json.load(f)
                    
                    adapters.append(adapter_info)
    
    return jsonify({"adapters": adapters})


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current server configuration."""
    return jsonify({
        "base_model": BASE_MODEL,
        "lora_output_dir": str(LORA_OUTPUT_DIR),
        "dataset_dir": str(DATASET_DIR),
        "default_lora_config": get_default_lora_config(),
    })


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Silvertongue LoRA Training Server")
    parser.add_argument("--port", type=int, default=5050, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model for training")
    parser.add_argument("--lora-output", default=str(LORA_OUTPUT_DIR), help="Output directory for LoRA adapters")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Dataset directory")
    
    args = parser.parse_args()
    
    BASE_MODEL = args.base_model
    LORA_OUTPUT_DIR = Path(args.lora_output)
    DATASET_DIR = Path(args.dataset_dir)
    
    print("=" * 60)
    print("Silvertongue LoRA Training Server")
    print("=" * 60)
    print(f"Base Model:  {BASE_MODEL}")
    print(f"LoRA Output: {LORA_OUTPUT_DIR}")
    print(f"Dataset Dir: {DATASET_DIR}")
    print(f"Listening:   http://{args.host}:{args.port}")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
