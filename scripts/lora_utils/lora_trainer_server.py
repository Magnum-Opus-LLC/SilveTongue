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
import psutil, sys
import importlib

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
        
def get_next_version(actor_name: str) -> str:
    """Get the next version number for an actor's adapter (v1, v2, etc.)."""
    actor_dir = LORA_OUTPUT_DIR / actor_name
    if not actor_dir.exists():
        return "v1"
    
    existing_versions = []
    for item in actor_dir.iterdir():
        if item.is_dir() and item.name.startswith('v'):
            try:
                version_num = int(item.name[1:])
                existing_versions.append(version_num)
            except ValueError:
                pass
    
    if not existing_versions:
        return "v1"
    
    return f"v{max(existing_versions) + 1}"


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


def train_lora_adapter(job_id: str, actor_name: str, dataset_path: Path, config: dict, base_adapter: Path = None):
    """
    Train a LoRA adapter for an actor.
    Runs in a background thread.
    
    Args:
        job_id: Unique job identifier
        actor_name: Name of the actor
        dataset_path: Path to training dataset
        config: Training configuration
        base_adapter: Optional path to existing adapter for continuous training (LoRA-on-LoRA)
    """
    try:
        active_jobs[job_id]["status"] = "loading"
        active_jobs[job_id]["progress"] = 0
        
        if base_adapter:
            print(f"[{job_id}] Continuous training from: {base_adapter}")
        
        # Try to use Unsloth first (much faster), fall back to PEFT
        try:
            from unsloth import FastLanguageModel
            use_unsloth = True
            print(f"[{job_id}] Using Unsloth for fast training")
        except ImportError:
            use_unsloth = False
            print(f"[{job_id}] Unsloth not available, using PEFT")
        
        if use_unsloth:
            train_with_unsloth(job_id, actor_name, dataset_path, config, base_adapter)
        else:
            train_with_peft(job_id, actor_name, dataset_path, config, base_adapter)
            
    except Exception as e:
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)
        active_jobs[job_id]["traceback"] = traceback.format_exc()
        print(f"[{job_id}] Training failed: {e}")
        traceback.print_exc()


def train_with_unsloth(job_id: str, actor_name: str, dataset_path: Path, config: dict, base_adapter: Path = None):
    """
    Train using Unsloth (2-5x faster than standard PEFT).
    Optimized for NVIDIA NIM container on Blackwell GPUs.
    Reference: https://build.nvidia.com/spark/unsloth
    
    Args:
        base_adapter: Optional path to existing LoRA adapter for continuous training
    """
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # --- Fix Unsloth compiled trainer missing psutil import ---
    try:
        import UnslothSFTTrainer as UST  # this matches the traceback module name
        UST.psutil = psutil
        print(f"[{job_id}] Patched runtime: injected psutil into UnslothSFTTrainer")
    except Exception as e:
        # fallback: patch any loaded module that contains it
        patched = False
        for name, mod in list(sys.modules.items()):
            if name and "UnslothSFTTrainer" in name:
                setattr(mod, "psutil", psutil)
                patched = True
        print(f"[{job_id}] Runtime patch fallback applied={patched} err={e}")

    active_jobs[job_id]["status"] = "loading_model"
    
    # Allow base_model override from config
    model_to_load = config.get("base_model", BASE_MODEL)
    
    # Load model with Unsloth optimizations
    # 4bit loading uses less VRAM and is faster on Blackwell
    if base_adapter:
        # Continuous training: load model with existing LoRA adapter
        print(f"[{job_id}] Loading base model with existing adapter: {base_adapter}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(base_adapter),  # Load from adapter directory
            max_seq_length=config["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        active_jobs[job_id]["base_adapter_loaded"] = str(base_adapter)
    else:
        # Fresh training: load base model
        print(f"[{job_id}] Loading base model: {model_to_load}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_to_load,
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
    
    # Add/configure LoRA adapters with Unsloth optimizations
    # For continuous training, this will train the loaded adapter further
    if not base_adapter:
        # Only add new LoRA layers if not doing continuous training
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
    else:
        # For continuous training, just enable gradient checkpointing
        # The adapter is already loaded from base_adapter
        print(f"[{job_id}] Continuing training on existing adapter weights")
    
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
    
    # Determine version number for this training run
    version = get_next_version(actor_name)
    output_dir = LORA_OUTPUT_DIR / actor_name / version
    output_dir.mkdir(parents=True, exist_ok=True)
    active_jobs[job_id]["version"] = version
    
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
    dataset_num_proc = max(psutil.cpu_count()+4, 2)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        dataset_num_proc=dataset_num_proc,
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    active_jobs[job_id]["progress"] = 95
    active_jobs[job_id]["status"] = "saving"
    
    # Save ONLY the LoRA adapter (not merged model)
    # Use save_pretrained() which saves just the adapter weights (~50-100MB)
    # NOT save_pretrained_merged() which saves the full merged model (14GB+)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training info
    info = {
        "actor": actor_name,
        "version": version,
        "base_model": BASE_MODEL,
        "base_adapter": str(base_adapter) if base_adapter else None,  # Track continuous training lineage
        "dataset": str(dataset_path),
        "config": config,
        "trained_at": datetime.now().isoformat(),
        "examples": len(dataset),
        "continuous_training": base_adapter is not None,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    active_jobs[job_id]["status"] = "completed"
    active_jobs[job_id]["progress"] = 100
    active_jobs[job_id]["output_dir"] = str(output_dir)
    
    # Move to completed
    completed_jobs[job_id] = active_jobs.pop(job_id)
    print(f"[{job_id}] Training completed! Adapter saved to {output_dir}")


def train_with_peft(job_id: str, actor_name: str, dataset_path: Path, config: dict, base_adapter: Path = None):
    """
    Train using standard PEFT (fallback if Unsloth unavailable).
    
    Args:
        base_adapter: Optional path to existing LoRA adapter for continuous training
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from datasets import load_dataset
    
    active_jobs[job_id]["status"] = "loading_model"
    
    # Load base model
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
    
    if base_adapter:
        # Continuous training: load existing adapter
        print(f"[{job_id}] Loading existing adapter for continuous training: {base_adapter}")
        model = PeftModel.from_pretrained(model, str(base_adapter), is_trainable=True)
        active_jobs[job_id]["base_adapter_loaded"] = str(base_adapter)
    else:
        # Fresh training: create new LoRA adapter
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
    
    # Determine version number for this training run
    version = get_next_version(actor_name)
    output_dir = LORA_OUTPUT_DIR / actor_name / version
    output_dir.mkdir(parents=True, exist_ok=True)
    active_jobs[job_id]["version"] = version
    
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
        "version": version,
        "base_model": BASE_MODEL,
        "base_adapter": str(base_adapter) if base_adapter else None,
        "dataset": str(dataset_path),
        "config": config,
        "trained_at": datetime.now().isoformat(),
        "continuous_training": base_adapter is not None,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    active_jobs[job_id]["status"] = "completed"
    active_jobs[job_id]["progress"] = 100
    active_jobs[job_id]["output_dir"] = str(output_dir)
    
    completed_jobs[job_id] = active_jobs.pop(job_id)
    print(f"[{job_id}] Training completed! Adapter saved to {output_dir}")


def check_gpu_available():
    """Check if GPU is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ==================== API Routes ====================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    gpu_available = check_gpu_available()
    gpu_name = None
    if gpu_available:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except:
            pass
    
    return jsonify({
        "status": "healthy",
        "base_model": BASE_MODEL,
        "lora_output_dir": str(LORA_OUTPUT_DIR),
        "active_jobs": len(active_jobs),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
    })


@app.route("/routes", methods=["GET"])
@app.route("/api/routes", methods=["GET"])
def list_routes():
    """List all registered routes for debugging."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods - {'HEAD', 'OPTIONS'}),
            "path": rule.rule
        })
    routes.sort(key=lambda x: x['path'])
    return jsonify({"routes": routes, "total": len(routes)})


@app.route("/api/train", methods=["POST"])
def start_training():
    """
    Start a LoRA training job.
    
    Request body:
    {
        "actor_name": "steve",
        "dataset_path": "/path/to/dataset.jsonl",  # or "train_file" - both accepted
        "base_adapter": "/path/to/adapter",  # optional: for continuous training (LoRA-on-LoRA)
        "config": { ... }  # optional, uses defaults if not provided
    }
    """
    data = request.json
    actor_name = data.get("actor_name")
    # Accept both 'dataset_path' and 'train_file' for compatibility
    dataset_path = data.get("dataset_path") or data.get("train_file")
    base_adapter = data.get("base_adapter")  # For continuous training
    base_model_override = data.get("base_model")  # Optional: override the default base model
    config = data.get("config", {})
    
    # Also accept flat config parameters (merge them into config)
    flat_params = ['lora_r', 'lora_alpha', 'lora_dropout', 'learning_rate', 
                   'num_epochs', 'batch_size', 'gradient_accumulation_steps', 'max_seq_length']
    for param in flat_params:
        if param in data and data[param] is not None:
            # Map some field names
            config_key = 'r' if param == 'lora_r' else param
            config[config_key] = data[param]
    
    # Handle base_model override
    if base_model_override:
        config['base_model'] = base_model_override
    
    if not actor_name:
        return jsonify({"error": "actor_name is required"}), 400
    
    if not dataset_path:
        return jsonify({"error": "dataset_path or train_file is required"}), 400
    
    # Resolve dataset path
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = DATASET_DIR / actor_name / "lora_training" / dataset_path
    
    if not dataset_path.exists():
        return jsonify({"error": f"Dataset not found: {dataset_path}"}), 404
    
    # Validate base_adapter if provided (for continuous training)
    base_adapter_path = None
    if base_adapter:
        base_adapter_path = Path(base_adapter)
        print(f"[INFO] Checking base adapter path: {base_adapter_path}")
        print(f"[INFO] Path exists: {base_adapter_path.exists()}")
        if not base_adapter_path.exists():
            # Try to help debug by listing what's in LORA_OUTPUT_DIR
            available = []
            if LORA_OUTPUT_DIR.exists():
                for d in LORA_OUTPUT_DIR.iterdir():
                    if d.is_dir():
                        available.append(str(d))
            return jsonify({
                "error": f"Base adapter not found: {base_adapter}",
                "lora_output_dir": str(LORA_OUTPUT_DIR),
                "available_adapters": available[:10]  # First 10
            }), 404
        if not (base_adapter_path / "adapter_config.json").exists():
            return jsonify({"error": f"Invalid adapter (no adapter_config.json): {base_adapter}"}), 400
        print(f"[INFO] Continuous training from base adapter: {base_adapter}")
    
    # Merge with defaults
    full_config = get_default_lora_config()
    full_config.update(config)
    
    # Create job
    job_id = f"{actor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    active_jobs[job_id] = {
        "job_id": job_id,
        "actor_name": actor_name,
        "dataset_path": str(dataset_path),
        "base_adapter": str(base_adapter_path) if base_adapter_path else None,
        "config": full_config,
        "status": "queued",
        "progress": 0,
        "started_at": datetime.now().isoformat(),
    }
    
    # Start training in background thread
    thread = threading.Thread(
        target=train_lora_adapter,
        args=(job_id, actor_name, dataset_path, full_config, base_adapter_path),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": f"Training started for {actor_name}" + (f" (continuing from {base_adapter_path.name})" if base_adapter_path else ""),
    })


# Backward compatibility: alias /train to /api/train
@app.route("/train", methods=["POST"])
def start_training_compat():
    """Backward-compatible alias for /api/train."""
    return start_training()


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


@app.route("/api/models", methods=["GET"])
def list_models():
    """
    List all trained LoRA models/adapters.
    Returns format expected by dashboard: {models: [{actor_name, adapter_path, info, is_valid}, ...]}
    """
    models = []
    
    if LORA_OUTPUT_DIR.exists():
        for adapter_dir in LORA_OUTPUT_DIR.iterdir():
            if adapter_dir.is_dir():
                info_file = adapter_dir / "training_info.json"
                adapter_config = adapter_dir / "adapter_config.json"
                
                # Check if this is a valid adapter directory (has training_info or adapter_config)
                if adapter_config.exists() or info_file.exists():
                    # is_valid = has adapter_config.json (true LoRA adapter, not merged model)
                    is_valid = adapter_config.exists()
                    
                    model_info = {
                        "actor_name": adapter_dir.name,
                        "adapter_path": str(adapter_dir),
                        "is_valid": is_valid,
                        "info": {}
                    }
                    
                    if info_file.exists():
                        try:
                            with open(info_file) as f:
                                model_info["info"] = json.load(f)
                        except json.JSONDecodeError:
                            pass
                    
                    # Add base model info if available
                    if "base_model" not in model_info["info"]:
                        model_info["info"]["base_model"] = BASE_MODEL
                    
                    models.append(model_info)
    
    return jsonify({"models": models})


@app.route("/api/models/<model_name>/rename", methods=["POST"])
def rename_model(model_name):
    """Rename a trained LoRA adapter."""
    import shutil
    import re
    
    data = request.json
    new_name = data.get("new_name", "").strip()
    
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400
    
    # Validate new name
    if not re.match(r'^[a-zA-Z0-9_-]+$', new_name):
        return jsonify({"error": "Invalid name. Use only letters, numbers, underscores, and hyphens."}), 400
    
    old_path = LORA_OUTPUT_DIR / model_name
    new_path = LORA_OUTPUT_DIR / new_name
    
    if not old_path.exists():
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    
    if new_path.exists():
        return jsonify({"error": f"Model '{new_name}' already exists"}), 409
    
    try:
        # Rename the directory
        shutil.move(str(old_path), str(new_path))
        
        # Update training_info.json if it exists
        info_file = new_path / "training_info.json"
        if info_file.exists():
            with open(info_file, "r") as f:
                info = json.load(f)
            info["actor"] = new_name
            info["renamed_from"] = model_name
            with open(info_file, "w") as f:
                json.dump(info, f, indent=2)
        
        return jsonify({
            "success": True,
            "old_name": model_name,
            "new_name": new_name,
            "path": str(new_path)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<model_name>", methods=["DELETE"])
def delete_model(model_name):
    """Delete a trained LoRA adapter."""
    import shutil
    
    model_path = LORA_OUTPUT_DIR / model_name
    
    if not model_path.exists():
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    
    try:
        shutil.rmtree(str(model_path))
        return jsonify({
            "success": True,
            "deleted": model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current server configuration."""
    # Common models available for LoRA training
    available_models = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct", 
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "nvidia/Nemotron-Mini-4B-Instruct",
    ]
    
    return jsonify({
        "base_model": BASE_MODEL,
        "available_models": available_models,
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
