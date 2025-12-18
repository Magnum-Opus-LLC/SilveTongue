# Dataset Management Guide

## Overview

Silvertongue now supports **timestamped datasets** and **dataset merging** to track learning sessions and combine training data efficiently.

## Timestamped Datasets

### How It Works

Every time an actor goes to sleep (consolidates memories), a **new timestamped dataset** is created:

```
steve_lora_dataset_20241217_143022.jsonl
steve_lora_dataset_20241217_165430.jsonl
steve_lora_dataset_20241218_091245.jsonl
```

Each dataset represents a **single learning session** and includes:
- All knowledge gained in that session
- Papers studied and insights
- Conversations and experiences
- Automatically generated training examples

### Benefits

1. **Session Tracking**: Know exactly which training data came from which learning session
2. **Incremental Training**: Train on new knowledge without re-training on old data
3. **Quality Control**: Review and select specific sessions for training
4. **History**: Track knowledge evolution over time

### Dataset Metadata

Each dataset includes a `.summary.json` file with:

```json
{
  "actor_name": "steve",
  "generated_at": "2024-12-17T14:30:22",
  "session_id": "learning_session_20241217_143000",
  "total_examples": 47,
  "by_category": {
    "concept": 23,
    "paper_insight": 15,
    "conversation": 9
  },
  "memory_stats": {
    "core_knowledge": 12,
    "experiences": 8,
    "total_papers_studied": 3
  }
}
```

## Merging Datasets

### Why Merge?

As your actor learns over multiple sessions, you may want to:
- Combine datasets from several sessions into one training file
- Remove duplicate examples across sessions
- Create a comprehensive dataset for full fine-tuning

### Using the Dashboard

1. **Go to Sleep Tab**
2. **View Available Datasets**
   - All timestamped datasets are listed with their stats
   - Each shows generation time, example count, and categories

3. **Select Datasets to Merge**
   - Check the boxes next to datasets you want to merge
   - Must select at least 2 datasets

4. **Click "Merge Selected"**
   - Automatically removes duplicates
   - Creates new timestamped merged dataset
   - Shows statistics about the merge

### Example Merge Flow

```
📋 Original Datasets:
├── steve_lora_dataset_20241217_143022.jsonl (47 examples)
├── steve_lora_dataset_20241217_165430.jsonl (35 examples)
└── steve_lora_dataset_20241218_091245.jsonl (52 examples)
     Total: 134 examples

🔀 After Merge:
└── steve_merged_dataset_20241218_102000.jsonl (118 examples)
     Duplicates removed: 16
```

## API Usage

### Automatic Timestamping

When calling the sleep endpoint, timestamps are automatic:

```python
POST /api/actor/steve/sleep
{
  "include_dreams": true,
  "format": "alpaca"
}

# Returns:
{
  "success": true,
  "summary": {
    "output_path": "array/data/steve/lora_training/steve_lora_dataset_20241217_143022.jsonl",
    "total_examples": 47,
    "generated_at": "2024-12-17T14:30:22"
  }
}
```

### Programmatic Merging

```python
from src.actions.sleep_processor import merge_datasets

result = merge_datasets(
    actor_name='steve',
    dataset_paths=[
        'array/data/steve/lora_training/steve_lora_dataset_20241217_143022.jsonl',
        'array/data/steve/lora_training/steve_lora_dataset_20241217_165430.jsonl',
        'array/data/steve/lora_training/steve_lora_dataset_20241218_091245.jsonl'
    ],
    output_name=None  # Auto-generates timestamped name
)

print(f"Merged {result['sources_merged']} datasets")
print(f"Total examples: {result['total_examples']}")
print(f"Duplicates removed: {result['duplicates_removed']}")
```

### Merge API Endpoint

```python
POST /api/actor/steve/datasets/merge
{
  "dataset_files": [
    "steve_lora_dataset_20241217_143022.jsonl",
    "steve_lora_dataset_20241217_165430.jsonl",
    "steve_lora_dataset_20241218_091245.jsonl"
  ],
  "output_name": null  # Optional: auto-generates if not provided
}

# Returns:
{
  "success": true,
  "message": "Merged 3 datasets into 118 examples",
  "result": {
    "output_path": "array/data/steve/lora_training/steve_merged_dataset_20241218_102000.jsonl",
    "total_examples": 118,
    "sources_merged": 3,
    "duplicates_removed": 16
  }
}
```

## Deduplication

The merge function automatically removes duplicates based on:
- **Instruction** + **Output** combination
- MD5 hash comparison
- Keeps first occurrence, removes subsequent duplicates

This ensures:
- No redundant training examples
- Efficient dataset size
- Better training quality

## Dataset Formats

All datasets support three formats:

### 1. Alpaca (Default)
```json
{
  "instruction": "What is machine learning?",
  "input": "",
  "output": "Machine learning is a field of AI focused on algorithms that learn from data.",
  "source": "learning_session_20241217",
  "category": "concept"
}
```

### 2. ShareGPT
```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is a field of AI..."}
  ]
}
```

### 3. Full (with metadata)
```json
{
  "instruction": "What is machine learning?",
  "input": "",
  "output": "Machine learning is a field of AI...",
  "source": "learning_session_20241217",
  "category": "concept",
  "domain": "artificial_intelligence",
  "quality_score": 0.95
}
```

## Best Practices

### 1. Session-Based Learning
- Let the actor complete a full learning session before sleeping
- Each sleep creates a cohesive dataset from that session
- Review session results before merging

### 2. Selective Merging
- Don't merge all datasets automatically
- Review quality of each session first
- Merge related sessions (e.g., all sessions on a specific topic)

### 3. Incremental Training
- Train on individual session datasets for quick updates
- Use merged datasets for full model refresh
- Keep original timestamped datasets for history

### 4. Quality Control
- Check example counts and categories before merging
- Review summaries for session quality
- Remove low-quality sessions before merging

## Example Workflow

```python
# 1. Learning Session 1 - Basics
actor.learn("What is machine learning?", max_iterations=5)
actor.sleep()  # Creates: steve_lora_dataset_20241217_143022.jsonl

# 2. Learning Session 2 - Advanced Topics
actor.learn("Explain neural network architectures", max_iterations=5)
actor.sleep()  # Creates: steve_lora_dataset_20241217_165430.jsonl

# 3. Learning Session 3 - Applications
actor.learn("Real-world ML applications", max_iterations=5)
actor.sleep()  # Creates: steve_lora_dataset_20241218_091245.jsonl

# 4. Merge all sessions for comprehensive training
merge_datasets(
    actor_name='steve',
    dataset_paths=[
        'steve_lora_dataset_20241217_143022.jsonl',
        'steve_lora_dataset_20241217_165430.jsonl',
        'steve_lora_dataset_20241218_091245.jsonl'
    ]
)
# Creates: steve_merged_dataset_20241218_102000.jsonl

# 5. Train with merged dataset
# Use for LoRA fine-tuning
```

## Troubleshooting

### Merge Button Disabled
- Must select at least 2 datasets
- Check that datasets exist in lora_training folder

### Duplicates Not Removed
- Merge function compares instruction+output
- Variations in whitespace are considered different
- Case-sensitive comparison

### Missing Timestamps
- Older datasets may not have timestamps
- Manually rename: `actor_lora_dataset.jsonl` → `actor_lora_dataset_20241201_000000.jsonl`

### Large Merged Datasets
- Merged datasets can be large (500+ examples)
- Consider training parameters: lower epochs for larger datasets
- Use batch processing for very large merges

## Future Enhancements

- **Smart Deduplication**: Semantic similarity instead of exact match
- **Dataset Versioning**: Track merged dataset lineage
- **Selective Examples**: Choose specific examples from each session
- **Dataset Preview**: View sample examples before merging
- **Training History**: Track which datasets were used for which LoRA adapters
