#!/usr/bin/env python3
"""
Test the timestamped dataset and merge functionality.

This example demonstrates:
1. How datasets are timestamped per learning session
2. How to merge multiple datasets into one
3. Duplicate removal during merge
"""

from src.utils.knowledge_base import get_array_dir
from src.actions.sleep_processor import merge_datasets
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sample_dataset(actor_name: str, session_id: str, examples: list) -> str:
    """Create a sample dataset for testing."""
    from datetime import datetime

    array_dir = get_array_dir()
    lora_dir = array_dir / 'data' / actor_name / 'lora_training'
    lora_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{actor_name}_lora_dataset_{timestamp}_{session_id}.jsonl"
    filepath = lora_dir / filename

    # Write examples
    with open(filepath, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"✓ Created {filename} with {len(examples)} examples")
    return str(filepath)


def main():
    actor = 'test_actor'

    print("\n🔬 Testing Timestamped Dataset & Merge Functionality\n")
    print("=" * 60)

    # Create sample datasets from different sessions
    print("\n📝 Creating sample datasets from different learning sessions...")

    session1_examples = [
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a field of AI focused on algorithms that learn from data.",
            "source": "learning_session_1",
            "category": "concept"
        },
        {
            "instruction": "Explain supervised learning",
            "input": "",
            "output": "Supervised learning uses labeled data to train models.",
            "source": "learning_session_1",
            "category": "concept"
        },
        {
            "instruction": "What is a neural network?",
            "input": "",
            "output": "A neural network is a computational model inspired by biological neurons.",
            "source": "learning_session_1",
            "category": "concept"
        }
    ]

    session2_examples = [
        {
            "instruction": "What is deep learning?",
            "input": "",
            "output": "Deep learning uses multi-layered neural networks to learn complex patterns.",
            "source": "learning_session_2",
            "category": "concept"
        },
        {
            "instruction": "What is machine learning?",  # Duplicate!
            "input": "",
            "output": "Machine learning is a field of AI focused on algorithms that learn from data.",
            "source": "learning_session_2",
            "category": "concept"
        },
        {
            "instruction": "Explain reinforcement learning",
            "input": "",
            "output": "Reinforcement learning trains agents through rewards and penalties.",
            "source": "learning_session_2",
            "category": "concept"
        }
    ]

    session3_examples = [
        {
            "instruction": "What is transfer learning?",
            "input": "",
            "output": "Transfer learning reuses pre-trained models for new tasks.",
            "source": "learning_session_3",
            "category": "concept"
        },
        {
            "instruction": "What is a neural network?",  # Duplicate!
            "input": "",
            "output": "A neural network is a computational model inspired by biological neurons.",
            "source": "learning_session_3",
            "category": "concept"
        }
    ]

    # Create the datasets
    dataset1 = create_sample_dataset(actor, 'session1', session1_examples)
    dataset2 = create_sample_dataset(actor, 'session2', session2_examples)
    dataset3 = create_sample_dataset(actor, 'session3', session3_examples)

    # Show initial state
    print(f"\n📊 Created 3 datasets:")
    print(f"   Session 1: {len(session1_examples)} examples")
    print(f"   Session 2: {len(session2_examples)} examples (1 duplicate)")
    print(f"   Session 3: {len(session3_examples)} examples (1 duplicate)")
    print(
        f"   Total: {len(session1_examples) + len(session2_examples) + len(session3_examples)} examples")
    print(
        f"   Expected after merge: {len(session1_examples) + len(session2_examples) + len(session3_examples) - 2} (removing 2 duplicates)")

    # Merge the datasets
    print(f"\n🔀 Merging datasets...")
    result = merge_datasets(
        actor_name=actor,
        dataset_paths=[dataset1, dataset2, dataset3],
        output_name=None  # Auto-generate timestamped name
    )

    # Show results
    print("\n✨ Merge Results:")
    print("=" * 60)
    print(f"✓ Success: {result['success']}")
    print(f"📄 Output: {Path(result['output_path']).name}")
    print(f"📊 Total examples: {result['total_examples']}")
    print(f"🔀 Sources merged: {result['sources_merged']}")
    print(f"🗑️  Duplicates removed: {result['duplicates_removed']}")

    print("\n📋 Source datasets:")
    for src in result['summary']['source_datasets']:
        print(f"   • {src['filename']}: {src['examples']} examples")

    print("\n🎉 Merge complete! The merged dataset contains unique examples from all sessions.")
    print(f"\n💾 Merged dataset saved to:")
    print(f"   {result['output_path']}")

    # Show what the dashboard UI would display
    print("\n" + "=" * 60)
    print("🖥️  Dashboard UI Preview:")
    print("=" * 60)
    print("\n[ ] Original datasets:")
    print(f"    [✓] {Path(dataset1).name} - {len(session1_examples)} examples")
    print(f"    [✓] {Path(dataset2).name} - {len(session2_examples)} examples")
    print(f"    [✓] {Path(dataset3).name} - {len(session3_examples)} examples")
    print(f"\n[ ] Merged dataset:")
    print(
        f"    [ ] {Path(result['output_path']).name} - {result['total_examples']} examples (merged)")

    print("\n💡 Usage in Dashboard:")
    print("   1. Go to Sleep tab")
    print("   2. Check the datasets you want to merge")
    print("   3. Click 'Merge Selected'")
    print("   4. New merged dataset will appear in the list")


if __name__ == "__main__":
    main()
