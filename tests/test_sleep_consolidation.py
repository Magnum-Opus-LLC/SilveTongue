#!/usr/bin/env python3
"""
Quick test of the sleep consolidation feature
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.actions.sleep_processor import SleepProcessor

def test_sleep_consolidation():
    """Test sleep with consolidation on an actor."""
    
    print("=" * 70)
    print("Testing Sleep with Consolidation")
    print("=" * 70)
    
    # Use steve or student actor
    actor_name = 'steve'  # Change to 'student' or another actor if needed
    
    processor = SleepProcessor(actor_name, verbose=True)
    
    print(f"\n🌙 Putting {actor_name} to sleep...")
    print("-" * 70)
    
    # Run sleep cycle (will consolidate and archive)
    result = processor.sleep()
    
    print("\n" + "=" * 70)
    print("Sleep Summary:")
    print(f"  Total Examples: {result.get('total_examples', 0)}")
    print(f"  Output: {result.get('output_path', 'N/A')}")
    print("=" * 70)
    
    return result

if __name__ == '__main__':
    test_sleep_consolidation()
