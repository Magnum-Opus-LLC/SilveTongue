"""
Cognitive Loop Example

Demonstrates Lingkui's autonomous learning engine - the Cognitive Loop.

The loop orchestrates:
    SEARCH → STUDY → THINK → REFLECT → UPDATE → EXPAND → (repeat)

Modes:
    - Standard: Run N iterations and stop
    - Non-Stop: Run until goal achieved or Ctrl+C
    - Forever: Run indefinitely, cycling through topics

Usage:
    python examples/cognitive_loop_example.py
    python examples/cognitive_loop_example.py "quantum computing" "understand quantum algorithms"
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.actions.cognitive_loop import (
    CognitiveLoop, 
    learn, 
    learn_nonstop,
    run_forever,
    start_learning, 
    resume_learning
)


def demo_full_learning():
    """Demonstrate a complete learning journey."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         🧠 LINGKUI COGNITIVE LOOP - AUTONOMOUS LEARNING 🧠          ║
║                                                                      ║
║     ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐          ║
║     │ SEARCH  │ → │  STUDY  │ → │  THINK  │ → │ REFLECT │          ║
║     └─────────┘   └─────────┘   └─────────┘   └─────────┘          ║
║          ↑                                         │                 ║
║          │                                         ↓                 ║
║     ┌─────────┐                             ┌─────────┐             ║
║     │ EXPAND  │ ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ←│ UPDATE  │             ║
║     └─────────┘                             └─────────┘             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get topic and goal
    if len(sys.argv) >= 3:
        topic = sys.argv[1]
        goal = sys.argv[2]
    else:
        topic = "machine learning in chemical engineering"
        goal = "Understand how ML is being applied to optimize chemical processes"
    
    print(f"\n📚 Topic: {topic}")
    print(f"🎯 Goal: {goal}")
    print("\n" + "="*70)
    
    # Create and run the cognitive loop
    loop = CognitiveLoop('student', verbose=True)
    
    # Configure the loop
    loop.configure(
        max_papers_per_search=15,
        max_papers_to_study=3,
        search_depth=2,
        min_citations=10,  # Only papers with 10+ citations
        year_from=2018,    # Recent papers
        auto_save=True
    )
    
    # Run the learning journey
    result = loop.learn(
        topic=topic,
        goal=goal,
        max_iterations=2  # Limit iterations for demo
    )
    
    # Print summary report
    print("\n")
    print(loop.get_summary_report())
    
    return result


def demo_step_by_step():
    """Demonstrate step-by-step control of the loop."""
    print("\n" + "="*70)
    print("📍 STEP-BY-STEP MODE")
    print("="*70)
    
    loop = CognitiveLoop('student', verbose=True)
    
    loop.configure(
        max_papers_per_search=10,
        max_papers_to_study=2,
        search_depth=1
    )
    
    # Start the loop
    loop.start(
        topic="transformer neural networks",
        goal="Understand the attention mechanism"
    )
    
    # Execute each phase manually
    print("\n[Press Enter to execute each phase...]")
    
    phases = ['SEARCH', 'STUDY', 'THINK', 'REFLECT', 'UPDATE', 'EXPAND']
    
    for expected_phase in phases:
        input(f"\n→ Execute {expected_phase} phase? ")
        state = loop.step()
        
        if state.phase.value == 'complete':
            break
    
    return loop.get_state()


def demo_resume():
    """Demonstrate resuming a saved loop."""
    print("\n" + "="*70)
    print("📂 SAVED LOOPS")
    print("="*70)
    
    loop = CognitiveLoop('student')
    saved_loops = loop.list_loops()
    
    if not saved_loops:
        print("No saved loops found.")
        return
    
    print("\nAvailable loops:\n")
    for i, l in enumerate(saved_loops[:5]):
        print(f"  {i+1}. [{l['loop_id']}] {l['topic'][:40]}...")
        print(f"      Phase: {l['phase']}, Iteration: {l['iteration']}")
        print(f"      Papers studied: {l['papers_studied']}")
        print()
    
    # Option to resume
    print("To resume a loop, use:")
    print("  loop = resume_learning('student', '<loop_id>')")
    print("  loop.step()  # Continue from where you left off")


def demo_quick_learn():
    """Quick one-liner learning."""
    print("\n" + "="*70)
    print("⚡ QUICK LEARN MODE")
    print("="*70)
    
    # One-liner to learn about a topic
    result = learn(
        actor_name='student',
        topic='graph neural networks',
        goal='Understand how GNNs work for molecular property prediction',
        max_iterations=1,
        max_papers_to_study=2,
        search_depth=1,
        min_citations=50
    )
    
    print("\n📊 Quick Learn Result:")
    print(f"   Papers studied: {result['statistics']['total_papers_studied']}")
    print(f"   Knowledge bits: {result['statistics']['total_knowledge_bits']}")
    print(f"   Conclusions: {len(result['conclusions'])}")
    
    return result


def demo_nonstop_mode():
    """Demonstrate non-stop learning mode."""
    print("\n" + "="*70)
    print("🔄 NON-STOP MODE")
    print("="*70)
    print("""
    Non-stop mode runs until:
    - Goal progress reaches threshold (default: 90%)
    - Max iterations reached (safety limit)
    - You press Ctrl+C (graceful save & exit)
    """)
    
    # Get topic
    if len(sys.argv) >= 3:
        topic = sys.argv[1]
        goal = sys.argv[2]
    else:
        topic = input("Topic [deep learning optimization]: ").strip() or "deep learning optimization"
        goal = input("Goal [understand key optimization techniques]: ").strip() or "understand key optimization techniques"
    
    print(f"\n📚 Topic: {topic}")
    print(f"🎯 Goal: {goal}")
    print("\n⚠️  Press Ctrl+C at any time to gracefully stop and save\n")
    
    loop = CognitiveLoop('student', verbose=True)
    loop.configure(
        max_papers_per_search=10,
        max_papers_to_study=3,
        search_depth=2,
        min_citations=20
    )
    
    # Custom callback for iteration completion
    def on_iteration(state, iteration):
        print(f"\n📈 Progress callback: Iteration {iteration} done")
        print(f"   Total papers: {state.total_papers_studied}")
        print(f"   Emerging questions: {len(state.emerging_questions)}")
    
    result = loop.learn_nonstop(
        topic=topic,
        goal=goal,
        goal_threshold=85,
        max_iterations=10,
        pause_between_iterations=2,  # 2 second pause
        checkpoint_interval=1,
        on_iteration_complete=on_iteration
    )
    
    print("\n")
    print(loop.get_summary_report())
    
    return result


def demo_forever_mode():
    """Demonstrate forever learning mode."""
    print("\n" + "="*70)
    print("🌀 FOREVER MODE - Autonomous Learning Agent")
    print("="*70)
    print("""
    Forever mode runs indefinitely:
    - Cycles through initial topics
    - Generates new topics from emerging questions
    - Never stops until Ctrl+C
    
    ⚠️  This will run FOREVER. Press Ctrl+C to stop.
    """)
    
    confirm = input("Start forever mode? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Initial topics to explore
    topics = [
        "transformer attention mechanisms",
        "graph neural networks for chemistry",
        "reinforcement learning in process control"
    ]
    
    print(f"\n📚 Initial topics: {len(topics)}")
    for t in topics:
        print(f"   • {t}")
    
    print("\n🚀 Starting forever mode...\n")
    
    loop = CognitiveLoop('student', verbose=True)
    loop.configure(
        max_papers_per_search=8,
        max_papers_to_study=2,
        search_depth=1,
        min_citations=30
    )
    
    # Custom goal generator
    def generate_goal(topic):
        return f"Develop comprehensive understanding of {topic} and identify practical applications"
    
    loop.run_forever(
        topics=topics,
        goal_generator=generate_goal,
        idle_search_interval=60,  # 1 minute idle between topic cycles
        goal_threshold=80,
        max_iterations=5  # Max iterations per topic
    )


def main():
    print("""
╭──────────────────────────────────────────────────────────────────────╮
│                                                                      │
│            🧠 LINGKUI COGNITIVE LOOP DEMONSTRATION 🧠                │
│                                                                      │
│   The Cognitive Loop is Lingkui's autonomous learning engine.        │
│   It orchestrates a continuous cycle of:                             │
│                                                                      │
│     1. SEARCH  - Discover relevant papers and resources              │
│     2. STUDY   - Deep dive into materials, extract knowledge         │
│     3. THINK   - Synthesize and reason about information             │
│     4. REFLECT - Evaluate against existing beliefs                   │
│     5. UPDATE  - Integrate new knowledge into mindset                │
│     6. EXPAND  - Identify gaps, generate new questions               │
│                                                                      │
╰──────────────────────────────────────────────────────────────────────╯
    """)
    
    print("Select demo mode:")
    print("  1. Full Learning Journey (limited iterations)")
    print("  2. Step-by-Step Control")
    print("  3. Quick Learn (single iteration)")
    print("  4. 🔄 Non-Stop Mode (until goal achieved)")
    print("  5. 🌀 Forever Mode (runs indefinitely)")
    print("  6. View Saved Loops")
    print("  7. Exit")
    
    try:
        choice = input("\nChoice [1-7]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"
    
    if choice == "1":
        demo_full_learning()
    elif choice == "2":
        demo_step_by_step()
    elif choice == "3":
        demo_quick_learn()
    elif choice == "4":
        demo_nonstop_mode()
    elif choice == "5":
        demo_forever_mode()
    elif choice == "6":
        demo_resume()
    else:
        print("Goodbye!")


if __name__ == '__main__':
    main()
