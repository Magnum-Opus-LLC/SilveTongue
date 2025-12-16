#!/usr/bin/env python
"""
Cognitive Evolution - Autonomous Learning Agent

This is the main entry point for running the Silvertongue cognitive learning system.
The agent autonomously:
1. Searches for academic papers on a topic
2. Fetches and reads full PDFs when available
3. Extracts knowledge and insights
4. Reflects on how new information relates to existing knowledge
5. Updates its knowledge base
6. Generates questions for further exploration
7. Saves all learning data for LoRA fine-tuning

Usage:
    python cognitive_evolution.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.actions.cognitive_loop import CognitiveLoop, learn, learn_nonstop, run_forever
from src.utils.silvertongue_chat import interactive_chat as chat_with_silvertongue

# Global variable to store selected actor
SELECTED_ACTOR = None


def get_available_actors():
    """Get list of available actors from array/actors directory."""
    from src.utils.chat_utils import load_config
    
    config = load_config()
    array_dir = Path(config.get('array_dir', 'array'))
    actors_dir = array_dir / 'actors'
    
    if not actors_dir.exists():
        return []
    
    actors = []
    for item in actors_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has the required files
            if (item / 'behavior.json').exists() or (item / 'characteristic').exists():
                actors.append(item.name)
    
    return sorted(actors)


def select_actor():
    """Prompt user to select an actor."""
    global SELECTED_ACTOR
    
    actors = get_available_actors()
    
    if not actors:
        print("\n⚠️  No actors found in array/actors/")
        print("   Using default: student")
        SELECTED_ACTOR = 'student'
        return SELECTED_ACTOR
    
    print("\n" + "─" * 40)
    print("🎭 SELECT ACTOR")
    print("─" * 40)
    
    for i, actor in enumerate(actors, 1):
        marker = "  "
        if SELECTED_ACTOR == actor:
            marker = "→ "
        print(f"  {marker}{i}. {actor}")
    
    print()
    
    try:
        choice = input(f"Select actor [1-{len(actors)}] (or Enter for '{actors[0]}'): ").strip()
        
        if not choice:
            SELECTED_ACTOR = actors[0]
        elif choice.isdigit() and 1 <= int(choice) <= len(actors):
            SELECTED_ACTOR = actors[int(choice) - 1]
        elif choice in actors:
            SELECTED_ACTOR = choice
        else:
            print(f"Invalid choice, using: {actors[0]}")
            SELECTED_ACTOR = actors[0]
            
    except (EOFError, KeyboardInterrupt):
        print(f"\nUsing: {actors[0]}")
        SELECTED_ACTOR = actors[0]
    
    print(f"\n✅ Selected actor: {SELECTED_ACTOR}")
    return SELECTED_ACTOR


def show_banner():
    """Display the application banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║      🧠 SILVERTONGUE COGNITIVE EVOLUTION - Learning Agent 🧠        ║
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
║   Now with full PDF fetching - learns from complete papers! 📄       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def get_topic_and_goal():
    """Prompt user for topic and goal."""
    print("\n" + "-" * 60)
    topic = input("📚 Topic to learn about: ").strip()
    if not topic:
        topic = "machine learning in chemical engineering"
        print(f"   Using default: {topic}")
    
    goal = input("🎯 Learning goal (or press Enter for default): ").strip()
    if not goal:
        goal = f"Develop comprehensive understanding of {topic}"
        print(f"   Using: {goal}")
    
    return topic, goal


def demo_full_learning():
    """Run a complete learning journey with limited iterations."""
    topic, goal = get_topic_and_goal()
    
    print(f"\n📚 Topic: {topic}")
    print(f"🎯 Goal: {goal}")
    print("\n" + "=" * 70)
    
    loop = CognitiveLoop(SELECTED_ACTOR, verbose=True)
    loop.configure(
        fetch_pdfs=True,
        save_session=True,
        max_papers_per_search=15,
        max_papers_to_study=3,
        search_depth=2,
        min_citations=10,
        year_from=2018,
        auto_save=True
    )
    
    result = loop.learn(
        topic=topic,
        goal=goal,
        max_iterations=2
    )
    
    print("\n")
    print(loop.get_summary_report())
    return result


def demo_step_by_step():
    """Step-by-step control of the learning loop."""
    topic, goal = get_topic_and_goal()
    
    print("\n" + "=" * 70)
    print("📍 STEP-BY-STEP MODE")
    print("=" * 70)
    print("\nYou control when each phase runs.")
    print("Press Enter to advance, 'q' to quit, 's' to skip phase.\n")
    
    loop = CognitiveLoop(SELECTED_ACTOR, verbose=True)
    loop.configure(
        fetch_pdfs=True,
        save_session=True,
        max_papers_per_search=10,
        max_papers_to_study=3
    )
    
    loop.start(topic, goal)
    
    phase_count = 0
    max_phases = 12  # 2 full iterations
    
    while loop.state.phase.name != 'COMPLETE' and phase_count < max_phases:
        phase_count += 1
        current_phase = loop.state.phase.name
        
        cmd = input(f"\n[{phase_count}] Next: {current_phase} - Press Enter to run (q=quit, s=skip): ").strip().lower()
        
        if cmd == 'q':
            print("Stopping...")
            break
        elif cmd == 's':
            print(f"Skipping {current_phase}...")
            loop._advance_phase()
            continue
        
        loop.step()
        
        if loop.state.phase.name == 'SEARCH' and loop.state.iteration > 1:
            print(f"\n{'─' * 60}")
            print(f"✅ Iteration {loop.state.iteration - 1} complete")
            print(f"   Papers: {loop.state.total_papers_studied}")
            print(f"   Knowledge bits: {loop.state.total_knowledge_bits}")
            print(f"{'─' * 60}")
    
    print("\n")
    print(loop.get_summary_report())


def demo_quick_learn():
    """Quick single-iteration learning."""
    topic, goal = get_topic_and_goal()
    
    print(f"\n⚡ Quick learning: {topic}")
    print("=" * 70)
    
    result = learn(
        actor_name=SELECTED_ACTOR,
        topic=topic,
        goal=goal,
        max_iterations=1,
        fetch_pdfs=True,
        save_session=True
    )
    
    if result:
        print(f"\n✅ Quick learn complete!")
        print(f"   Papers studied: {result.get('total_papers_studied', 0)}")
        print(f"   Knowledge bits: {result.get('total_knowledge_bits', 0)}")
    
    return result


def demo_nonstop_mode():
    """Non-stop learning until goal achieved."""
    topic, goal = get_topic_and_goal()
    
    print("\n" + "=" * 70)
    print("🔄 NON-STOP MODE")
    print("=" * 70)
    print(f"\n📚 Topic: {topic}")
    print(f"🎯 Goal: {goal}")
    print("\n⚠️  Press Ctrl+C at any time to gracefully stop and save\n")
    
    loop = CognitiveLoop(SELECTED_ACTOR, verbose=True)
    loop.configure(
        fetch_pdfs=True,
        save_session=True,
        max_papers_per_search=10,
        max_papers_to_study=3,
        search_depth=2,
        min_citations=20
    )
    
    def on_iteration(state, iteration):
        print(f"\n📈 Progress: Iteration {iteration} done")
        print(f"   Total papers: {state.total_papers_studied}")
        print(f"   Emerging questions: {len(state.emerging_questions)}")
    
    result = loop.learn_nonstop(
        topic=topic,
        goal=goal,
        goal_threshold=85,
        max_iterations=10,
        pause_between_iterations=2,
        checkpoint_interval=1,
        on_iteration_complete=on_iteration
    )
    
    print("\n")
    print(loop.get_summary_report())
    return result


def demo_forever_mode():
    """Forever learning mode - autonomous multi-topic exploration."""
    print("\n" + "=" * 70)
    print("🌀 FOREVER MODE - Autonomous Learning Agent")
    print("=" * 70)
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
    
    topics_input = input("\nEnter topics (comma-separated, or Enter for defaults): ").strip()
    
    if topics_input:
        topics = [t.strip() for t in topics_input.split(',') if t.strip()]
    else:
        topics = [
            "transformer attention mechanisms",
            "state space models for sequences",
            "efficient deep learning architectures"
        ]
    
    print(f"\n📚 Initial topics: {len(topics)}")
    for t in topics:
        print(f"   • {t}")
    
    print("\n🚀 Starting forever mode...\n")
    
    loop = CognitiveLoop(SELECTED_ACTOR, verbose=True)
    loop.configure(
        fetch_pdfs=True,
        save_session=True,
        max_papers_per_search=8,
        max_papers_to_study=2,
        search_depth=1,
        min_citations=30
    )
    
    def generate_goal(topic):
        return f"Develop comprehensive understanding of {topic} and identify practical applications"
    
    loop.run_forever(
        topics=topics,
        goal_generator=generate_goal,
        idle_search_interval=60,
        goal_threshold=80,
        max_iterations=5
    )


def demo_no_pdf_mode():
    """Fast learning using abstracts only (no PDF download)."""
    topic, goal = get_topic_and_goal()
    
    print(f"\n⚡ Fast learning (abstracts only): {topic}")
    print("=" * 70)
    
    loop = CognitiveLoop(SELECTED_ACTOR, verbose=True)
    loop.configure(
        fetch_pdfs=False,  # Disable PDF fetching
        save_session=True,
        max_papers_per_search=15,
        max_papers_to_study=5,  # Can study more since no PDF download
        min_citations=10
    )
    
    result = loop.learn(
        topic=topic,
        goal=goal,
        max_iterations=2
    )
    
    print("\n")
    print(loop.get_summary_report())
    return result


def view_learning_sessions():
    """View saved learning sessions."""
    from src.utils.learning_session import LearningSessionManager
    
    print("\n" + "=" * 70)
    print("📚 SAVED LEARNING SESSIONS")
    print("=" * 70)
    
    manager = LearningSessionManager(SELECTED_ACTOR)
    sessions = manager.list_sessions()
    
    if not sessions:
        print("\nNo saved sessions found.")
        return
    
    print(f"\nFound {len(sessions)} sessions:\n")
    for i, session in enumerate(sessions, 1):
        print(f"  {i}. {session['topic'][:50]}")
        print(f"     ID: {session['session_id']}")
        print(f"     Created: {session['created_at'][:19]}")
        print(f"     Papers: {session['papers_studied']}, Q&A pairs: {session['qa_pairs']}")
        print()
    
    # Option to export training data
    export = input("Export all training data to JSONL? [y/N]: ").strip().lower()
    if export == 'y':
        output = manager.export_all_training_data()
        print(f"\n✅ Exported to: {output}")


def main():
    show_banner()
    
    # First, select which actor to use
    select_actor()
    
    print("\n" + "─" * 40)
    print(f"🎭 Active Actor: {SELECTED_ACTOR}")
    print("─" * 40)
    
    print("\nSelect mode:")
    print("  1. 📖 Full Learning Journey (2 iterations)")
    print("  2. 🎮 Step-by-Step Control")
    print("  3. ⚡ Quick Learn (1 iteration)")
    print("  4. 🔄 Non-Stop Mode (until goal achieved)")
    print("  5. 🌀 Forever Mode (runs indefinitely)")
    print("  6. 🚀 Fast Mode (abstracts only, no PDF)")
    print("  7. 💬 Chat with Silvertongue")
    print("  8. 📚 View Saved Sessions"))
    print("  9. 🔄 Change Actor")
    print("  0. 🚪 Exit")
    
    try:
        choice = input("\nChoice [0-9]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        return
    
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
        demo_no_pdf_mode()
    elif choice == "7":
        chat_with_silvertongue(SELECTED_ACTOR)
    elif choice == "8":
        view_learning_sessions()
    elif choice == "9":
        select_actor()
        main()  # Restart menu with new actor
    else:
        print("Goodbye!")


if __name__ == "__main__":
    main()
