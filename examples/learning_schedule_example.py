"""
Learning Schedule Example

Demonstrates how to use the learning schedule system to study papers.

Usage:
    python examples/learning_schedule_example.py /path/to/papers
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.actions.learning_scheduler import LearningScheduleRunner


def main():
    if len(sys.argv) < 2:
        print("Usage: python learning_schedule_example.py <papers_directory>")
        print("\nExample:")
        print("  python learning_schedule_example.py ~/Documents/research_papers")
        sys.exit(1)
    
    papers_dir = sys.argv[1]
    
    if not os.path.exists(papers_dir):
        print(f"Error: Directory not found: {papers_dir}")
        sys.exit(1)
    
    # Create a runner for the student actor
    runner = LearningScheduleRunner('student')
    
    # List existing schedules
    print("\n📋 Existing Schedules:")
    schedules = runner.list_schedules()
    if schedules:
        for s in schedules:
            print(f"   - {s['id']}: {s['name']} ({s['progress']}) [{s['status']}]")
    else:
        print("   (none)")
    
    # Create a new schedule
    schedule_id = input("\nEnter schedule ID (or press Enter for 'papers_study'): ").strip()
    if not schedule_id:
        schedule_id = 'papers_study'
    
    schedule_name = input("Enter schedule name (or press Enter for default): ").strip()
    if not schedule_name:
        schedule_name = f"Study: {os.path.basename(papers_dir)}"
    
    focus_input = input("Enter focus areas (comma-separated, or press Enter for default): ").strip()
    if focus_input:
        focus_areas = [f.strip() for f in focus_input.split(',')]
    else:
        focus_areas = ['key concepts', 'methodology', 'findings', 'applications']
    
    print(f"\n🔧 Creating schedule '{schedule_id}'...")
    
    try:
        schedule = runner.create_schedule(
            schedule_id=schedule_id,
            papers_directory=papers_dir,
            name=schedule_name,
            focus_areas=focus_areas,
            depth='comprehensive'
        )
        
        print(f"\n✅ Schedule created!")
        print(f"   Papers found: {schedule['progress']['total_papers']}")
        print(f"   Focus areas: {', '.join(focus_areas)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Ask to run
    run_now = input("\nRun schedule now? (y/n): ").strip().lower()
    
    if run_now == 'y':
        max_papers_input = input("Max papers to process (or press Enter for all): ").strip()
        max_papers = int(max_papers_input) if max_papers_input else None
        
        print("\n🚀 Starting study session...")
        result = runner.run_schedule(schedule_id, max_papers=max_papers)
        
        # Export summary if there were successful results
        if result['successful'] > 0:
            export = input("\nExport knowledge summary to markdown? (y/n): ").strip().lower()
            if export == 'y':
                output_path = f"knowledge_summary_{schedule_id}.md"
                runner.export_knowledge_summary(schedule_id, output_path)
                print(f"✅ Summary exported to: {output_path}")
    
    print("\n✨ Done!")


if __name__ == '__main__':
    main()
