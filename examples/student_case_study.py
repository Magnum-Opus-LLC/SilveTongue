import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from thought_processor import ThoughtProcessor

def run_student_case_study():
    print("Initializing Student Actor...")
    processor = ThoughtProcessor(actor_name='student')
    
    # Case Study Topic: A non-engineering problem to test the engineering mindset transfer
    case_study_query = (
        "I am struggling to manage my time effectively between work, exercise, and social life. "
        "I feel overwhelmed and inefficient. How can I fix this?"
    )
    
    print(f"\nCase Study Query: {case_study_query}\n")
    
    result = processor.process(case_study_query)
    
    print("\n--- Final Plan / Guidance ---")
    print(result['plan'])

if __name__ == "__main__":
    run_student_case_study()
