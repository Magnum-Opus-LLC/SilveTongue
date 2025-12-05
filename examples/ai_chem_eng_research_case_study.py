import sys
import os
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from thought_processor import ThoughtProcessor

def run_research_case_study():
    print("Initializing Student Actor for Research Case Study...")
    processor = ThoughtProcessor(actor_name='student')
    
    # Case Study Topic: AI in Chemical Engineering
    case_study_query = "Do a research on AI in chemical engineering impact."
    
    print(f"\nCase Study Query: {case_study_query}\n")
    
    result = processor.process(case_study_query)
    
    output_file = Path(__file__).parent / 'ai_chem_eng_case_study_output.md'
    
    with open(output_file, 'w') as f:
        f.write(f"# Case Study: {case_study_query}\n\n")
        f.write("## Subconscious Analysis\n")
        f.write(f"```json\n{result['analysis']}\n```\n\n")
        
        f.write("## Conscious Prompt (including Research)\n")
        f.write(f"```text\n{result['conscious_prompt']}\n```\n\n")
        
        f.write("## Final Plan / Answer\n")
        f.write(f"{result['plan']}\n")
        
    print(f"\nCase study output saved to: {output_file}")

if __name__ == "__main__":
    run_research_case_study()
