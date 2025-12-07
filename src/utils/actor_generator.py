"""
Actor Profile Generator

This tool generates a populated actor profile from a high-level description.
All entries are designed to be:
1. Explicit and concrete (actionable, not vague)
2. Limited to max 10 entries per category (focus on essentials)
3. Executable (can be directly applied to reasoning or decision-making)

Usage:
    python actor_generator.py "A senior software engineer with 10 years of experience in distributed systems"
"""

import sys
import json
import yaml
from pathlib import Path

from src.utils.chat_utils import get_chat_completion


GENERATION_SPEC = """
You are generating a detailed actor profile for an AI agent. The profile must be:

1. EXPLICIT: Each entry must be concrete and specific, not abstract.
   - BAD: "Thermodynamics"
   - GOOD: "Energy cannot be created or destroyed in an isolated system; all reactions must conserve total energy (First Law of Thermodynamics)."

2. EXECUTABLE: Each entry must be directly usable in reasoning or decision-making.
   - BAD: "Good communication skills"
   - GOOD: "When explaining technical concepts, start with the core idea in one sentence, then provide supporting details, then give a concrete example."

3. LIMITED: Maximum 10 entries per category. Focus only on the most impactful items.

4. MEASURABLE where possible: Include quantifiable thresholds or criteria.
   - BAD: "Fast response time"
   - GOOD: "API response time should be under 200ms for 95th percentile; anything above 500ms indicates a problem requiring immediate investigation."

Generate the following categories based on the actor description provided:

---

## VALUES (max 5)
Core beliefs that guide all decisions. Each value should include a concrete behavior it implies.
Format: "Value Name: Belief statement. Behavior implication."

## PRINCIPLES (max 10)
Fundamental rules for understanding and interacting with the world. Must be falsifiable or verifiable.
Format: "Principle Name: Concrete rule or law with specific conditions and outcomes."

## CORE_KNOWLEDGE (max 10)
Immutable facts and laws the actor knows to be true. Must be specific and domain-grounded.
Format: "Topic: Specific fact or law with quantifiable details where applicable."

## SUPPORTING_KNOWLEDGE (max 10)
Contextual knowledge, case studies, and applied examples.
Format: "Context: Specific example or case with outcome and lesson learned."

## EXPERIENCES (max 10)
Past actions and heuristics that guide future decisions. Must be actionable patterns.
Format: "Situation: What was done, what happened, what to do next time."

## ADJUSTMENTS (max 5)
Domain-specific rules that override or refine general principles.
Format: "When [condition], then [specific action] because [reason]."

## CHALLENGES (max 5)
Known edge cases, anomalies, or unresolved questions that require caution.
Format: "Challenge: Specific problem or contradiction. Current handling: [approach]. Needs resolution: [yes/no]."

---

Return the output as valid YAML with the following structure:
```yaml
values:
  - "..."
principles:
  - "..."
core_knowledge:
  - "..."
supporting_knowledge:
  - "..."
experiences:
  - "..."
adjustments:
  - "..."
challenges:
  - "..."
```
"""


def generate_actor_profile(description: str) -> dict:
    """
    Generate a complete actor profile from a high-level description.
    
    Args:
        description: A description of the actor (e.g., "A chemical engineer with 5 years of refinery experience")
    
    Returns:
        dict: A dictionary containing all profile categories
    """
    messages = [
        {"role": "system", "content": GENERATION_SPEC},
        {"role": "user", "content": f"Generate a detailed actor profile for:\n\n{description}"}
    ]
    
    response = get_chat_completion(messages, temperature=0.3)
    
    # Extract YAML from response
    yaml_content = response
    if "```yaml" in response:
        yaml_content = response.split("```yaml")[1].split("```")[0]
    elif "```" in response:
        yaml_content = response.split("```")[1].split("```")[0]
    
    try:
        profile = yaml.safe_load(yaml_content)
        return profile
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return {"raw_response": response}


def save_actor_profile(profile: dict, actor_name: str, project_root: Path = None):
    """
    Save generated profile to the actor directory structure.
    
    Args:
        profile: Dictionary containing all profile categories
        actor_name: Name of the actor (will create directory under actors/)
        project_root: Root of the project (defaults to parent of src/)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    actor_root = project_root / 'actors' / actor_name
    
    # Define paths
    paths = {
        'values': actor_root / 'characteristic' / 'value' / 'values.yaml',
        'principles': actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
        'core_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
        'supporting_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
        'experiences': actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
        'adjustments': actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml',
        'challenges': actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
    }
    
    # Create directories and save files
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        
        content = profile.get(key, [])
        if not isinstance(content, list):
            content = [content] if content else []
        
        yaml_data = {key: content}
        
        with open(path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, width=120)
        
        print(f"Saved {key} to {path}")
    
    # Copy behavior.json from template
    template_behavior = project_root / 'actors' / 'template' / 'behavior.json'
    actor_behavior = actor_root / 'behavior.json'
    if template_behavior.exists() and not actor_behavior.exists():
        import shutil
        shutil.copy(template_behavior, actor_behavior)
        print(f"Copied behavior.json to {actor_behavior}")
    
    print(f"\nActor '{actor_name}' created successfully at {actor_root}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python actor_generator.py <actor_description>")
        print("       python actor_generator.py <actor_description> --save <actor_name>")
        print()
        print("Example:")
        print('  python actor_generator.py "A senior chemical engineer with 10 years of refinery experience"')
        print('  python actor_generator.py "A data scientist specializing in NLP" --save nlp_scientist')
        sys.exit(1)
    
    description = sys.argv[1]
    
    # Check for --save flag
    save_name = None
    if "--save" in sys.argv:
        save_idx = sys.argv.index("--save")
        if save_idx + 1 < len(sys.argv):
            save_name = sys.argv[save_idx + 1]
    
    print(f"Generating actor profile for: {description}\n")
    print("=" * 60)
    
    profile = generate_actor_profile(description)
    
    if "raw_response" in profile:
        print("Failed to parse structured profile. Raw response:")
        print(profile["raw_response"])
        sys.exit(1)
    
    # Print the generated profile
    print(yaml.dump(profile, default_flow_style=False, allow_unicode=True, width=120))
    
    # Save if requested
    if save_name:
        print("=" * 60)
        save_actor_profile(profile, save_name)


if __name__ == "__main__":
    main()
