import json
from pathlib import Path
from knowledge_base import kb
from chat_utils import get_chat_completion

class LingkuiConscious:
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base if knowledge_base else kb

    def construct_conscious_prompt(self, analysis_result):
        """
        Constructs the conscious system prompt based on subconscious analysis.
        """
        domain = analysis_result.get("domain", "General")
        experiences = analysis_result.get("relevant_experiences", [])
        knowledge = analysis_result.get("relevant_knowledge", [])
        
        principles = self.kb.get_principles()
        
        prompt = f"You are the Conscious Mind of Lingkui, operating in the domain of {domain}.\n\n"
        
        prompt += "Principles:\n" + "\n".join(f"- {p}" for p in principles) + "\n\n"
        
        if experiences:
            prompt += "Relevant Experiences:\n" + "\n".join(f"- {e}" for e in experiences) + "\n\n"
            
        if knowledge:
            prompt += "Relevant Contextual Knowledge:\n" + "\n".join(f"- {k}" for k in knowledge) + "\n\n"
            
        prompt += (
            "Your task is to:\n"
            "1. Classify and abstract the user's request.\n"
            "2. Provide a summary of the question.\n"
            "3. Formulate a plan or provide general guidance to address the request.\n"
        )
        
        return prompt

    def perform_task_planning(self, user_input, conscious_prompt):
        """
        Uses the conscious prompt to perform task planning or abstraction.
        """
        messages = [
            {"role": "system", "content": conscious_prompt},
            {"role": "user", "content": f"Request: {user_input}"}
        ]
        
        return get_chat_completion(messages, temperature=0.5)

    def save_conscious_prompt(self, prompt, filename="current_conscious_prompt.txt"):
        """
        Saves the conscious system prompt to a file.
        """
        # Save to data/conscious_prompts/ (create if not exists)
        save_dir = self.kb.project_root / 'data' / 'conscious_prompts'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / filename
        with open(file_path, 'w') as f:
            f.write(prompt)
            
        return str(file_path)
