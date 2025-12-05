import json
import uuid
from datetime import datetime
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

    def identify_research_needs(self, user_input, conscious_prompt):
        """
        Determines if external research is needed and returns a search query if so.
        Returns a tuple: (reasoning, search_query)
        """
        research_check_prompt = (
            f"{conscious_prompt}\n\n"
            "Task: Evaluate if external research is required for this request.\n"
            "1. Reflect on the user's request based on your principles, expertise, and current context.\n"
            "2. Determine if you have sufficient internal knowledge to provide a high-quality, accurate response.\n"
            "3. If you need external info (e.g., current events, specific data, recent papers), formulate a specific search query.\n"
            "4. If no research is needed, explain why your internal knowledge is sufficient.\n\n"
            "Output Format (JSON):\n"
            "{\n"
            '  "reasoning": "Your internal thought process and judgment...",\n'
            '  "search_query": "The query string" or null\n'
            "}\n"
            "Ensure the output is valid JSON."
        )
        
        messages = [
            {"role": "system", "content": research_check_prompt},
            {"role": "user", "content": f"Request: {user_input}"}
        ]
        
        response_text = get_chat_completion(messages, temperature=0.2).strip()
        
        # Attempt to parse JSON
        try:
            # Clean up potential markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "")
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "")
                
            data = json.loads(response_text)
            reasoning = data.get("reasoning", "No reasoning provided.")
            search_query = data.get("search_query")
            return reasoning, search_query
        except json.JSONDecodeError:
            # Fallback for non-JSON response
            if "NO_RESEARCH" in response_text.upper():
                return response_text, None
            return response_text, response_text  # Assume the whole text is the query/reasoning mixed

    def perform_task_planning(self, user_input, conscious_prompt):
        """
        Uses the conscious prompt to perform task planning or abstraction.
        """
        messages = [
            {"role": "system", "content": conscious_prompt},
            {"role": "user", "content": f"Request: {user_input}"}
        ]
        
        return get_chat_completion(messages, temperature=0.5)

    def log_interaction(self, user_input, conscious_prompt, response, analysis):
        """
        Logs the entire interaction including prompt, query, response, and analysis.
        """
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Directory: data/{actor_name}/{timestamp}_{uuid}/
        actor_name = getattr(self.kb, 'actor_name', 'unknown_actor')
        save_dir = self.kb.project_root / 'data' / actor_name / f"{timestamp}_{interaction_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Conscious Prompt
        with open(save_dir / "conscious_system_prompt.txt", 'w') as f:
            f.write(conscious_prompt)
            
        # Save User Query
        with open(save_dir / "user_query.txt", 'w') as f:
            f.write(user_input)
            
        # Save Response/Plan
        with open(save_dir / "response.txt", 'w') as f:
            f.write(str(response))
            
        # Save Subconscious Analysis
        with open(save_dir / "subconscious_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return str(save_dir)

    def save_conscious_prompt(self, prompt, filename="current_conscious_prompt.txt"):
        """
        Saves the conscious system prompt to a file.
        (Kept for backward compatibility or quick debugging)
        """
        # Save to data/conscious_prompts/ (create if not exists)
        save_dir = self.kb.project_root / 'data' / 'conscious_prompts'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / filename
        with open(file_path, 'w') as f:
            f.write(prompt)
            
        return str(file_path)
