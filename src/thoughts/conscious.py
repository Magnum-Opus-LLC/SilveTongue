import json
import uuid
from datetime import datetime
from pathlib import Path
from src.utils.knowledge_base import kb
from src.utils.chat_utils import get_chat_completion

class SilvertongueConscious:
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
        
        prompt = f"You are the Conscious Mind of Silvertongue, operating in the domain of {domain}.\n\n"
        
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

    def evaluate_progress(self, user_input, history, conscious_prompt):
        """
        Decides the next step in the thought process loop.
        Returns a JSON object with 'thought', 'next_action', and 'args'.
        """
        # Count how many research steps have been done
        research_count = sum(1 for h in history if h['step'] == 'Research Results')
        has_draft = any(h['step'] == 'Draft' for h in history)
        
        planning_prompt = (
            f"{conscious_prompt}\n\n"
            "You are in a self-reflective thought loop. Review the current history of thoughts and actions.\n"
            "Decide what to do next to best answer the user's request.\n\n"
            f"STATUS: Research actions taken: {research_count}. Draft exists: {has_draft}.\n\n"
            "STRICT RULES:\n"
            "1. MAXIMUM 2 RESEARCH ACTIONS ALLOWED. If you have already done 2 research steps, you MUST move to DRAFT or FINISH.\n"
            "2. If 'Research Results' exists in history with content, you have enough info to DRAFT. Do NOT research again.\n"
            "3. After DRAFT, you can only REVIEW or FINISH. Never RESEARCH after drafting.\n"
            "4. If a draft exists and is acceptable, choose FINISH.\n"
            "5. Be decisive. If unsure, DRAFT with available information.\n\n"
            "Possible Actions:\n"
            "- RESEARCH: ONLY if research_count < 2 AND no research results exist yet.\n"
            "- DRAFT: Write a comprehensive response using available information.\n"
            "- REVIEW: Critique an existing draft (only if draft exists).\n"
            "- FINISH: Complete the task (use if draft exists and is good).\n\n"
            "Output ONLY valid JSON (no markdown, no explanation outside JSON):\n"
            "{\n"
            '  "thought": "Your internal reasoning...",\n'
            '  "next_action": "RESEARCH" | "DRAFT" | "REVIEW" | "FINISH",\n'
            '  "args": "Search query, or empty string for DRAFT/REVIEW/FINISH"\n'
            "}\n"
        )
        
        history_str = "\n".join([f"{h['step']}: {h['content'][:500]}..." if len(str(h['content'])) > 500 else f"{h['step']}: {h['content']}" for h in history])
        
        messages = [
            {"role": "system", "content": planning_prompt},
            {"role": "user", "content": f"User Request: {user_input}\n\nHistory:\n{history_str}"}
        ]
        
        response_text = get_chat_completion(messages, temperature=0.2).strip()
        
        try:
            # Clean up potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            response_text = response_text.strip()
            return json.loads(response_text)
        except (json.JSONDecodeError, IndexError):
            # Try regex extraction as fallback
            import re
            thought_match = re.search(r'"thought"\s*:\s*"([^"]+)"', response_text)
            action_match = re.search(r'"next_action"\s*:\s*"(RESEARCH|DRAFT|REVIEW|FINISH)"', response_text)
            args_match = re.search(r'"args"\s*:\s*"([^"]+)"', response_text)
            
            if action_match:
                return {
                    "thought": thought_match.group(1) if thought_match else "Extracted from malformed JSON",
                    "next_action": action_match.group(1),
                    "args": args_match.group(1) if args_match else ""
                }
            
            # Final fallback: if we have research results, draft; otherwise finish
            if research_count > 0 or has_draft:
                return {"thought": "JSON parsing failed, proceeding to draft", "next_action": "DRAFT", "args": ""}
            return {"thought": "JSON parsing failed", "next_action": "FINISH", "args": response_text}

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
