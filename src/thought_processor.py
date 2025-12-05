from subconscious import LingkuiSubconscious
from conscious import LingkuiConscious
from knowledge_base import KnowledgeBase
from research import researcher

class ThoughtProcessor:
    def __init__(self, actor_name='template'):
        self.kb = KnowledgeBase(actor_name)
        self.subconscious = LingkuiSubconscious(self.kb)
        self.conscious = LingkuiConscious(self.kb)

    def process(self, user_input):
        # 1. Subconscious Analysis
        print("--- Subconscious Analysis ---")
        analysis = self.subconscious.analyze_request(user_input)
        print(f"Analysis Result: {analysis}")
        
        # 2. Conscious Prompt Construction
        print("\n--- Conscious Prompt Construction ---")
        conscious_prompt = self.conscious.construct_conscious_prompt(analysis)
        
        # Initialize Loop State
        history = []
        final_response = None
        max_iterations = 5
        iteration = 0
        
        print("\n--- Entering Thought Loop ---")
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n[Iteration {iteration}]")
            
            # A. Evaluate Progress / Decide Next Step
            decision = self.conscious.evaluate_progress(user_input, history, conscious_prompt)
            thought = decision.get("thought")
            action = decision.get("next_action")
            args = decision.get("args")
            
            print(f"Thought: {thought}")
            print(f"Proposed Action: {action}")
            
            # B. Subconscious Validation
            is_valid, feedback = self.subconscious.validate_thought(f"Action: {action}\nArgs: {args}", context=str(history))
            if not is_valid:
                print(f"Subconscious Veto: {feedback}")
                history.append({"step": "Subconscious Veto", "content": feedback})
                continue # Skip execution, let conscious mind re-plan based on veto
            
            # C. Execute Action
            result = None
            if action == "RESEARCH":
                print(f"Executing Research: {args}")
                result = researcher.conduct_research(args)
                history.append({"step": "Research Results", "content": result})
                
            elif action == "DRAFT":
                print("Drafting Response...")
                # We can use perform_task_planning or a new method. Let's reuse perform_task_planning for now but context aware
                # Actually, let's just ask the conscious mind to draft based on history
                draft_prompt = f"{conscious_prompt}\n\nHistory:\n{history}\n\nDraft a response to: {user_input}"
                # Simple call for now, ideally we'd have a specific draft method
                result = self.conscious.perform_task_planning(user_input, draft_prompt)
                history.append({"step": "Draft", "content": result})
                final_response = result # Tentative final response
                
            elif action == "REVIEW":
                print("Reviewing Draft...")
                # Review the last draft in history
                last_draft = next((h['content'] for h in reversed(history) if h['step'] == "Draft"), None)
                if last_draft:
                    review_prompt = f"{conscious_prompt}\n\nCritique this draft:\n{last_draft}"
                    result = self.conscious.perform_task_planning("Critique Draft", review_prompt)
                    history.append({"step": "Review", "content": result})
                else:
                    history.append({"step": "Error", "content": "No draft to review."})
                    
            elif action == "FINISH":
                print("Finishing...")
                if not final_response:
                     # If we decided to finish but haven't drafted, the 'args' might be the answer
                     final_response = args
                break
            
            else:
                print(f"Unknown action: {action}")
                break
                
        # 4. Save Prompt (Snapshot of final state)
        saved_path = self.conscious.save_conscious_prompt(conscious_prompt)
        
        # 5. Log Interaction
        log_path = self.conscious.log_interaction(user_input, conscious_prompt, final_response, analysis)
        print(f"\nInteraction logged to: {log_path}")
        
        return {
            "analysis": analysis,
            "conscious_prompt": conscious_prompt,
            "plan": final_response,
            "log_path": log_path,
            "history": history
        }

if __name__ == "__main__":
    processor = ThoughtProcessor()
    # Example usage
    processor.process("How do I optimize a chemical reaction?")
