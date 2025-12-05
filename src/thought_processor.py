from subconscious import LingkuiSubconscious
from conscious import LingkuiConscious
from knowledge_base import KnowledgeBase

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
        
        # 3. Save Prompt
        saved_path = self.conscious.save_conscious_prompt(conscious_prompt)
        print(f"Conscious prompt saved to: {saved_path}")
        
        # 4. Task Planning / Abstraction
        print("\n--- Task Planning / Abstraction ---")
        plan = self.conscious.perform_task_planning(user_input, conscious_prompt)
        print(f"Plan: {plan}")
        
        return {
            "analysis": analysis,
            "conscious_prompt": conscious_prompt,
            "plan": plan
        }

if __name__ == "__main__":
    processor = ThoughtProcessor()
    # Example usage
    processor.process("How do I optimize a chemical reaction?")
