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
        
        # 3. Research Check
        print("\n--- Research Check ---")
        search_query = self.conscious.identify_research_needs(user_input, conscious_prompt)
        if search_query:
            print(f"Research needed: {search_query}")
            research_results = researcher.conduct_research(search_query)
            print("Research completed.")
            
            # Append research findings to conscious prompt
            conscious_prompt += f"\n\nResearch Findings for '{search_query}':\n{research_results}\n"
        else:
            print("No external research needed.")
        
        # 4. Save Prompt
        saved_path = self.conscious.save_conscious_prompt(conscious_prompt)
        print(f"Conscious prompt saved to: {saved_path}")
        
        # 5. Task Planning / Abstraction
        print("\n--- Task Planning / Abstraction ---")
        plan = self.conscious.perform_task_planning(user_input, conscious_prompt)
        print(f"Plan: {plan}")
        
        # 6. Log Interaction
        log_path = self.conscious.log_interaction(user_input, conscious_prompt, plan, analysis)
        print(f"\nInteraction logged to: {log_path}")
        
        return {
            "analysis": analysis,
            "conscious_prompt": conscious_prompt,
            "plan": plan,
            "log_path": log_path
        }

if __name__ == "__main__":
    processor = ThoughtProcessor()
    # Example usage
    processor.process("How do I optimize a chemical reaction?")
