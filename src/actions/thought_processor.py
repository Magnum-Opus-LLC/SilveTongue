from src.thoughts.subconscious import SilvertongueSubconscious
from src.thoughts.conscious import SilvertongueConscious
from src.utils.knowledge_base import KnowledgeBase
from src.utils.research import researcher


class ThoughtProcessor:
    def __init__(self, actor_name='template'):
        self.kb = KnowledgeBase(actor_name)
        self.subconscious = SilvertongueSubconscious(self.kb)
        self.conscious = SilvertongueConscious(self.kb)

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
            decision = self.conscious.evaluate_progress(
                user_input, history, conscious_prompt)
            thought = decision.get("thought")
            action = decision.get("next_action")
            args = decision.get("args")

            print(f"Thought: {thought}")
            print(f"Proposed Action: {action}")

            # B. Subconscious Validation
            is_valid, feedback = self.subconscious.validate_thought(
                f"Action: {action}\nArgs: {args}", context=str(history))
            if not is_valid:
                print(f"Subconscious Veto: {feedback}")
                history.append(
                    {"step": "Subconscious Veto", "content": feedback})
                continue  # Skip execution, let conscious mind re-plan based on veto

            # C. Execute Action
            result = None
            if action == "RESEARCH":
                # Check if we've already done too much research
                research_count = sum(
                    1 for h in history if h['step'] == 'Research Results')
                if research_count >= 2:
                    print("Research limit reached. Forcing DRAFT.")
                    action = "DRAFT"
                else:
                    print(f"Executing Research: {args}")
                    result = researcher.conduct_research(args)
                    history.append(
                        {"step": "Research Results", "content": result})
                    continue  # Go to next iteration after research

            if action == "DRAFT":
                print("Drafting Response...")
                # Build context from research results
                research_content = "\n".join(
                    [h['content'] for h in history if h['step'] == 'Research Results'])
                draft_prompt = f"{conscious_prompt}\n\nResearch Findings:\n{research_content}\n\nBased on the above research and your knowledge, draft a comprehensive response to: {user_input}"
                result = self.conscious.perform_task_planning(
                    user_input, draft_prompt)
                history.append({"step": "Draft", "content": result})
                final_response = result

            elif action == "REVIEW":
                print("Reviewing Draft...")
                # Review the last draft in history
                last_draft = next((h['content'] for h in reversed(
                    history) if h['step'] == "Draft"), None)
                if last_draft:
                    review_prompt = f"{conscious_prompt}\n\nCritique this draft:\n{last_draft}"
                    result = self.conscious.perform_task_planning(
                        "Critique Draft", review_prompt)
                    history.append({"step": "Review", "content": result})
                else:
                    history.append(
                        {"step": "Error", "content": "No draft to review."})

            elif action == "FINISH":
                print("Finishing...")
                if not final_response:
                    # If we decided to finish but haven't drafted, the 'args' might be the answer
                    final_response = args
                break

            else:
                print(f"Unknown action: {action}")
                break

        if not final_response:
            print(
                "Max iterations reached or no final response generated. Drafting quick summary...")
            final_response = self.conscious.perform_task_planning(
                user_input, f"{conscious_prompt}\n\nHistory:\n{history}\n\nDraft a final response based on the available information.")

        # 4. Save Prompt (Snapshot of final state)
        saved_path = self.conscious.save_conscious_prompt(conscious_prompt)

        # 5. Log Interaction
        log_path = self.conscious.log_interaction(
            user_input, conscious_prompt, final_response, analysis)
        print(f"\nInteraction logged to: {log_path}")

        # 6. Record Learning Event
        insights = self._extract_insights(history, final_response)
        self.kb.record_learning_event(
            event_type="user_interaction",
            description=f"Processed query about: {analysis.get('domain', 'General')}",
            query=user_input,
            insights=insights,
            mindset_changes=[]
        )

        return {
            "analysis": analysis,
            "conscious_prompt": conscious_prompt,
            "plan": final_response,
            "log_path": log_path,
            "history": history
        }

    def _extract_insights(self, history, final_response):
        """Extract key insights from the interaction history."""
        insights = []

        # Extract from research results
        for h in history:
            if h['step'] == 'Research Results' and h['content']:
                # Take first 100 chars as a summary
                insights.append(f"Researched: {h['content'][:100]}...")

        # Note if drafting occurred
        if any(h['step'] == 'Draft' for h in history):
            insights.append("Generated comprehensive draft response")

        # Note if review occurred
        if any(h['step'] == 'Review' for h in history):
            insights.append("Self-reviewed and refined response")

        return insights[:5]  # Limit to 5 insights


if __name__ == "__main__":
    processor = ThoughtProcessor()
    # Example usage
    processor.process("How do I optimize a chemical reaction?")
