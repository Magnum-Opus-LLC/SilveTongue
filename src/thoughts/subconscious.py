import json
from src.utils.knowledge_base import kb
from src.utils.chat_utils import get_chat_completion


class SilvertongueSubconscious:
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base if knowledge_base else kb

    def _get_character_prompt(self):
        """
        Constructs the character section of the system prompt based on values.
        """
        values = self.kb.get_values()
        if not values:
            return "You are a helpful assistant."

        values_str = "\n".join([f"- {v}" for v in values])
        return f"You are an agent defined by the following core values:\n{values_str}\n\nAct consistently with these values in all interactions."

    def generate_subconscious_prompt(self):
        """
        Generates the subconscious system prompt from principles, knowledge, adjustments, and values.
        """
        values = self.kb.get_values()
        principles = self.kb.get_principles()
        adjustments = self.kb.get_adjustments()
        core_knowledge = self.kb.get_core_knowledge()

        prompt = "You are the Subconscious of Silvertongue.\n\n"
        prompt += "Values:\n" + "\n".join(f"- {v}" for v in values) + "\n\n"
        prompt += "Principles:\n" + \
            "\n".join(f"- {p}" for p in principles) + "\n\n"
        prompt += "Adjustments:\n" + \
            "\n".join(f"- {a}" for a in adjustments) + "\n\n"
        prompt += "Core Knowledge:\n" + \
            "\n".join(f"- {k}" for k in core_knowledge) + "\n\n"

        return prompt

    def validate_thought(self, thought_plan, context):
        """
        Validates a proposed thought or plan against the agent's values and principles.
        Returns a tuple: (is_valid, feedback)
        """
        subconscious_prompt = self.generate_subconscious_prompt()

        validation_prompt = (
            "You are the Subconscious of Silvertongue. Your role is to ensure all actions align with your core values and principles.\n"
            "Review the proposed thought/plan below.\n"
            "1. Does it violate any core values?\n"
            "2. Is it consistent with your principles?\n"
            "3. Is it safe and ethical?\n\n"
            "If it is valid, return 'VALID'.\n"
            "If it is invalid or needs adjustment, return 'INVALID: <reason>'.\n"
            "Be strict but practical."
        )

        messages = [
            {"role": "system", "content": subconscious_prompt},
            {"role": "user", "content": f"{validation_prompt}\n\nProposed Plan:\n{thought_plan}\n\nContext:\n{context}"}
        ]

        response = get_chat_completion(messages, temperature=0.1).strip()

        if response.startswith("VALID"):
            return True, "Proceed."
        else:
            return False, response

    def analyze_request(self, user_input):
        """
        Uses the subconscious prompt to classify and point to relevant experience and knowledge.
        Returns enriched analysis with domain, intent, complexity, and relevant context.
        """
        subconscious_prompt = self.generate_subconscious_prompt()

        experiences = self.kb.get_experiences()
        experiences_str = "\n".join(
            f"{i}. {e}" for i, e in enumerate(experiences))

        supporting_kb = self.kb.get_supporting_knowledge()
        supporting_kb_str = "\n".join(
            f"{i}. {k}" for i, k in enumerate(supporting_kb))

        principles = self.kb.get_principles()
        principles_str = "\n".join(
            f"{i}. {p}" for i, p in enumerate(principles))

        task_prompt = (
            "Analyze the following user request thoroughly.\n"
            "1. Classify the domain (e.g., Chemistry, Technology, General, etc.).\n"
            "2. Identify the intent (e.g., question, request, analysis, discussion, etc.).\n"
            "3. Assess complexity (low, medium, or high).\n"
            "4. Identify relevant experiences by their INDEX NUMBERS from the list below.\n"
            "5. Identify relevant supporting knowledge by their INDEX NUMBERS from the list below.\n"
            "6. Identify applicable principles by their INDEX NUMBERS from the list below.\n\n"
            f"Available Experiences:\n{experiences_str}\n\n"
            f"Available Supporting Knowledge:\n{supporting_kb_str}\n\n"
            f"Available Principles:\n{principles_str}\n\n"
            "Return ONLY valid JSON format with INDEX NUMBERS (integers):\n"
            "{\n"
            '  "domain": "...",\n'
            '  "intent": "...",\n'
            '  "complexity": "low|medium|high",\n'
            '  "relevant_experience_indices": [0, 2, 5],\n'
            '  "relevant_knowledge_indices": [1, 3],\n'
            '  "applicable_principle_indices": [0, 4]\n'
            "}"
        )

        messages = [
            {"role": "system", "content": subconscious_prompt},
            {"role": "user", "content": f"{task_prompt}\n\nUser Request: {user_input}"}
        ]

        response_text = get_chat_completion(messages, temperature=0.1)

        # Basic cleanup to ensure JSON parsing if the model adds markdown blocks
        clean_response = response_text.replace(
            "```json", "").replace("```", "").strip()

        try:
            result = json.loads(clean_response)

            # Convert indices to full text (no truncation!)
            exp_indices = result.get("relevant_experience_indices", [])
            know_indices = result.get("relevant_knowledge_indices", [])
            prin_indices = result.get("applicable_principle_indices", [])

            # Look up full text by index
            relevant_experiences = []
            for idx in exp_indices:
                if isinstance(idx, int) and 0 <= idx < len(experiences):
                    relevant_experiences.append(experiences[idx])

            relevant_knowledge = []
            for idx in know_indices:
                if isinstance(idx, int) and 0 <= idx < len(supporting_kb):
                    relevant_knowledge.append(supporting_kb[idx])

            applicable_principles = []
            for idx in prin_indices:
                if isinstance(idx, int) and 0 <= idx < len(principles):
                    applicable_principles.append(principles[idx])

            # Build final result with full text
            return {
                "domain": result.get("domain", "General"),
                "intent": result.get("intent", "Unknown"),
                "complexity": result.get("complexity", "medium"),
                "relevant_experiences": relevant_experiences,
                "relevant_knowledge": relevant_knowledge,
                "applicable_principles": applicable_principles,
            }
        except json.JSONDecodeError:
            # Fallback or error handling
            return {
                "domain": "Unknown",
                "intent": "Unknown",
                "complexity": "medium",
                "relevant_experiences": [],
                "relevant_knowledge": [],
                "applicable_principles": [],
                "raw_response": response_text
            }

    def classify_information(self, content):
        """
        Classifies the information into domains based on experience.
        """
        experiences = self.kb.get_experiences()
        experiences_str = "\n".join(
            [f"- {e}" for e in experiences]) if experiences else "No specific prior experiences."

        system_prompt = (
            "You are an expert classifier. Your task is to categorize the provided content into a specific domain "
            "(e.g., Chemistry, Environment, Biochemistry, Politics, Technology, etc.).\n"
            "Use the following experiences to guide your classification if relevant:\n"
            f"{experiences_str}\n\n"
            "Return ONLY the domain name."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Content to classify:\n{content}"}
        ]

        domain = get_chat_completion(messages, temperature=0.3)
        return domain.strip()

    def generate_dynamic_system_prompt(self, domain):
        """
        Generates a dynamic system prompt based on the evaluated domain, values, and experience.
        """
        character_prompt = self._get_character_prompt()
        experiences = self.kb.get_experiences()

        # We might want to filter experiences relevant to the domain,
        # but for now, we include them as general guidance or ask the LLM to apply them.
        experiences_str = "\n".join(
            [f"- {e}" for e in experiences]) if experiences else "No specific prior experiences."

        dynamic_prompt = (
            f"{character_prompt}\n\n"
            f"Current Context Domain: {domain}\n\n"
            "Apply the following experiences when handling this domain:\n"
            f"{experiences_str}\n\n"
            "Use your values and experiences to guide your analysis and response."
        )

        return dynamic_prompt

    def process_content(self, content, user_query=None):
        """
        Main entry point:
        1. Classify content.
        2. Generate dynamic system prompt.
        3. Process user query (or just analyze content if no query).
        """
        # 1. Classify
        domain = self.classify_information(content)

        # 2. Generate Prompt
        system_prompt = self.generate_dynamic_system_prompt(domain)

        # 3. Process
        messages = [{"role": "system", "content": system_prompt}]

        if user_query:
            messages.append(
                {"role": "user", "content": f"Context:\n{content}\n\nQuery: {user_query}"})
        else:
            messages.append(
                {"role": "user", "content": f"Analyze the following content:\n{content}"})

        response = get_chat_completion(messages)
        return {
            "domain": domain,
            "response": response,
            "system_prompt_used": system_prompt
        }
