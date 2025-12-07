import json
from src.utils.knowledge_base import kb
from src.utils.chat_utils import get_chat_completion


class LingkuiSubconscious:
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

        prompt = "You are the Subconscious of Lingkui.\n\n"
        prompt += "Values:\n" + "\n".join(f"- {v}" for v in values) + "\n\n"
        prompt += "Principles:\n" + "\n".join(f"- {p}" for p in principles) + "\n\n"
        prompt += "Adjustments:\n" + "\n".join(f"- {a}" for a in adjustments) + "\n\n"
        prompt += "Core Knowledge:\n" + "\n".join(f"- {k}" for k in core_knowledge) + "\n\n"
        
        return prompt

    def validate_thought(self, thought_plan, context):
        """
        Validates a proposed thought or plan against the agent's values and principles.
        Returns a tuple: (is_valid, feedback)
        """
        subconscious_prompt = self.generate_subconscious_prompt()
        
        validation_prompt = (
            "You are the Subconscious of Lingkui. Your role is to ensure all actions align with your core values and principles.\n"
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
        """
        subconscious_prompt = self.generate_subconscious_prompt()
        
        experiences = self.kb.get_experiences()
        experiences_str = "\n".join(f"{i}. {e}" for i, e in enumerate(experiences))
        
        supporting_kb = self.kb.get_supporting_knowledge()
        supporting_kb_str = "\n".join(f"{i}. {k}" for i, k in enumerate(supporting_kb))

        task_prompt = (
            "Analyze the following user request.\n"
            "1. Classify the domain.\n"
            "2. Identify relevant experiences from the list below (return the text).\n"
            "3. Identify relevant supporting knowledge from the list below (return the text).\n\n"
            f"Available Experiences:\n{experiences_str}\n\n"
            f"Available Supporting Knowledge:\n{supporting_kb_str}\n\n"
            "Return ONLY valid JSON format: {\"domain\": \"...\", \"relevant_experiences\": [\"...\"], \"relevant_knowledge\": [\"...\"]}"
        )

        messages = [
            {"role": "system", "content": subconscious_prompt},
            {"role": "user", "content": f"{task_prompt}\n\nUser Request: {user_input}"}
        ]

        response_text = get_chat_completion(messages, temperature=0.1)
        
        # Basic cleanup to ensure JSON parsing if the model adds markdown blocks
        clean_response = response_text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(clean_response)
        except json.JSONDecodeError:
            # Fallback or error handling
            return {
                "domain": "Unknown",
                "relevant_experiences": [],
                "relevant_knowledge": [],
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
