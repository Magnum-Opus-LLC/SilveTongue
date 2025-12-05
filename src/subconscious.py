import json
from knowledge_base import kb
from chat_utils import get_chat_completion


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
