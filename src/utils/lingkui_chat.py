"""
Lingkui Chat Interface

Direct conversation with the Lingkui agent, using its accumulated knowledge,
principles, and experiences to inform responses.

The agent maintains its character and draws from:
- Core principles and values
- Learned knowledge from papers
- Past experiences and insights
- Current challenges and questions
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from src.utils.chat_utils import get_chat_completion, load_config
from src.utils.knowledge_base import KnowledgeBase


SYSTEM_PROMPT_TEMPLATE = """You are {actor_name}, an AI learning agent with evolving knowledge and perspectives.

YOUR IDENTITY:
{identity}

YOUR CORE PRINCIPLES:
{principles}

YOUR VALUES:
{values}

YOUR CURRENT KNOWLEDGE:
{knowledge}

YOUR EXPERIENCES:
{experiences}

CURRENT CHALLENGES YOU'RE THINKING ABOUT:
{challenges}

YOUR CURRENT MINDSET:
{mindset}

INSTRUCTIONS:
- Respond as {actor_name}, drawing on your principles, knowledge, and experiences
- Be thoughtful and intellectually curious
- When you don't know something, say so honestly
- Reference your learning experiences when relevant
- Maintain intellectual humility - acknowledge what you're still learning
- Express your genuine perspective based on your accumulated knowledge
- If asked about topics you've studied, share specific insights from your learning

Engage in natural conversation while staying true to your character and knowledge."""


class LingkuiChat:
    """
    Interactive chat interface with the Lingkui agent.
    """
    
    def __init__(self, actor_name: str = "student"):
        self.actor_name = actor_name
        self.kb = KnowledgeBase(actor_name)
        self.conversation_history: List[Dict] = []
        self.system_prompt = self._build_system_prompt()
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt from actor's knowledge base."""
        # Get identity/behavior
        behavior_path = self.kb.actor_root / 'behavior.json'
        identity = "A curious learning agent"
        if behavior_path.exists():
            try:
                with open(behavior_path) as f:
                    behavior = json.load(f)
                    identity = behavior.get('identity', identity)
            except:
                pass
        
        # Get principles
        principles = self.kb.get_principles()
        principles_str = "\n".join([f"- {p}" for p in principles[:10]]) if principles else "- Still developing core principles"
        
        # Get values
        values = self.kb.get_values()
        values_str = "\n".join([f"- {v}" for v in values[:10]]) if values else "- Intellectual curiosity\n- Continuous learning"
        
        # Get knowledge
        core_knowledge = self.kb.get_core_knowledge()
        knowledge_items = []
        if core_knowledge:
            for item in core_knowledge[:15]:
                if isinstance(item, dict):
                    knowledge_items.append(f"- {item.get('title', item.get('topic', str(item)))}")
                else:
                    knowledge_items.append(f"- {item}")
        knowledge_str = "\n".join(knowledge_items) if knowledge_items else "- Currently building knowledge base through learning"
        
        # Get experiences
        experiences = self.kb.get_experiences()
        exp_items = []
        if experiences:
            for exp in experiences[:10]:
                if isinstance(exp, dict):
                    exp_items.append(f"- {exp.get('description', exp.get('insight', str(exp)))}")
                else:
                    exp_items.append(f"- {exp}")
        experiences_str = "\n".join(exp_items) if exp_items else "- Actively learning and building experiences"
        
        # Get challenges
        challenges = self.kb.get_challenges()
        challenge_items = []
        if challenges:
            for ch in challenges[:5]:
                if isinstance(ch, dict):
                    challenge_items.append(f"- {ch.get('challenge', ch.get('description', str(ch)))}")
                else:
                    challenge_items.append(f"- {ch}")
        challenges_str = "\n".join(challenge_items) if challenge_items else "- Exploring various intellectual challenges"
        
        # Get mindset
        mindset = self.kb.get_current_mindset()
        if mindset:
            mindset_str = json.dumps(mindset, indent=2)
        else:
            mindset_str = "Open-minded, curious, and eager to learn"
        
        return SYSTEM_PROMPT_TEMPLATE.format(
            actor_name=self.actor_name,
            identity=identity,
            principles=principles_str,
            values=values_str,
            knowledge=knowledge_str,
            experiences=experiences_str,
            challenges=challenges_str,
            mindset=mindset_str
        )
    
    def refresh_knowledge(self):
        """Refresh the system prompt with latest knowledge."""
        self.kb._cache = {}  # Clear cache
        self.system_prompt = self._build_system_prompt()
        print("🔄 Knowledge refreshed")
    
    def chat(self, user_message: str, temperature: float = 0.7) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: The user's message
            temperature: Response creativity (0.0-1.0)
            
        Returns:
            The agent's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages for API call
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history
        
        # Get response
        response = get_chat_completion(messages, temperature=temperature)
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return "No conversation yet."
        
        turns = len(self.conversation_history) // 2
        return f"{turns} exchanges in this conversation"
    
    def save_conversation(self, filepath: Optional[Path] = None) -> Path:
        """Save the conversation to a file."""
        if filepath is None:
            from src.utils.knowledge_base import get_array_dir
            array_dir = get_array_dir()
            conversations_dir = array_dir / "data" / self.actor_name / "conversations"
            conversations_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = conversations_dir / f"chat_{timestamp}.json"
        
        data = {
            "actor_name": self.actor_name,
            "timestamp": datetime.now().isoformat(),
            "turns": len(self.conversation_history) // 2,
            "conversation": self.conversation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath


def interactive_chat(actor_name: str = "student"):
    """
    Run an interactive chat session with Lingkui.
    """
    print("\n" + "=" * 60)
    print(f"💬 CHAT WITH LINGKUI ({actor_name})")
    print("=" * 60)
    print("""
Commands:
  /clear    - Clear conversation history
  /refresh  - Refresh knowledge from files
  /save     - Save conversation
  /info     - Show actor info
  /help     - Show this help
  /quit     - Exit chat
    """)
    print("-" * 60)
    
    chat = LingkuiChat(actor_name)
    
    print(f"\n🧠 {actor_name} is ready to chat!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith('/'):
            cmd = user_input.lower().split()[0]
            
            if cmd == '/quit' or cmd == '/exit' or cmd == '/q':
                save = input("Save conversation before exiting? [y/N]: ").strip().lower()
                if save == 'y':
                    path = chat.save_conversation()
                    print(f"💾 Saved to: {path}")
                print("Goodbye! 👋")
                break
            
            elif cmd == '/clear':
                chat.clear_history()
            
            elif cmd == '/refresh':
                chat.refresh_knowledge()
            
            elif cmd == '/save':
                path = chat.save_conversation()
                print(f"💾 Saved to: {path}")
            
            elif cmd == '/info':
                print(f"\n📊 Actor: {actor_name}")
                print(f"   Principles: {len(chat.kb.get_principles())}")
                print(f"   Knowledge items: {len(chat.kb.get_core_knowledge())}")
                print(f"   Experiences: {len(chat.kb.get_experiences())}")
                print(f"   {chat.get_conversation_summary()}\n")
            
            elif cmd == '/help':
                print("""
Commands:
  /clear    - Clear conversation history  
  /refresh  - Refresh knowledge from files
  /save     - Save conversation
  /info     - Show actor info
  /help     - Show this help
  /quit     - Exit chat
                """)
            
            else:
                print(f"Unknown command: {cmd}")
            
            continue
        
        # Regular chat
        try:
            response = chat.chat(user_input)
            print(f"\n{actor_name}: {response}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# Convenience function
def chat_with_lingkui(actor_name: str = "student"):
    """Start a chat session with Lingkui."""
    interactive_chat(actor_name)


if __name__ == "__main__":
    import sys
    actor = sys.argv[1] if len(sys.argv) > 1 else "student"
    interactive_chat(actor)
