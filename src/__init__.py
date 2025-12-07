"""
Lingkui - Cognitive AI Agent Framework

Structure:
- utils/: Helper functions and utilities (chat, knowledge_base, research, actor_generator)
- thoughts/: Cognitive components (conscious, subconscious)
- actions/: Execution components (thought_processor, tools)
"""

from src.utils import KnowledgeBase, kb, get_chat_completion, researcher
from src.thoughts import LingkuiConscious, LingkuiSubconscious
from src.actions import ThoughtProcessor

__all__ = [
    'KnowledgeBase',
    'kb',
    'get_chat_completion',
    'researcher',
    'LingkuiConscious',
    'LingkuiSubconscious',
    'ThoughtProcessor'
]