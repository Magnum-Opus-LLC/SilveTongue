"""
Lingkui - Cognitive AI Agent Framework

Structure:
- utils/: Helper functions and utilities (chat, knowledge_base, research, actor_generator)
- thoughts/: Cognitive components (conscious, subconscious)
- actions/: Execution components (thought_processor, study_processor, learning_scheduler)
- scholar_search/: Google Scholar search and analysis tools
"""

from src.utils import KnowledgeBase, kb, get_chat_completion, researcher
from src.thoughts import LingkuiConscious, LingkuiSubconscious
from src.actions import ThoughtProcessor, LearningScheduleRunner
from src.scholar_search import ScholarSearcher, scholar_searcher, quick_search, quick_survey

__all__ = [
    # Knowledge & Utils
    'KnowledgeBase',
    'kb',
    'get_chat_completion',
    'researcher',
    # Cognitive Components
    'LingkuiConscious',
    'LingkuiSubconscious',
    'ThoughtProcessor',
    # Learning System
    'LearningScheduleRunner',
    # Scholar Search
    'ScholarSearcher',
    'scholar_searcher',
    'quick_search',
    'quick_survey'
]