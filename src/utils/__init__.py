"""
Lingkui Utilities Module

Contains helper functions and classes:
- chat_utils: OpenAI API wrapper
- knowledge_base: Actor knowledge management
- actor_generator: Profile generation tool
- research: Web search and scraping
- paper_parser: Parse papers from PDF, TXT, MD
- database: SQLite storage (optional)
"""

from src.utils.chat_utils import get_chat_completion, load_config
from src.utils.knowledge_base import KnowledgeBase, kb
from src.utils.actor_generator import generate_actor_profile, save_actor_profile
from src.utils.research import researcher, ResearchManager
from src.utils.paper_parser import PaperParser, paper_parser

__all__ = [
    'get_chat_completion',
    'load_config',
    'KnowledgeBase',
    'kb',
    'generate_actor_profile',
    'save_actor_profile',
    'researcher',
    'ResearchManager',
    'PaperParser',
    'paper_parser'
]
