"""
Lingkui Thoughts Module

Contains the cognitive components that generate system prompts and reasoning:
- conscious: High-level planning and decision making
- subconscious: Classification, retrieval, and validation
"""

from src.thoughts.conscious import LingkuiConscious
from src.thoughts.subconscious import LingkuiSubconscious

__all__ = [
    'LingkuiConscious',
    'LingkuiSubconscious'
]
