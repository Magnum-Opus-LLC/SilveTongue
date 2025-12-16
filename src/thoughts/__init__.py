"""
Silvertongue Thoughts Module

Contains the cognitive components that generate system prompts and reasoning:
- conscious: High-level planning and decision making
- subconscious: Classification, retrieval, and validation
"""

from src.thoughts.conscious import SilvertongueConscious
from src.thoughts.subconscious import SilvertongueSubconscious

__all__ = [
    'SilvertongueConscious',
    'SilvertongueSubconscious'
]
