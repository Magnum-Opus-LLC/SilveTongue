"""
Lingkui Actions Module

Contains the execution components:
- thought_processor: Main orchestrator for the cognitive loop
- (future) tools: External tool integrations
- (future) scripts: Automation scripts
"""

from src.actions.thought_processor import ThoughtProcessor

__all__ = [
    'ThoughtProcessor'
]
