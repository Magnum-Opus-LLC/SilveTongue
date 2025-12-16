"""
Analysis Tool Module
====================
Tools for analyzing learning session data and knowledge evolution.

Classes:
    TopicBuzzwordAnalyzer: Analyze buzzwords and topic patterns
    KnowledgeAnalyzer: Deep knowledge evolution analysis
    IterationAnalyzer: Iteration-based trend analysis
    KnowledgeScopeAnalyzer: Knowledge scope expansion analysis
    KnowledgeEvolutionAnalyzer: Basic knowledge evolution metrics

Usage:
    from src.analysis_tool import (
        TopicBuzzwordAnalyzer,
        KnowledgeAnalyzer,
        IterationAnalyzer,
        KnowledgeScopeAnalyzer,
        KnowledgeEvolutionAnalyzer
    )
    
    # Analyze buzzwords
    analyzer = TopicBuzzwordAnalyzer('steve')
    analyzer.analyze_topic_buzzwords()
    
    # Deep knowledge analysis
    ka = KnowledgeAnalyzer('steve')
    ka.analyze_principle_timeline()
    
    # Iteration trends
    ia = IterationAnalyzer('steve')
    ia.analyze_learning_velocity()
    
    # Scope expansion
    sa = KnowledgeScopeAnalyzer('steve')
    sa.analyze_scope_expansion()
"""

from .buzzword_analyzer import TopicBuzzwordAnalyzer
from .knowledge_analyzer import KnowledgeAnalyzer
from .iteration_analyzer import IterationAnalyzer
from .scope_analyzer import KnowledgeScopeAnalyzer
from .evolution_analyzer import KnowledgeEvolutionAnalyzer

__all__ = [
    'TopicBuzzwordAnalyzer',
    'KnowledgeAnalyzer',
    'IterationAnalyzer',
    'KnowledgeScopeAnalyzer',
    'KnowledgeEvolutionAnalyzer'
]
