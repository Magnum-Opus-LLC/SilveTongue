"""
Report Tool Module
==================
Tools for analyzing and visualizing learning session data.

Classes:
    - LearningSessionAnalyzer: Comprehensive session analysis
    - KnowledgeTimelapse: Time-lapse visualizations of knowledge expansion
    - KnowledgeReportGenerator: Full report generation with wordclouds

Usage:
    from src.report_tool import LearningSessionAnalyzer, KnowledgeTimelapse, KnowledgeReportGenerator
    
    # Quick analysis
    analyzer = LearningSessionAnalyzer('steve')
    analyzer.load_sessions()
    results = analyzer.analyze_all()
    
    # Generate timelapse visualizations
    timelapse = KnowledgeTimelapse('steve')
    timelapse.load_data()
    timelapse.generate_all_visualizations()
    
    # Generate comprehensive report
    generator = KnowledgeReportGenerator('steve')
    generator.load_data()
    generator.generate_markdown_report('runs/report')
"""

from .session_analyzer import LearningSessionAnalyzer
from .timelapse import KnowledgeTimelapse
from .report_generator import KnowledgeReportGenerator

__all__ = [
    'LearningSessionAnalyzer',
    'KnowledgeTimelapse', 
    'KnowledgeReportGenerator'
]
