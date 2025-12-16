#!/usr/bin/env python3
"""
Tests for Analysis Tool Module
==============================
Tests for buzzword, knowledge, iteration, scope, and evolution analyzers.
"""

import unittest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import analyzers
from src.analysis_tool import (
    TopicBuzzwordAnalyzer,
    KnowledgeAnalyzer,
    IterationAnalyzer,
    KnowledgeScopeAnalyzer,
    KnowledgeEvolutionAnalyzer
)


class TestTopicBuzzwordAnalyzer(unittest.TestCase):
    """Tests for TopicBuzzwordAnalyzer."""
    
    def test_extract_buzzwords(self):
        """Test buzzword extraction from text."""
        # Create minimal mock
        with patch.object(TopicBuzzwordAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = TopicBuzzwordAnalyzer.__new__(TopicBuzzwordAnalyzer)
            analyzer.BUZZWORDS = ['transformer', 'attention', 'neural', 'model', 'edge']
            
            text = "Transformer models use attention mechanisms for neural network processing on edge devices"
            result = analyzer.extract_buzzwords(text)
            
            self.assertIn('transformer', result)
            self.assertIn('attention', result)
            self.assertIn('neural', result)
            self.assertIn('edge', result)
    
    def test_extract_buzzwords_case_insensitive(self):
        """Test that buzzword extraction is case insensitive."""
        with patch.object(TopicBuzzwordAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = TopicBuzzwordAnalyzer.__new__(TopicBuzzwordAnalyzer)
            analyzer.BUZZWORDS = ['transformer', 'attention']
            
            text = "TRANSFORMER and Attention mechanisms"
            result = analyzer.extract_buzzwords(text)
            
            self.assertIn('transformer', result)
            self.assertIn('attention', result)
    
    def test_extract_buzzwords_empty_text(self):
        """Test buzzword extraction with empty text."""
        with patch.object(TopicBuzzwordAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = TopicBuzzwordAnalyzer.__new__(TopicBuzzwordAnalyzer)
            analyzer.BUZZWORDS = ['transformer', 'attention']
            
            result = analyzer.extract_buzzwords("")
            
            self.assertEqual(len(result), 0)


class TestKnowledgeAnalyzer(unittest.TestCase):
    """Tests for KnowledgeAnalyzer."""
    
    def test_categorize_domain_ml_architecture(self):
        """Test domain categorization for ML architecture."""
        with patch.object(KnowledgeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeAnalyzer.__new__(KnowledgeAnalyzer)
            analyzer.DOMAIN_KEYWORDS = {
                "ML Architecture": ['transformer', 'attention', 'cnn', 'rnn'],
                "Edge AI & Efficiency": ['edge', 'latency', 'quantization'],
            }
            
            result = analyzer.categorize_domain("Transformer architecture with attention layers")
            
            self.assertIn("ML Architecture", result)
    
    def test_categorize_domain_multiple(self):
        """Test that multiple domains can be detected."""
        with patch.object(KnowledgeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeAnalyzer.__new__(KnowledgeAnalyzer)
            analyzer.DOMAIN_KEYWORDS = {
                "ML Architecture": ['transformer'],
                "Edge AI & Efficiency": ['edge', 'latency'],
            }
            
            result = analyzer.categorize_domain("Transformer model deployed on edge with low latency")
            
            self.assertIn("ML Architecture", result)
            self.assertIn("Edge AI & Efficiency", result)
    
    def test_categorize_domain_general(self):
        """Test that unrecognized text returns General."""
        with patch.object(KnowledgeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeAnalyzer.__new__(KnowledgeAnalyzer)
            analyzer.DOMAIN_KEYWORDS = {
                "ML Architecture": ['transformer'],
            }
            
            result = analyzer.categorize_domain("Some random text")
            
            self.assertEqual(result, ["General"])
    
    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        with patch.object(KnowledgeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeAnalyzer.__new__(KnowledgeAnalyzer)
            analyzer.KEY_TERMS = ['transformer', 'quantization', 'pruning']
            
            result = analyzer.extract_key_phrases("Quantization and pruning for transformer models")
            
            self.assertIn('transformer', result)
            self.assertIn('quantization', result)
            self.assertIn('pruning', result)


class TestIterationAnalyzer(unittest.TestCase):
    """Tests for IterationAnalyzer."""
    
    def test_categorize_domain(self):
        """Test topic to domain categorization."""
        with patch.object(IterationAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = IterationAnalyzer.__new__(IterationAnalyzer)
            
            self.assertEqual(analyzer._categorize_domain("Edge deployment strategies"), "Edge Deployment")
            self.assertEqual(analyzer._categorize_domain("Continual learning with replay"), "Continual Learning")
            self.assertEqual(analyzer._categorize_domain("Adversarial robustness"), "Robustness")
            self.assertEqual(analyzer._categorize_domain("Random topic"), "General")
    
    def test_categorize_mindset(self):
        """Test mindset categorization."""
        with patch.object(IterationAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = IterationAnalyzer.__new__(IterationAnalyzer)
            
            self.assertEqual(analyzer._categorize_mindset("Improved efficiency through quantization"), "Efficiency Focus")
            self.assertEqual(analyzer._categorize_mindset("Better robustness against adversarial attacks"), "Robustness Focus")
            self.assertEqual(analyzer._categorize_mindset("Continual learning with replay buffer"), "Continual Learning")
            self.assertEqual(analyzer._categorize_mindset("Some general change"), "General")
    
    def test_extract_papers_count(self):
        """Test paper count extraction from insights."""
        with patch.object(IterationAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = IterationAnalyzer.__new__(IterationAnalyzer)
            
            insights = ["Studied 42 papers on transformers", "Gained new understanding"]
            result = analyzer._extract_papers_count(insights)
            
            self.assertEqual(result, 42)
    
    def test_extract_items_persisted(self):
        """Test items persisted extraction from insights."""
        with patch.object(IterationAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = IterationAnalyzer.__new__(IterationAnalyzer)
            
            insights = ["Persisted 15 items to knowledge base"]
            result = analyzer._extract_items_persisted(insights)
            
            self.assertEqual(result, 15)


class TestKnowledgeScopeAnalyzer(unittest.TestCase):
    """Tests for KnowledgeScopeAnalyzer."""
    
    def test_extract_themes(self):
        """Test theme extraction."""
        with patch.object(KnowledgeScopeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeScopeAnalyzer.__new__(KnowledgeScopeAnalyzer)
            analyzer.THEME_PATTERNS = {
                "Transformer Architecture": ['transformer', 'attention mechanism'],
                "Model Compression": ['quantization', 'pruning'],
            }
            
            result = analyzer._extract_themes("Transformer with attention mechanism and quantization")
            
            self.assertIn("Transformer Architecture", result)
            self.assertIn("Model Compression", result)
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        with patch.object(KnowledgeScopeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeScopeAnalyzer.__new__(KnowledgeScopeAnalyzer)
            analyzer.KEYWORDS = ['transformer', 'attention', 'edge', 'latency']
            
            result = analyzer._extract_keywords("Transformer deployed on edge with low latency")
            
            self.assertIn('transformer', result)
            self.assertIn('edge', result)
            self.assertIn('latency', result)
    
    def test_categorize_to_domain(self):
        """Test domain categorization."""
        with patch.object(KnowledgeScopeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeScopeAnalyzer.__new__(KnowledgeScopeAnalyzer)
            
            self.assertEqual(analyzer._categorize_to_domain("continual learning with replay"), "Continual Learning")
            self.assertEqual(analyzer._categorize_to_domain("quantization and pruning"), "Model Efficiency")
            self.assertEqual(analyzer._categorize_to_domain("adversarial robustness"), "Robustness & Security")
            self.assertEqual(analyzer._categorize_to_domain("random text here"), "General ML")


class TestKnowledgeEvolutionAnalyzer(unittest.TestCase):
    """Tests for KnowledgeEvolutionAnalyzer."""
    
    def test_extract_topics(self):
        """Test topic extraction from text."""
        with patch.object(KnowledgeEvolutionAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeEvolutionAnalyzer.__new__(KnowledgeEvolutionAnalyzer)
            analyzer.TOPIC_PATTERNS = [r'transformer[s]?', r'attention', r'deep learning']
            
            result = analyzer._extract_topics("Transformers and attention in deep learning")
            
            self.assertIn('transformer', result)
            self.assertIn('attention', result)
            self.assertIn('deep learning', result)
    
    def test_extract_topics_empty(self):
        """Test topic extraction with no matches."""
        with patch.object(KnowledgeEvolutionAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeEvolutionAnalyzer.__new__(KnowledgeEvolutionAnalyzer)
            analyzer.TOPIC_PATTERNS = [r'transformer[s]?', r'attention']
            
            result = analyzer._extract_topics("Random text with no topics")
            
            self.assertEqual(len(result), 0)


class TestAnalyzerIntegration(unittest.TestCase):
    """Integration tests using real data if available."""
    
    @unittest.skipUnless(
        Path("array/actors/steve/learning_book.json").exists(),
        "Steve actor data not available"
    )
    def test_buzzword_analyzer_with_steve(self):
        """Test buzzword analyzer with real Steve data."""
        analyzer = TopicBuzzwordAnalyzer('steve')
        
        self.assertGreater(len(analyzer.topics), 0)
        
        result = analyzer.analyze_topic_buzzwords()
        self.assertIn('overall', result)
        self.assertIn('by_topic', result)
    
    @unittest.skipUnless(
        Path("array/actors/steve/learning_book.json").exists(),
        "Steve actor data not available"
    )
    def test_knowledge_analyzer_with_steve(self):
        """Test knowledge analyzer with real Steve data."""
        analyzer = KnowledgeAnalyzer('steve')
        
        summary = analyzer.get_summary()
        self.assertGreater(summary['total_items'], 0)
    
    @unittest.skipUnless(
        Path("array/actors/steve/learning_book.json").exists(),
        "Steve actor data not available"
    )
    def test_iteration_analyzer_with_steve(self):
        """Test iteration analyzer with real Steve data."""
        analyzer = IterationAnalyzer('steve')
        
        summary = analyzer.get_summary()
        self.assertGreater(summary['total_iterations'], 0)
    
    @unittest.skipUnless(
        Path("array/actors/steve/learning_book.json").exists(),
        "Steve actor data not available"
    )
    def test_scope_analyzer_with_steve(self):
        """Test scope analyzer with real Steve data."""
        analyzer = KnowledgeScopeAnalyzer('steve')
        
        summary = analyzer.get_summary()
        self.assertGreater(summary['total_iterations'], 0)
        self.assertGreater(summary['final_themes'], 0)
    
    @unittest.skipUnless(
        Path("array/actors/steve/learning_book.json").exists(),
        "Steve actor data not available"
    )
    def test_evolution_analyzer_with_steve(self):
        """Test evolution analyzer with real Steve data."""
        analyzer = KnowledgeEvolutionAnalyzer('steve')
        
        stats = analyzer.get_growth_statistics()
        self.assertGreater(stats['total_iterations'], 0)


if __name__ == "__main__":
    unittest.main()
