#!/usr/bin/env python3
"""
Tests for Analysis Tool Module
==============================
Tests for buzzword, knowledge, iteration, scope, and evolution analyzers.
"""

import pytest
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


class TestTopicBuzzwordAnalyzer:
    """Tests for TopicBuzzwordAnalyzer."""
    
    def test_extract_buzzwords(self):
        """Test buzzword extraction from text."""
        # Create minimal mock
        with patch.object(TopicBuzzwordAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = TopicBuzzwordAnalyzer.__new__(TopicBuzzwordAnalyzer)
            analyzer.BUZZWORDS = ['transformer', 'attention', 'neural', 'model', 'edge']
            
            text = "Transformer models use attention mechanisms for neural network processing on edge devices"
            result = analyzer.extract_buzzwords(text)
            
            assert 'transformer' in result
            assert 'attention' in result
            assert 'neural' in result
            assert 'edge' in result
    
    def test_extract_buzzwords_case_insensitive(self):
        """Test that buzzword extraction is case insensitive."""
        with patch.object(TopicBuzzwordAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = TopicBuzzwordAnalyzer.__new__(TopicBuzzwordAnalyzer)
            analyzer.BUZZWORDS = ['transformer', 'attention']
            
            text = "TRANSFORMER and Attention mechanisms"
            result = analyzer.extract_buzzwords(text)
            
            assert 'transformer' in result
            assert 'attention' in result
    
    def test_extract_buzzwords_empty_text(self):
        """Test buzzword extraction with empty text."""
        with patch.object(TopicBuzzwordAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = TopicBuzzwordAnalyzer.__new__(TopicBuzzwordAnalyzer)
            analyzer.BUZZWORDS = ['transformer', 'attention']
            
            result = analyzer.extract_buzzwords("")
            
            assert len(result) == 0


class TestKnowledgeAnalyzer:
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
            
            assert "ML Architecture" in result
    
    def test_categorize_domain_multiple(self):
        """Test that multiple domains can be detected."""
        with patch.object(KnowledgeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeAnalyzer.__new__(KnowledgeAnalyzer)
            analyzer.DOMAIN_KEYWORDS = {
                "ML Architecture": ['transformer'],
                "Edge AI & Efficiency": ['edge', 'latency'],
            }
            
            result = analyzer.categorize_domain("Transformer model deployed on edge with low latency")
            
            assert "ML Architecture" in result
            assert "Edge AI & Efficiency" in result
    
    def test_categorize_domain_general(self):
        """Test that unrecognized text returns General."""
        with patch.object(KnowledgeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeAnalyzer.__new__(KnowledgeAnalyzer)
            analyzer.DOMAIN_KEYWORDS = {
                "ML Architecture": ['transformer'],
            }
            
            result = analyzer.categorize_domain("Some random text")
            
            assert result == ["General"]
    
    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        with patch.object(KnowledgeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeAnalyzer.__new__(KnowledgeAnalyzer)
            analyzer.KEY_TERMS = ['transformer', 'quantization', 'pruning']
            
            result = analyzer.extract_key_phrases("Quantization and pruning for transformer models")
            
            assert 'transformer' in result
            assert 'quantization' in result
            assert 'pruning' in result


class TestIterationAnalyzer:
    """Tests for IterationAnalyzer."""
    
    def test_categorize_domain(self):
        """Test topic to domain categorization."""
        with patch.object(IterationAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = IterationAnalyzer.__new__(IterationAnalyzer)
            
            assert analyzer._categorize_domain("Edge deployment strategies") == "Edge Deployment"
            assert analyzer._categorize_domain("Continual learning with replay") == "Continual Learning"
            assert analyzer._categorize_domain("Adversarial robustness") == "Robustness"
            assert analyzer._categorize_domain("Random topic") == "General"
    
    def test_categorize_mindset(self):
        """Test mindset categorization."""
        with patch.object(IterationAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = IterationAnalyzer.__new__(IterationAnalyzer)
            
            assert analyzer._categorize_mindset("Improved efficiency through quantization") == "Efficiency Focus"
            assert analyzer._categorize_mindset("Better robustness against adversarial attacks") == "Robustness Focus"
            assert analyzer._categorize_mindset("Continual learning with replay buffer") == "Continual Learning"
            assert analyzer._categorize_mindset("Some general change") == "General"
    
    def test_extract_papers_count(self):
        """Test paper count extraction from insights."""
        with patch.object(IterationAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = IterationAnalyzer.__new__(IterationAnalyzer)
            
            insights = ["Studied 42 papers on transformers", "Gained new understanding"]
            result = analyzer._extract_papers_count(insights)
            
            assert result == 42
    
    def test_extract_items_persisted(self):
        """Test items persisted extraction from insights."""
        with patch.object(IterationAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = IterationAnalyzer.__new__(IterationAnalyzer)
            
            insights = ["Persisted 15 items to knowledge base"]
            result = analyzer._extract_items_persisted(insights)
            
            assert result == 15


class TestKnowledgeScopeAnalyzer:
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
            
            assert "Transformer Architecture" in result
            assert "Model Compression" in result
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        with patch.object(KnowledgeScopeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeScopeAnalyzer.__new__(KnowledgeScopeAnalyzer)
            analyzer.KEYWORDS = ['transformer', 'attention', 'edge', 'latency']
            
            result = analyzer._extract_keywords("Transformer deployed on edge with low latency")
            
            assert 'transformer' in result
            assert 'edge' in result
            assert 'latency' in result
    
    def test_categorize_to_domain(self):
        """Test domain categorization."""
        with patch.object(KnowledgeScopeAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeScopeAnalyzer.__new__(KnowledgeScopeAnalyzer)
            
            assert analyzer._categorize_to_domain("continual learning with replay") == "Continual Learning"
            assert analyzer._categorize_to_domain("quantization and pruning") == "Model Efficiency"
            assert analyzer._categorize_to_domain("adversarial robustness") == "Robustness & Security"
            assert analyzer._categorize_to_domain("random text here") == "General ML"


class TestKnowledgeEvolutionAnalyzer:
    """Tests for KnowledgeEvolutionAnalyzer."""
    
    def test_extract_topics(self):
        """Test topic extraction from text."""
        with patch.object(KnowledgeEvolutionAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeEvolutionAnalyzer.__new__(KnowledgeEvolutionAnalyzer)
            analyzer.TOPIC_PATTERNS = [r'transformer[s]?', r'attention', r'deep learning']
            
            result = analyzer._extract_topics("Transformers and attention in deep learning")
            
            assert 'transformer' in result
            assert 'attention' in result
            assert 'deep learning' in result
    
    def test_extract_topics_empty(self):
        """Test topic extraction with no matches."""
        with patch.object(KnowledgeEvolutionAnalyzer, '__init__', lambda x, y, z="array": None):
            analyzer = KnowledgeEvolutionAnalyzer.__new__(KnowledgeEvolutionAnalyzer)
            analyzer.TOPIC_PATTERNS = [r'transformer[s]?', r'attention']
            
            result = analyzer._extract_topics("Random text with no topics")
            
            assert len(result) == 0


class TestAnalyzerIntegration:
    """Integration tests using real data if available."""
    
    @pytest.fixture
    def steve_data_available(self):
        """Check if Steve's data is available."""
        path = Path("array/actors/steve/learning_book.json")
        return path.exists()
    
    @pytest.mark.skipif(
        not Path("array/actors/steve/learning_book.json").exists(),
        reason="Steve actor data not available"
    )
    def test_buzzword_analyzer_with_steve(self):
        """Test buzzword analyzer with real Steve data."""
        analyzer = TopicBuzzwordAnalyzer('steve')
        
        assert len(analyzer.topics) > 0
        
        result = analyzer.analyze_topic_buzzwords()
        assert 'overall' in result
        assert 'by_topic' in result
    
    @pytest.mark.skipif(
        not Path("array/actors/steve/learning_book.json").exists(),
        reason="Steve actor data not available"
    )
    def test_knowledge_analyzer_with_steve(self):
        """Test knowledge analyzer with real Steve data."""
        analyzer = KnowledgeAnalyzer('steve')
        
        summary = analyzer.get_summary()
        assert summary['total_items'] > 0
    
    @pytest.mark.skipif(
        not Path("array/actors/steve/learning_book.json").exists(),
        reason="Steve actor data not available"
    )
    def test_iteration_analyzer_with_steve(self):
        """Test iteration analyzer with real Steve data."""
        analyzer = IterationAnalyzer('steve')
        
        summary = analyzer.get_summary()
        assert summary['total_iterations'] > 0
    
    @pytest.mark.skipif(
        not Path("array/actors/steve/learning_book.json").exists(),
        reason="Steve actor data not available"
    )
    def test_scope_analyzer_with_steve(self):
        """Test scope analyzer with real Steve data."""
        analyzer = KnowledgeScopeAnalyzer('steve')
        
        summary = analyzer.get_summary()
        assert summary['total_iterations'] > 0
        assert summary['final_themes'] > 0
    
    @pytest.mark.skipif(
        not Path("array/actors/steve/learning_book.json").exists(),
        reason="Steve actor data not available"
    )
    def test_evolution_analyzer_with_steve(self):
        """Test evolution analyzer with real Steve data."""
        analyzer = KnowledgeEvolutionAnalyzer('steve')
        
        stats = analyzer.get_growth_statistics()
        assert stats['total_iterations'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
