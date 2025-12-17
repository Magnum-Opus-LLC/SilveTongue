#!/usr/bin/env python3
"""
Tests for Sleep Processor Module
================================
Tests for memory consolidation and LoRA training data generation.
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.actions.sleep_processor import (
    SleepProcessor,
    TrainingExample,
    MemoryConsolidation,
    sleep,
    quick_sleep
)


class TestTrainingExample(unittest.TestCase):
    """Tests for TrainingExample dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        example = TrainingExample(
            instruction="Test instruction",
            input="Test input",
            output="Test output",
            source="test",
            category="unit_test"
        )
        
        result = example.to_dict()
        
        self.assertEqual(result['instruction'], "Test instruction")
        self.assertEqual(result['input'], "Test input")
        self.assertEqual(result['output'], "Test output")
        self.assertEqual(result['source'], "test")
        self.assertEqual(result['category'], "unit_test")
    
    def test_to_alpaca_format(self):
        """Test conversion to Alpaca format."""
        example = TrainingExample(
            instruction="Test instruction",
            input="Test input",
            output="Test output"
        )
        
        result = example.to_alpaca_format()
        
        self.assertEqual(set(result.keys()), {'instruction', 'input', 'output'})
        self.assertEqual(result['instruction'], "Test instruction")
    
    def test_to_sharegpt_format(self):
        """Test conversion to ShareGPT format."""
        example = TrainingExample(
            instruction="System prompt",
            input="User message",
            output="Assistant response"
        )
        
        result = example.to_sharegpt_format()
        
        self.assertIn('conversations', result)
        self.assertEqual(len(result['conversations']), 3)
        self.assertEqual(result['conversations'][0]['from'], 'system')
        self.assertEqual(result['conversations'][1]['from'], 'human')
        self.assertEqual(result['conversations'][2]['from'], 'gpt')


class TestMemoryConsolidation(unittest.TestCase):
    """Tests for MemoryConsolidation dataclass."""
    
    def test_default_initialization(self):
        """Test default values."""
        mem = MemoryConsolidation(actor_name='test')
        
        self.assertEqual(mem.actor_name, 'test')
        self.assertEqual(mem.values, [])
        self.assertEqual(mem.principles, [])
        self.assertEqual(mem.total_papers_studied, 0)
    
    def test_can_store_data(self):
        """Test storing data in memory consolidation."""
        mem = MemoryConsolidation(actor_name='test')
        mem.values = ['value1', 'value2']
        mem.core_knowledge = ['knowledge1']
        mem.total_papers_studied = 5
        
        self.assertEqual(len(mem.values), 2)
        self.assertEqual(len(mem.core_knowledge), 1)
        self.assertEqual(mem.total_papers_studied, 5)


class TestSleepProcessor(unittest.TestCase):
    """Tests for SleepProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.TemporaryDirectory()
        
        # Create mock knowledge base
        self.mock_kb = MagicMock()
        self.mock_kb.get_values.return_value = ['Value1', 'Value2']
        self.mock_kb.get_principles.return_value = ['Principle1']
        self.mock_kb.get_core_knowledge.return_value = ['Knowledge item 1', 'Knowledge item 2']
        self.mock_kb.get_supporting_knowledge.return_value = ['Supporting 1']
        self.mock_kb.get_experiences.return_value = ['Experience 1']
        self.mock_kb.get_adjustments.return_value = []
        self.mock_kb.get_challenges.return_value = ['Challenge 1']
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.test_dir.cleanup()
    
    @patch('src.actions.sleep_processor.load_config')
    @patch('src.actions.sleep_processor.KnowledgeBase')
    def test_initialization(self, mock_kb_class, mock_load_config):
        """Test processor initialization."""
        mock_load_config.return_value = {'array_dir': self.test_dir.name}
        mock_kb_class.return_value = self.mock_kb
        
        processor = SleepProcessor('test_actor', verbose=False)
        
        self.assertEqual(processor.actor_name, 'test_actor')
        self.assertIsNone(processor.memories)
        self.assertEqual(processor.training_examples, [])
    
    @patch('src.actions.sleep_processor.load_config')
    @patch('src.actions.sleep_processor.KnowledgeBase')
    def test_consolidate_memories(self, mock_kb_class, mock_load_config):
        """Test memory consolidation from knowledge base."""
        mock_load_config.return_value = {'array_dir': self.test_dir.name}
        mock_kb_class.return_value = self.mock_kb
        
        processor = SleepProcessor('test_actor', verbose=False)
        memories = processor.consolidate_memories()
        
        self.assertIsNotNone(memories)
        self.assertEqual(memories.actor_name, 'test_actor')
        self.assertEqual(len(memories.values), 2)
        self.assertEqual(len(memories.principles), 1)
        self.assertEqual(len(memories.core_knowledge), 2)
    
    @patch('src.actions.sleep_processor.load_config')
    @patch('src.actions.sleep_processor.KnowledgeBase')
    def test_generate_identity_examples(self, mock_kb_class, mock_load_config):
        """Test generation of identity-based training examples."""
        mock_load_config.return_value = {'array_dir': self.test_dir.name}
        mock_kb_class.return_value = self.mock_kb
        
        processor = SleepProcessor('test_actor', verbose=False)
        processor.consolidate_memories()
        processor.generate_training_pairs()
        
        # Should have generated some identity examples
        identity_examples = [ex for ex in processor.training_examples if ex.source == 'identity']
        self.assertGreater(len(identity_examples), 0)
    
    @patch('src.actions.sleep_processor.load_config')
    @patch('src.actions.sleep_processor.KnowledgeBase')
    def test_generate_knowledge_examples(self, mock_kb_class, mock_load_config):
        """Test generation of knowledge-based training examples."""
        mock_load_config.return_value = {'array_dir': self.test_dir.name}
        mock_kb_class.return_value = self.mock_kb
        
        processor = SleepProcessor('test_actor', verbose=False)
        processor.consolidate_memories()
        processor.generate_training_pairs()
        
        # Should have generated knowledge examples
        knowledge_examples = [ex for ex in processor.training_examples if ex.source == 'knowledge_base']
        self.assertGreater(len(knowledge_examples), 0)
    
    @patch('src.actions.sleep_processor.load_config')
    @patch('src.actions.sleep_processor.KnowledgeBase')
    def test_filter_duplicates(self, mock_kb_class, mock_load_config):
        """Test that duplicate examples are filtered out."""
        mock_load_config.return_value = {'array_dir': self.test_dir.name}
        mock_kb_class.return_value = self.mock_kb
        
        processor = SleepProcessor('test_actor', verbose=False)
        
        # Add duplicate examples (outputs need to be 50+ characters to pass min length filter)
        long_output1 = "This is a sufficiently long output that definitely has more than fifty characters in total"
        long_output2 = "This is a different output that also has more than fifty characters to pass the filter"
        
        processor.training_examples = [
            TrainingExample(instruction="Same instruction", input="Same input", output=long_output1),
            TrainingExample(instruction="Same instruction", input="Same input", output=long_output1),
            TrainingExample(instruction="Different instruction", input="Different input", output=long_output2),
        ]
        
        processor._filter_examples()
        
        # Should have removed one duplicate
        self.assertEqual(len(processor.training_examples), 2)
    
    @patch('src.actions.sleep_processor.load_config')
    @patch('src.actions.sleep_processor.KnowledgeBase')
    def test_export_dataset_alpaca(self, mock_kb_class, mock_load_config):
        """Test exporting dataset in Alpaca format."""
        mock_load_config.return_value = {'array_dir': self.test_dir.name}
        mock_kb_class.return_value = self.mock_kb
        
        processor = SleepProcessor('test_actor', verbose=False)
        processor.training_examples = [
            TrainingExample(instruction="Test", input="Input", output="Output that is long enough"),
        ]
        
        output_path = Path(self.test_dir.name) / 'test_output.jsonl'
        processor.export_dataset(str(output_path), format='alpaca')
        
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            self.assertIn('instruction', data)
            self.assertIn('input', data)
            self.assertIn('output', data)
    
    @patch('src.actions.sleep_processor.load_config')
    @patch('src.actions.sleep_processor.KnowledgeBase')
    def test_generate_summary(self, mock_kb_class, mock_load_config):
        """Test summary generation."""
        mock_load_config.return_value = {'array_dir': self.test_dir.name}
        mock_kb_class.return_value = self.mock_kb
        
        processor = SleepProcessor('test_actor', verbose=False)
        processor.consolidate_memories()
        processor.training_examples = [
            TrainingExample(instruction="Test", input="", output="Output", source="test", category="test_cat"),
        ]
        
        summary = processor._generate_summary()
        
        self.assertEqual(summary['actor_name'], 'test_actor')
        self.assertEqual(summary['total_examples'], 1)
        self.assertIn('by_category', summary)
        self.assertIn('by_source', summary)
        self.assertIn('memory_stats', summary)
        self.assertIn('recommended_training_params', summary)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""
    
    @patch('src.actions.sleep_processor.SleepProcessor')
    def test_sleep_function(self, mock_processor_class):
        """Test sleep convenience function."""
        mock_processor = MagicMock()
        mock_processor.sleep.return_value = {'total_examples': 10}
        mock_processor_class.return_value = mock_processor
        
        result = sleep('test_actor')
        
        mock_processor_class.assert_called_once_with('test_actor')
        mock_processor.sleep.assert_called_once()
    
    @patch('src.actions.sleep_processor.SleepProcessor')
    def test_quick_sleep_function(self, mock_processor_class):
        """Test quick_sleep convenience function."""
        mock_processor = MagicMock()
        mock_processor.sleep.return_value = {'output_path': '/path/to/data.jsonl'}
        mock_processor_class.return_value = mock_processor
        
        result = quick_sleep('test_actor')
        
        mock_processor_class.assert_called_once_with('test_actor', verbose=False)


if __name__ == '__main__':
    unittest.main()
