"""
Silvertongue Actions Module

Contains the execution components:
- thought_processor: Main orchestrator for the cognitive loop
- study_processor: Process papers and extract knowledge
- learning_scheduler: Manage and run learning schedules
- cognitive_loop: The unified cognitive loop manifesto
- sleep_processor: Memory consolidation for LoRA fine-tuning
"""

from src.actions.thought_processor import ThoughtProcessor
from src.actions.study_processor import StudyProcessor, study_processor
from src.actions.learning_scheduler import (
    LearningScheduleRunner,
    create_learning_schedule,
    run_learning_schedule,
    discover_papers
)
from src.actions.cognitive_loop import (
    CognitiveLoop,
    CognitiveState,
    LoopPhase,
    learn,
    learn_nonstop,
    run_forever,
    start_learning,
    resume_learning
)
from src.actions.sleep_processor import (
    SleepProcessor,
    sleep,
    quick_sleep
)

__all__ = [
    # Thought processing
    'ThoughtProcessor',
    
    # Study processing  
    'StudyProcessor',
    'study_processor',
    
    # Learning schedules
    'LearningScheduleRunner',
    'create_learning_schedule',
    'run_learning_schedule',
    'discover_papers',
    
    # Cognitive Loop (The Manifesto)
    'CognitiveLoop',
    'CognitiveState',
    'LoopPhase',
    'learn',
    'learn_nonstop',
    'run_forever',
    'start_learning',
    'resume_learning',
    
    # Sleep (LoRA Training Data Generation)
    'SleepProcessor',
    'sleep',
    'quick_sleep'
]
