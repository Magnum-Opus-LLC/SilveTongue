# Student Learning Schedule

This directory contains learning schedules for the student actor.

## Usage

```python
from src.actions.learning_scheduler import LearningScheduleRunner

# Create a runner for the student actor
runner = LearningScheduleRunner('student')

# Create a new schedule pointing to your papers directory
runner.create_schedule(
    schedule_id='my_papers',
    papers_directory='/path/to/your/papers',
    name='My Research Papers',
    focus_areas=['machine learning', 'neural networks'],
    depth='comprehensive'
)

# Run the schedule
runner.run_schedule('my_papers')

# View extracted knowledge
knowledge = runner.get_schedule_knowledge('my_papers')

# Export to markdown
runner.export_knowledge_summary('my_papers', 'summary.md')
```

## Directory Structure

```
learning_schedule/
├── schedules/       # Schedule configuration files
├── notes/           # Generated study notes per schedule
└── knowledge/       # Extracted knowledge bits per schedule
```
