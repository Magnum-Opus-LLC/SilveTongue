"""
Sleep Processor - Memory Consolidation for LoRA Fine-Tuning

Just like humans consolidate memories during sleep, this module processes
an actor's accumulated knowledge, experiences, and learning sessions into
training data for LoRA fine-tuning.

The result is a unique "mind" that can be trained into a base model,
giving it the actor's:
- Personality and values
- Domain expertise
- Reasoning patterns
- Communication style

Output Format: JSONL with instruction-following format compatible with:
- Axolotl
- LLaMA-Factory
- Unsloth
- Standard HuggingFace trainers

Usage:
    from src.actions.sleep_processor import SleepProcessor
    
    processor = SleepProcessor('steve')
    
    # Generate training data
    dataset = processor.sleep()
    
    # Or step by step
    processor.consolidate_memories()
    processor.generate_training_pairs()
    processor.export_dataset('steve_lora_data.jsonl')
"""

import json
import yaml
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from src.utils.knowledge_base import KnowledgeBase
from src.utils.chat_utils import get_chat_completion, load_config


@dataclass
class TrainingExample:
    """A single training example for LoRA fine-tuning."""
    instruction: str
    input: str  # Optional context/input
    output: str  # Expected response
    
    # Metadata for filtering/analysis
    source: str = ""  # Where this came from (cognitive_loop, chat, etc.)
    category: str = ""  # Type of knowledge (fact, reasoning, reflection, etc.)
    domain: str = ""  # Subject domain
    quality_score: float = 1.0  # For filtering low-quality examples
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "source": self.source,
            "category": self.category,
            "domain": self.domain,
        }
    
    def to_alpaca_format(self) -> Dict:
        """Convert to Alpaca format (instruction, input, output)."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }
    
    def to_sharegpt_format(self) -> Dict:
        """Convert to ShareGPT format (conversations list)."""
        messages = []
        if self.instruction:
            messages.append({"from": "system", "value": self.instruction})
        if self.input:
            messages.append({"from": "human", "value": self.input})
        if self.output:
            messages.append({"from": "gpt", "value": self.output})
        return {"conversations": messages}


@dataclass 
class MemoryConsolidation:
    """Consolidated memories ready for training data generation."""
    actor_name: str
    
    # Identity
    values: List[str] = field(default_factory=list)
    principles: List[str] = field(default_factory=list)
    
    # Knowledge
    core_knowledge: List[str] = field(default_factory=list)
    supporting_knowledge: List[str] = field(default_factory=list)
    
    # Experience
    experiences: List[str] = field(default_factory=list)
    adjustments: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    
    # Learning sessions
    learning_iterations: List[Dict] = field(default_factory=list)
    study_sessions: List[Dict] = field(default_factory=list)
    reflections: List[Dict] = field(default_factory=list)
    
    # Cognitive loop outputs
    thoughts: List[Dict] = field(default_factory=list)
    enlightenments: List[Dict] = field(default_factory=list)
    mindset_updates: List[str] = field(default_factory=list)
    
    # Chat history
    conversations: List[Dict] = field(default_factory=list)
    
    # Statistics
    total_papers_studied: int = 0
    total_knowledge_items: int = 0
    domains_explored: List[str] = field(default_factory=list)


class SleepProcessor:
    """
    The Sleep Processor - Memory consolidation for creating unique actor minds.
    
    Processes all of an actor's accumulated data into LoRA training format,
    enabling fine-tuning of language models to embody the actor's unique
    personality, knowledge, and reasoning patterns.
    """
    
    def __init__(self, actor_name: str = 'template', verbose: bool = True):
        """
        Initialize the Sleep Processor.
        
        Args:
            actor_name: Name of the actor to process
            verbose: Whether to print progress
        """
        self.actor_name = actor_name
        self.verbose = verbose
        
        # Load config
        config = load_config()
        self.array_dir = Path(config.get('array_dir', 'array'))
        
        # Paths
        self.actor_dir = self.array_dir / 'actors' / actor_name
        self.data_dir = self.array_dir / 'data' / actor_name
        self.output_dir = self.data_dir / 'lora_training'
        
        # Knowledge base
        self.kb = KnowledgeBase(actor_name)
        
        # State
        self.memories: Optional[MemoryConsolidation] = None
        self.training_examples: List[TrainingExample] = []
        
        # Generation settings
        self.min_quality_score = 0.5
        self.include_metadata = True
        
    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def sleep(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point - run the complete sleep cycle.
        
        Like biological sleep, this consolidates memories and prepares
        for "waking up" with improved capabilities.
        
        Returns:
            Summary of the sleep cycle results
        """
        self._log(f"\n🌙 {self.actor_name} is going to sleep...")
        self._log("=" * 60)
        
        # Phase 1: Gather and consolidate memories
        self._log("\n💭 Phase 1: Consolidating memories...")
        self.consolidate_memories()
        
        # Phase 2: Generate training pairs
        self._log("\n📝 Phase 2: Generating training pairs...")
        self.generate_training_pairs()
        
        # Phase 3: Dream (generate synthetic examples)
        self._log("\n💫 Phase 3: Dreaming - generating synthetic examples...")
        self.dream()
        
        # Phase 4: Quality filtering
        self._log("\n🔍 Phase 4: Quality filtering...")
        self._filter_examples()
        
        # Phase 5: Export dataset with timestamp
        self._log("\n💾 Phase 5: Exporting dataset...")
        export_path = self.export_dataset(output_path)  # Will auto-generate timestamped name if None
        
        # Phase 6: Consolidate and condense knowledge
        self._log("\n🗜️ Phase 6: Consolidating and condensing knowledge...")
        self._consolidate_and_archive()
        
        # Summary
        summary = self._generate_summary()
        
        self._log(f"\n✨ {self.actor_name} has completed sleep!")
        self._log(f"   📊 Generated {len(self.training_examples)} training examples")
        self._log(f"   💾 Saved to: {export_path}")

        # Log sleep event and condensed summary to learning_book.json
        self._log_sleep_event(summary)
        
        return summary

    def _consolidate_and_archive(self):
        """
        Consolidate and condense principles, experiences, challenges, and knowledge.
        Archive the originals and update with condensed versions.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_base = self.actor_dir / 'archive'
        archive_base.mkdir(parents=True, exist_ok=True)
        
        # Categories to consolidate
        categories = [
            ('principles', self.memories.principles, self.kb.paths['principles']),
            ('experiences', self.memories.experiences, self.kb.paths['experiences']),
            ('challenges', self.memories.challenges, self.kb.paths['challenges']),
            ('core_knowledge', self.memories.core_knowledge, self.kb.paths['core_knowledge']),
            ('supporting_knowledge', self.memories.supporting_knowledge, self.kb.paths['supporting_knowledge']),
        ]
        
        # Get usage statistics to inform condensation decisions
        usage_stats = self.kb.get_usage_stats()
        
        for category_name, items, file_path in categories:
            if not items or len(items) < 3:  # Skip if too few items
                self._log(f"   ⊘ Skipping {category_name} (only {len(items)} items)")
                continue
            
            # Get usage info for this category
            category_usage = usage_stats.get(category_name, {'access_count': 0, 'items': {}})
            frequently_used = self._identify_frequently_used_items(category_name, items, category_usage)
            
            self._log(f"   🔄 Consolidating {len(items)} {category_name}...")
            if frequently_used:
                self._log(f"      📌 {len(frequently_used)} frequently-used items will be preserved")
            
            # Archive original items
            archive_dir = archive_base / category_name
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_file = archive_dir / f"{category_name}_{timestamp}.yaml"
            
            # Save original to archive
            archive_data = {
                category_name: items,
                '_meta': {
                    'archived_at': datetime.now().isoformat(),
                    'archived_by': 'sleep_processor',
                    'reason': 'Memory consolidation during sleep',
                    'original_count': len(items)
                }
            }
            
            try:
                with open(archive_file, 'w') as f:
                    yaml.dump(archive_data, f, default_flow_style=False, allow_unicode=True)
                self._log(f"      ✓ Archived {len(items)} items to {archive_file.name}")
            except Exception as e:
                self._log(f"      ⚠ Could not archive {category_name}: {e}")
                continue
            
            # Condense using LLM with usage awareness
            condensed_items = self._condense_items(category_name, items, frequently_used)
            
            if condensed_items:
                # Update the main file with condensed items
                new_data = {
                    category_name: condensed_items,
                    '_meta': {
                        'last_updated': datetime.now().isoformat(),
                        'updated_by': 'sleep_processor',
                        'consolidation_info': {
                            'original_count': len(items),
                            'condensed_count': len(condensed_items),
                            'archived_to': str(archive_file),
                            'timestamp': timestamp
                        }
                    }
                }
                
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w') as f:
                        yaml.dump(new_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                    self._log(f"      ✓ Updated with {len(condensed_items)} condensed items")
                except Exception as e:
                    self._log(f"      ⚠ Could not update {category_name}: {e}")
    
    def _identify_frequently_used_items(self, category_name: str, items: List[str], usage: dict) -> List[int]:
        """Identify items that are frequently accessed and should be kept separate."""
        if not usage or 'items' not in usage:
            return []
        
        # Get items with significant usage (accessed in last 7 days or >5 times)
        from datetime import datetime, timedelta
        recent_threshold = (datetime.now() - timedelta(days=7)).isoformat()
        frequently_used = []
        
        for idx_str, stats in usage['items'].items():
            idx = int(idx_str)
            if idx < len(items):
                # Keep if accessed >5 times OR accessed recently
                if stats['count'] > 5 or (stats.get('last_accessed', '') > recent_threshold):
                    frequently_used.append(idx)
        
        return frequently_used
    
    def _condense_items(self, category_name: str, items: List[str], frequently_used: List[int] = None) -> List[str]:
        """
        Use LLM to condense and remove redundancy from a list of items.
        The model decides the optimal number of items to retain.
        Returns condensed list with references to originals.
        """
        if len(items) <= 5:
            return items
        
        # Calculate target size - be less aggressive for better quality
        # Aim for 20-40% reduction depending on size
        if len(items) > 50:
            target_min = int(len(items) * 0.6)   # Keep at least 60%
            target_max = int(len(items) * 0.8)   # Up to 80%
        elif len(items) > 20:
            target_min = int(len(items) * 0.65)  # Keep at least 65%
            target_max = int(len(items) * 0.85)  # Up to 85%
        else:
            target_min = int(len(items) * 0.7)   # Keep at least 70%
            target_max = int(len(items) * 0.9)   # Up to 90%
        
        # Mark frequently-used items
        frequently_used = frequently_used or []
        items_with_markers = []
        for i, item in enumerate(items):
            marker = " 🔥FREQUENTLY_USED" if i in frequently_used else ""
            items_with_markers.append(f"{i+1}. {item}{marker}")
        
        # Prepare prompt with better guidance
        prompt = f"""You are consolidating {category_name} for the AI actor "{self.actor_name}".

Original {category_name} ({len(items)} items):
{chr(10).join(items_with_markers)}

NOTE: Items marked with 🔥FREQUENTLY_USED are actively being accessed and should be kept separate (not merged).

Task: Intelligently condense these items into {target_min}-{target_max} consolidated items by:
1. **Merging closely related concepts** - combine items that overlap significantly
2. **Removing exact duplicates** - eliminate any repeated information
3. **Preserving unique insights** - keep distinct, valuable information separate
4. **Creating clear item references** - prefix each with source item numbers [Items X,Y,Z]
5. **Maintaining specificity** - don't over-generalize; keep technical details

Guidelines:
- Merge 2-4 related items together when they share a common theme
- Keep unique, unrelated items separate (don't force merging of unrelated concepts)
- Aim for {target_min}-{target_max} final items (about 20-40% reduction)
- Each condensed item should be comprehensive but focused

Return ONLY a valid JSON array of strings. Each string must start with item references.
Format: ["[Items 1,3] Combined concept...", "[Items 2,5,7] Merged principle...", "[Item 4] Unique insight..."]

JSON array only, no code blocks, no explanations:"""

        try:
            response = get_chat_completion([
                {"role": "system", "content": f"You are an expert at consolidating knowledge for {self.actor_name}. You merge related concepts while preserving unique insights. Always include item references."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith('```'):
                # Remove markdown code blocks
                lines = response.split('\n')
                response = '\n'.join(line for line in lines if not line.strip().startswith('```'))
                response = response.strip()
            
            # Parse JSON response
            condensed = json.loads(response)
            
            if isinstance(condensed, list) and len(condensed) > 0:
                reduction_pct = int((1 - len(condensed)/len(items)) * 100)
                self._log(f"      ✓ Model condensed {len(items)} → {len(condensed)} items ({reduction_pct}% reduction)")
                
                # Accept if we reduced the count meaningfully (at least 10% reduction)
                if len(condensed) < len(items) * 0.9:
                    # Verify all items have references
                    items_with_refs = [item for item in condensed if '[Item' in item]
                    if len(items_with_refs) == len(condensed):
                        self._log(f"      ✓ All condensed items include source references")
                    else:
                        self._log(f"      ⚠ Warning: {len(condensed) - len(items_with_refs)} items missing references")
                    return condensed
                else:
                    self._log(f"      ⚠ Insufficient reduction ({reduction_pct}%), keeping originals")
                    return items
            else:
                self._log(f"      ⚠ Invalid condensation result, keeping originals")
                return items
                
        except json.JSONDecodeError as e:
            self._log(f"      ⚠ JSON parse failed: {e}")
            self._log(f"      Response preview: {response[:200]}...")
            return items
        except Exception as e:
            self._log(f"      ⚠ Condensation failed: {e}, keeping originals")
            return items

    def _log_sleep_event(self, summary: dict):
        """Append a sleep event and condensed snapshot to learning_book.json."""
        import json
        learning_book_path = self.actor_dir / 'learning_book.json'
        now = datetime.now().isoformat()
        # Load or create learning_book
        if learning_book_path.exists():
            try:
                with open(learning_book_path, 'r') as f:
                    book = json.load(f)
            except Exception:
                book = {}
        else:
            book = {}

        # Prepare sleep event
        sleep_event = {
            "type": "sleep",
            "timestamp": now,
            "summary": summary,
            "message": f"Sleep cycle completed. {summary.get('total_examples', 0)} training examples generated.",
        }

        # Add condensed snapshot (values, principles, knowledge, experiences)
        snapshot = {
            "values": self.memories.values,
            "principles": self.memories.principles,
            "core_knowledge": self.memories.core_knowledge[:10],
            "experiences": self.memories.experiences[:10],
            "challenges": self.memories.challenges[:5],
            "total_examples": summary.get('total_examples', 0),
            "total_knowledge_items": summary.get('memory_stats', {}).get('total_knowledge_items', 0),
        }
        sleep_event["snapshot"] = snapshot

        # Append to learning_history (or create)
        if "learning_history" not in book:
            book["learning_history"] = []
        book["learning_history"].append(sleep_event)

        # Optionally, update a summary field
        book["last_sleep"] = {
            "timestamp": now,
            "summary": summary,
            "snapshot": snapshot
        }

        # Save back
        try:
            with open(learning_book_path, 'w') as f:
                json.dump(book, f, indent=2, ensure_ascii=False)
            self._log(f"   📝 Sleep event logged in learning_book.json")
        except Exception as e:
            self._log(f"   ⚠ Could not log sleep event: {e}")
    
    def consolidate_memories(self) -> MemoryConsolidation:
        """
        Phase 1: Gather all actor data into consolidated memories.
        """
        self.memories = MemoryConsolidation(actor_name=self.actor_name)
        
        # Load identity from knowledge base
        self.memories.values = self.kb.get_values()
        self.memories.principles = self.kb.get_principles()
        self._log(f"   ✓ Loaded {len(self.memories.values)} values, {len(self.memories.principles)} principles")
        
        # Load knowledge
        self.memories.core_knowledge = self.kb.get_core_knowledge()
        self.memories.supporting_knowledge = self.kb.get_supporting_knowledge()
        self._log(f"   ✓ Loaded {len(self.memories.core_knowledge)} core, {len(self.memories.supporting_knowledge)} supporting knowledge items")
        
        # Load experiences
        self.memories.experiences = self.kb.get_experiences()
        self.memories.adjustments = self.kb.get_adjustments()
        self.memories.challenges = self.kb.get_challenges()
        self._log(f"   ✓ Loaded {len(self.memories.experiences)} experiences, {len(self.memories.challenges)} challenges")
        
        # Load learning book
        self._load_learning_book()
        
        # Load cognitive loops
        self._load_cognitive_loops()
        
        # Load learning sessions (LoRA data)
        self._load_learning_sessions()
        
        # Load chat history if available
        self._load_chat_history()
        
        # Calculate statistics
        self.memories.total_knowledge_items = (
            len(self.memories.core_knowledge) + 
            len(self.memories.supporting_knowledge) +
            len(self.memories.experiences)
        )
        
        return self.memories
    
    def _load_learning_book(self):
        """Load learning book data."""
        learning_book_path = self.actor_dir / 'learning_book.json'
        if learning_book_path.exists():
            try:
                with open(learning_book_path, 'r') as f:
                    data = json.load(f)
                
                for iteration in data.get('iterations', []):
                    self.memories.learning_iterations.append(iteration)
                    
                    # Extract thoughts
                    if 'thoughts' in iteration:
                        for thought in iteration['thoughts']:
                            self.memories.thoughts.append({
                                'topic': iteration.get('topic', ''),
                                'thought': thought,
                                'iteration': iteration.get('iteration', 0)
                            })
                    
                    # Extract mindset updates
                    if 'mindset_update' in iteration:
                        self.memories.mindset_updates.append(iteration['mindset_update'])
                
                self._log(f"   ✓ Loaded {len(self.memories.learning_iterations)} learning iterations")
                
            except Exception as e:
                self._log(f"   ⚠ Error loading learning book: {e}")
    
    def _load_cognitive_loops(self):
        """Load cognitive loop data."""
        loops_dir = self.data_dir / 'cognitive_loops'
        if loops_dir.exists():
            loop_files = list(loops_dir.glob('loop_*.json'))
            
            for loop_file in loop_files:
                try:
                    with open(loop_file, 'r') as f:
                        loop_data = json.load(f)
                    
                    # Extract study notes
                    for note in loop_data.get('study_notes', []):
                        self.memories.study_sessions.append(note)
                    
                    # Extract enlightenments
                    for enlightenment in loop_data.get('enlightenments', []):
                        self.memories.enlightenments.append(enlightenment)
                    
                    # Extract reflections
                    if 'synthesis' in loop_data:
                        self.memories.reflections.append({
                            'topic': loop_data.get('topic', ''),
                            'synthesis': loop_data['synthesis'],
                            'conclusions': loop_data.get('conclusions', [])
                        })
                    
                    # Track domains
                    topic = loop_data.get('topic', '')
                    if topic and topic not in self.memories.domains_explored:
                        self.memories.domains_explored.append(topic)
                    
                    # Count papers
                    self.memories.total_papers_studied += len(loop_data.get('studied_papers', []))
                    
                except Exception as e:
                    self._log(f"   ⚠ Error loading {loop_file.name}: {e}")
            
            self._log(f"   ✓ Loaded {len(loop_files)} cognitive loops, {self.memories.total_papers_studied} papers studied")
    
    def _load_learning_sessions(self):
        """Load existing learning session data (already in training format)."""
        sessions_dir = self.data_dir / 'learning_sessions'
        if sessions_dir.exists():
            session_files = list(sessions_dir.glob('session_*.json'))
            
            for session_file in session_files:
                try:
                    with open(session_file, 'r') as f:
                        session = json.load(f)
                    
                    # These are already in conversation format
                    if 'conversations' in session:
                        self.memories.conversations.extend(session['conversations'])
                    
                except Exception as e:
                    self._log(f"   ⚠ Error loading session {session_file.name}: {e}")
            
            self._log(f"   ✓ Loaded {len(session_files)} learning sessions")
    
    def _load_chat_history(self):
        """Load chat history if available."""
        chat_dir = self.data_dir / 'chat_history'
        if chat_dir.exists():
            chat_files = list(chat_dir.glob('*.json'))
            
            for chat_file in chat_files:
                try:
                    with open(chat_file, 'r') as f:
                        chat_data = json.load(f)
                    
                    if isinstance(chat_data, list):
                        self.memories.conversations.extend(chat_data)
                    elif 'messages' in chat_data:
                        self.memories.conversations.extend(chat_data['messages'])
                    
                except Exception as e:
                    self._log(f"   ⚠ Error loading chat {chat_file.name}: {e}")
            
            if chat_files:
                self._log(f"   ✓ Loaded {len(chat_files)} chat history files")
    
    def generate_training_pairs(self) -> List[TrainingExample]:
        """
        Phase 2: Generate training pairs from consolidated memories.
        
        Creates diverse training examples that capture:
        - Identity expression (values, principles)
        - Knowledge retrieval and explanation
        - Reasoning patterns
        - Experience-based advice
        - Reflection and synthesis
        """
        if not self.memories:
            self.consolidate_memories()
        
        self.training_examples = []
        
        # Generate identity examples
        self._generate_identity_examples()
        
        # Generate knowledge examples
        self._generate_knowledge_examples()
        
        # Generate experience examples
        self._generate_experience_examples()
        
        # Generate reasoning examples from study sessions
        self._generate_reasoning_examples()
        
        # Generate reflection examples
        self._generate_reflection_examples()
        
        # Generate from existing conversations
        self._generate_conversation_examples()
        
        # Generate synthesis examples
        self._generate_synthesis_examples()
        
        self._log(f"   ✓ Generated {len(self.training_examples)} training examples")
        
        return self.training_examples
    
    def _generate_identity_examples(self):
        """Generate examples that express the actor's identity."""
        
        # Values-based examples
        if self.memories.values:
            values_str = "\n".join(f"- {v}" for v in self.memories.values)
            
            self.training_examples.append(TrainingExample(
                instruction=f"You are {self.actor_name}, an AI learning agent. What are your core values?",
                input="",
                output=f"My core values guide everything I do:\n\n{values_str}\n\nThese values shape how I approach learning, interact with others, and make decisions.",
                source="identity",
                category="values",
            ))
            
            # Individual value explanations
            for value in self.memories.values[:5]:  # Limit to prevent too many similar examples
                self.training_examples.append(TrainingExample(
                    instruction=f"You are {self.actor_name}. How does the value '{value}' influence your approach?",
                    input="",
                    output=f"The value of {value} is fundamental to who I am. It influences how I process information, form conclusions, and engage with new ideas. When I encounter challenging concepts or conflicting information, I return to this value as a guiding principle.",
                    source="identity",
                    category="values",
                ))
        
        # Principles-based examples
        if self.memories.principles:
            principles_str = "\n".join(f"- {p}" for p in self.memories.principles)
            
            self.training_examples.append(TrainingExample(
                instruction=f"You are {self.actor_name}. What principles guide your reasoning?",
                input="",
                output=f"I operate according to these guiding principles:\n\n{principles_str}\n\nThese principles help me maintain consistency and rigor in my thinking.",
                source="identity",
                category="principles",
            ))
    
    def _generate_knowledge_examples(self):
        """Generate examples from knowledge base."""
        
        # Core knowledge Q&A
        for i, knowledge in enumerate(self.memories.core_knowledge[:50]):  # Limit examples
            # Generate a question about this knowledge
            self.training_examples.append(TrainingExample(
                instruction=f"You are {self.actor_name}, a knowledgeable AI agent. Explain the following concept based on your understanding.",
                input=f"Concept: {knowledge[:100]}..." if len(knowledge) > 100 else f"Concept: {knowledge}",
                output=knowledge,
                source="knowledge_base",
                category="core_knowledge",
            ))
        
        # Supporting knowledge with context
        for knowledge in self.memories.supporting_knowledge[:30]:
            self.training_examples.append(TrainingExample(
                instruction=f"You are {self.actor_name}. Share your knowledge on this topic.",
                input=knowledge[:50] + "...",
                output=knowledge,
                source="knowledge_base",
                category="supporting_knowledge",
            ))
    
    def _generate_experience_examples(self):
        """Generate examples from experiences."""
        
        for experience in self.memories.experiences[:30]:
            self.training_examples.append(TrainingExample(
                instruction=f"You are {self.actor_name}. Based on your learning experience, what insight can you share?",
                input="",
                output=experience,
                source="experience",
                category="experiential_knowledge",
            ))
        
        # Challenge-based examples (things the actor is still thinking about)
        for challenge in self.memories.challenges[:20]:
            self.training_examples.append(TrainingExample(
                instruction=f"You are {self.actor_name}. What are you currently uncertain about or still exploring?",
                input="",
                output=f"I'm still exploring and thinking deeply about: {challenge}\n\nThis represents an area where I'm actively seeking to deepen my understanding.",
                source="challenges",
                category="epistemic_humility",
            ))
    
    def _generate_reasoning_examples(self):
        """Generate examples from study sessions showing reasoning."""
        
        for session in self.memories.study_sessions[:40]:
            if isinstance(session, dict):
                paper_title = session.get('title', session.get('paper', 'a research paper'))
                insights = session.get('insights', session.get('key_points', []))
                
                if insights:
                    insights_str = "\n".join(f"- {i}" for i in insights[:5])
                    
                    self.training_examples.append(TrainingExample(
                        instruction=f"You are {self.actor_name}. After studying a paper, share your key insights.",
                        input=f"I've been studying: {paper_title}",
                        output=f"From studying this paper, I've extracted these key insights:\n\n{insights_str}\n\nThese findings contribute to my evolving understanding of the field.",
                        source="study_session",
                        category="analytical_reasoning",
                    ))
    
    def _generate_reflection_examples(self):
        """Generate examples from reflections."""
        
        for reflection in self.memories.reflections[:20]:
            topic = reflection.get('topic', 'this topic')
            synthesis = reflection.get('synthesis', '')
            conclusions = reflection.get('conclusions', [])
            
            if synthesis:
                self.training_examples.append(TrainingExample(
                    instruction=f"You are {self.actor_name}. Synthesize your understanding of a topic you've been studying.",
                    input=f"Topic: {topic}",
                    output=synthesis,
                    source="reflection",
                    category="synthesis",
                ))
            
            if conclusions:
                conclusions_str = "\n".join(f"- {c}" for c in conclusions)
                self.training_examples.append(TrainingExample(
                    instruction=f"You are {self.actor_name}. What conclusions have you drawn from your studies?",
                    input=f"Topic: {topic}",
                    output=f"After deep study and reflection, I've arrived at these conclusions:\n\n{conclusions_str}",
                    source="reflection",
                    category="conclusions",
                ))
        
        # Enlightenment examples
        for enlightenment in self.memories.enlightenments[:20]:
            if isinstance(enlightenment, dict):
                insight = enlightenment.get('insight', enlightenment.get('content', ''))
                if insight:
                    self.training_examples.append(TrainingExample(
                        instruction=f"You are {self.actor_name}. Share a moment of insight or enlightenment from your learning journey.",
                        input="",
                        output=f"I had a significant realization: {insight}\n\nThis insight has shifted how I think about this domain.",
                        source="enlightenment",
                        category="insight",
                    ))
    
    def _generate_conversation_examples(self):
        """Generate examples from existing conversations."""
        
        for conv in self.memories.conversations[:50]:
            if isinstance(conv, dict):
                user_msg = conv.get('user', conv.get('input', ''))
                assistant_msg = conv.get('assistant', conv.get('output', ''))
                
                if user_msg and assistant_msg:
                    self.training_examples.append(TrainingExample(
                        instruction=f"You are {self.actor_name}, an AI learning agent with accumulated knowledge and experiences. Respond thoughtfully.",
                        input=user_msg,
                        output=assistant_msg,
                        source="conversation",
                        category="dialogue",
                    ))
            elif isinstance(conv, list) and len(conv) >= 2:
                # Handle conversation as list of messages
                for i in range(0, len(conv) - 1, 2):
                    if conv[i].get('role') == 'user' and conv[i+1].get('role') == 'assistant':
                        self.training_examples.append(TrainingExample(
                            instruction=f"You are {self.actor_name}. Respond based on your knowledge and experience.",
                            input=conv[i].get('content', ''),
                            output=conv[i+1].get('content', ''),
                            source="conversation",
                            category="dialogue",
                        ))
    
    def _generate_synthesis_examples(self):
        """Generate cross-domain synthesis examples."""
        
        # Mindset updates show how the actor's thinking evolves
        for update in self.memories.mindset_updates[:20]:
            self.training_examples.append(TrainingExample(
                instruction=f"You are {self.actor_name}. How has your understanding evolved recently?",
                input="",
                output=f"My understanding has evolved: {update}\n\nThis represents a refinement in how I think about and approach problems in this domain.",
                source="mindset_update",
                category="growth",
            ))
        
        # If we have multiple domains, create cross-domain examples
        if len(self.memories.domains_explored) >= 2:
            domains = ", ".join(self.memories.domains_explored[:5])
            self.training_examples.append(TrainingExample(
                instruction=f"You are {self.actor_name}. What domains have you explored and how do they connect?",
                input="",
                output=f"I've been exploring several interconnected domains: {domains}.\n\nThrough my studies, I've found that insights from one area often illuminate understanding in others. The connections between these fields reveal deeper patterns and principles.",
                source="synthesis",
                category="cross_domain",
            ))
    
    def _filter_examples(self):
        """Phase 3: Filter low-quality examples."""
        original_count = len(self.training_examples)
        
        # Remove duplicates based on content hash
        seen_hashes = set()
        unique_examples = []
        
        for example in self.training_examples:
            content_hash = hashlib.md5(
                (example.instruction + example.input + example.output).encode()
            ).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_examples.append(example)
        
        # Filter out very short outputs
        filtered = [ex for ex in unique_examples if len(ex.output) > 50]
        
        self.training_examples = filtered
        
        removed = original_count - len(self.training_examples)
        if removed > 0:
            self._log(f"   ✓ Removed {removed} duplicate/low-quality examples")
    
    def export_dataset(
        self, 
        output_path: str = None,
        format: str = "alpaca",
        session_id: str = None
    ) -> str:
        """
        Phase 4: Export training dataset to file with timestamp.
        
        Args:
            output_path: Path to save the dataset (if None, auto-generates timestamped name)
            format: Output format - 'alpaca', 'sharegpt', or 'full'
            session_id: Optional session ID to link this dataset to a learning session
        
        Returns:
            Path to the saved file
        """
        # Generate timestamped filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.actor_name}_lora_dataset_{timestamp}.jsonl"
            output_path = str(self.output_dir / filename)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert examples to chosen format
        with open(output_file, 'w') as f:
            for example in self.training_examples:
                if format == "alpaca":
                    data = example.to_alpaca_format()
                elif format == "sharegpt":
                    data = example.to_sharegpt_format()
                else:  # full format with metadata
                    data = example.to_dict()
                
                f.write(json.dumps(data) + '\n')
        
        # Also save a summary file
        summary_path = output_file.with_suffix('.summary.json')
        summary = self._generate_summary()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(output_file)
    
    def _generate_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Generate a summary of the sleep cycle."""
        
        category_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        for example in self.training_examples:
            category_counts[example.category] += 1
            source_counts[example.source] += 1
        
        return {
            "actor_name": self.actor_name,
            "generated_at": datetime.now().isoformat(),
            "session_id": session_id,
            "total_examples": len(self.training_examples),
            "by_category": dict(category_counts),
            "by_source": dict(source_counts),
            "memory_stats": {
                "values": len(self.memories.values) if self.memories else 0,
                "principles": len(self.memories.principles) if self.memories else 0,
                "core_knowledge": len(self.memories.core_knowledge) if self.memories else 0,
                "experiences": len(self.memories.experiences) if self.memories else 0,
                "domains_explored": self.memories.domains_explored if self.memories else [],
                "total_papers_studied": self.memories.total_papers_studied if self.memories else 0,
            },
            "recommended_training_params": {
                "epochs": 3 if len(self.training_examples) > 500 else 5,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        }
    
    def dream(self) -> List[TrainingExample]:
        """
        Generate additional synthetic training examples through "dreaming".
        
        Uses the LLM to create variations and expansions of existing knowledge,
        similar to how dreams help consolidate and extend learning.
        """
        self._log("\n💫 Entering dream state - generating synthetic examples...")
        
        if not self.memories:
            self.consolidate_memories()
        
        dream_examples = []
        
        # Dream about connections between knowledge items
        if len(self.memories.core_knowledge) >= 2:
            sample_knowledge = self.memories.core_knowledge[:5]
            prompt = f"""You are {self.actor_name}, an AI agent with the following knowledge:

{chr(10).join('- ' + k for k in sample_knowledge)}

Generate 3 insightful questions someone might ask about these topics, and provide thoughtful answers that demonstrate deep understanding. Format as JSON array with 'question' and 'answer' keys."""
            
            try:
                response = get_chat_completion([
                    {"role": "system", "content": f"You are {self.actor_name}, embodying their values and knowledge."},
                    {"role": "user", "content": prompt}
                ], temperature=0.7)
                
                # Parse and add examples
                qa_pairs = json.loads(response)
                for qa in qa_pairs:
                    dream_examples.append(TrainingExample(
                        instruction=f"You are {self.actor_name}. Answer based on your accumulated knowledge.",
                        input=qa.get('question', ''),
                        output=qa.get('answer', ''),
                        source="dream",
                        category="synthetic",
                    ))
            except Exception as e:
                self._log(f"   ⚠ Dream generation error: {e}")
        
        self.training_examples.extend(dream_examples)
        self._log(f"   ✓ Generated {len(dream_examples)} dream examples")
        
        return dream_examples


# Convenience functions
def sleep(actor_name: str = 'template', output_path: Optional[str] = None) -> Dict[str, Any]:
    """Put an actor to sleep and generate LoRA training data."""
    processor = SleepProcessor(actor_name)
    return processor.sleep(output_path)


def quick_sleep(actor_name: str = 'template') -> str:
    """Quick sleep - returns path to generated dataset."""
    processor = SleepProcessor(actor_name, verbose=False)
    result = processor.sleep()
    return result.get('output_path', '')


def merge_datasets(
    actor_name: str,
    dataset_paths: List[str],
    output_name: str = None
) -> Dict[str, Any]:
    """
    Merge multiple LoRA datasets from different learning sessions.
    
    Args:
        actor_name: Name of the actor
        dataset_paths: List of paths to .jsonl dataset files to merge
        output_name: Optional output filename (auto-generates timestamped name if None)
    
    Returns:
        Summary of the merged dataset
    """
    from src.utils.knowledge_base import get_array_dir
    
    array_dir = get_array_dir()
    lora_dir = array_dir / 'data' / actor_name / 'lora_training'
    lora_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all examples from source datasets
    all_examples = []
    source_summaries = []
    
    for dataset_path in dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            continue
        
        # Load examples
        with open(path, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    all_examples.append(example)
                except json.JSONDecodeError:
                    continue
        
        # Load summary if exists
        summary_path = path.with_suffix('.summary.json')
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                source_summaries.append({
                    'filename': path.name,
                    'generated_at': summary.get('generated_at'),
                    'examples': summary.get('total_examples', 0)
                })
    
    # Remove duplicates based on instruction+output combination
    seen = set()
    unique_examples = []
    for example in all_examples:
        key = hashlib.md5(
            (example.get('instruction', '') + example.get('output', '')).encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique_examples.append(example)
    
    # Generate output filename
    if output_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"{actor_name}_merged_dataset_{timestamp}.jsonl"
    
    output_path = lora_dir / output_name
    
    # Write merged dataset
    with open(output_path, 'w') as f:
        for example in unique_examples:
            f.write(json.dumps(example) + '\n')
    
    # Create merged summary
    merged_summary = {
        'actor_name': actor_name,
        'generated_at': datetime.now().isoformat(),
        'type': 'merged',
        'total_examples': len(unique_examples),
        'source_datasets': source_summaries,
        'merged_from': [Path(p).name for p in dataset_paths],
        'duplicates_removed': len(all_examples) - len(unique_examples)
    }
    
    # Save summary
    summary_path = output_path.with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(merged_summary, f, indent=2)
    
    return {
        'success': True,
        'output_path': str(output_path),
        'total_examples': len(unique_examples),
        'sources_merged': len(dataset_paths),
        'duplicates_removed': len(all_examples) - len(unique_examples),
        'summary': merged_summary
    }


if __name__ == "__main__":
    import sys
    
    actor = sys.argv[1] if len(sys.argv) > 1 else 'steve'
    
    print(f"\n🌙 Initiating sleep cycle for {actor}...")
    processor = SleepProcessor(actor)
    result = processor.sleep()
    
    print(f"\n📊 Sleep Summary:")
    print(json.dumps(result, indent=2))
