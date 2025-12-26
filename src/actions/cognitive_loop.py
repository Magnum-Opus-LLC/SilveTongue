"""
Cognitive Loop - The Manifesto

A unified cognitive architecture that orchestrates Silvertongue's complete learning cycle:

    ┌──────────────────────────────────────────────────────────────────┐
    │                    THE COGNITIVE LOOP                             │
    │                                                                   │
    │   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐   │
    │   │ SEARCH  │ ──► │  STUDY  │ ──► │  THINK  │ ──► │ REFLECT │   │
    │   └─────────┘     └─────────┘     └─────────┘     └─────────┘   │
    │        ▲                                               │         │
    │        │                                               ▼         │
    │   ┌─────────┐                                    ┌─────────┐     │
    │   │ EXPAND  │ ◄────────────────────────────────  │ UPDATE  │     │
    │   └─────────┘                                    └─────────┘     │
    │                                                                   │
    └──────────────────────────────────────────────────────────────────┘

PHASES:
    1. SEARCH  - Discover relevant papers, articles, and knowledge sources
    2. STUDY   - Deep dive into materials, extract knowledge bits
    3. THINK   - Process and synthesize information through conscious reasoning
    4. REFLECT - Evaluate enlightenment: confirms, expands, challenges beliefs
    5. UPDATE  - Integrate new knowledge, update mindset and principles
    6. EXPAND  - Identify knowledge gaps, generate new search queries

This creates an autonomous learning agent that:
- Seeks knowledge proactively based on curiosity and goals
- Studies materials systematically with structured extraction
- Thinks critically about new information
- Reflects on how it changes existing beliefs
- Updates its knowledge base and mindset
- Expands its search based on emerging questions
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

from src.utils.knowledge_base import KnowledgeBase
from src.utils.chat_utils import get_chat_completion
from src.utils.research import ResearchManager
from src.utils.learning_session import LearningSessionManager
from src.thoughts.conscious import SilvertongueConscious
from src.thoughts.subconscious import SilvertongueSubconscious
from src.actions.study_processor import StudyProcessor
from src.actions.learning_scheduler import LearningScheduleRunner
from src.scholar_search import ScholarSearcher, ScholarPaper


class LoopPhase(Enum):
    """Phases of the cognitive loop."""
    SEARCH = "search"
    STUDY = "study"
    THINK = "think"
    REFLECT = "reflect"
    UPDATE = "update"
    EXPAND = "expand"
    COMPLETE = "complete"
    PAUSED = "paused"


@dataclass
class CognitiveState:
    """Represents the current state of the cognitive loop."""
    loop_id: str
    phase: LoopPhase
    iteration: int
    topic: str
    goal: str
    
    # Search state
    search_queries: List[str] = field(default_factory=list)
    discovered_papers: List[Dict] = field(default_factory=list)
    selected_papers: List[Dict] = field(default_factory=list)
    
    # Study state
    studied_papers: List[Dict] = field(default_factory=list)
    knowledge_bits: List[Dict] = field(default_factory=list)
    study_notes: List[Dict] = field(default_factory=list)
    
    # Think state
    thoughts: List[Dict] = field(default_factory=list)
    synthesis: str = ""
    conclusions: List[str] = field(default_factory=list)
    
    # Reflect state
    enlightenments: List[Dict] = field(default_factory=list)
    confirmations: List[Dict] = field(default_factory=list)
    expansions: List[Dict] = field(default_factory=list)
    challenges: List[Dict] = field(default_factory=list)
    revolutionary_ideas: List[Dict] = field(default_factory=list)
    
    # Update state
    mindset_updates: List[str] = field(default_factory=list)
    knowledge_evolved: List[Dict] = field(default_factory=list)
    
    # Expand state  
    emerging_questions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    next_search_queries: List[str] = field(default_factory=list)
    
    # Meta
    started_at: str = ""
    updated_at: str = ""
    total_papers_studied: int = 0
    total_knowledge_bits: int = 0
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['phase'] = self.phase.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CognitiveState':
        data['phase'] = LoopPhase(data['phase'])
        return cls(**data)


# Prompts for cognitive processing

SEARCH_STRATEGY_PROMPT = """You are planning a research strategy for learning about a topic.

TOPIC: {topic}
GOAL: {goal}

CURRENT KNOWLEDGE:
{current_knowledge}

PREVIOUS SEARCHES (if any):
{previous_searches}

Generate search queries that will help achieve the learning goal.
Consider:
1. Core concepts that need understanding
2. Recent advances and state-of-the-art
3. Foundational papers and seminal works
4. Practical applications and case studies
5. Gaps in current knowledge

Return JSON:
{{
    "reasoning": "Your strategy and reasoning",
    "queries": [
        {{
            "query": "search query string",
            "purpose": "what this query aims to find",
            "priority": "high|medium|low"
        }}
    ],
    "focus_areas": ["area1", "area2"]
}}
"""

PAPER_SELECTION_PROMPT = """You are selecting papers for deep study from search results.

TOPIC: {topic}
GOAL: {goal}
FOCUS AREAS: {focus_areas}

DISCOVERED PAPERS:
{papers}

Select the most valuable papers for achieving the learning goal.
Consider:
1. Citation count (impact and validation)
2. Relevance to focus areas
3. Recency vs foundational value
4. Diversity of perspectives
5. Practical applicability

Return JSON:
{{
    "selected": [
        {{
            "title": "paper title",
            "reason": "why selected",
            "expected_value": "what you expect to learn",
            "priority": 1-5
        }}
    ],
    "skipped_reasons": ["why some papers were not selected"]
}}
"""

SYNTHESIS_PROMPT = """You are synthesizing knowledge from studied materials.

TOPIC: {topic}
GOAL: {goal}

KNOWLEDGE BITS EXTRACTED:
{knowledge_bits}

STUDY NOTES:
{study_notes}

Synthesize this information into coherent understanding:
1. What are the key themes and patterns?
2. What are the main findings and contributions?
3. How do different sources relate to each other?
4. What practical insights emerge?
5. What remains unclear or contradictory?

Return JSON:
{{
    "synthesis": "comprehensive synthesis paragraph",
    "key_themes": ["theme1", "theme2"],
    "main_findings": ["finding1", "finding2"],
    "connections": ["how concepts relate"],
    "practical_insights": ["actionable insights"],
    "uncertainties": ["what remains unclear"],
    "conclusions": ["key conclusions"]
}}
"""

REFLECTION_PROMPT = """You are reflecting on new knowledge against existing beliefs.

TOPIC: {topic}
NEW SYNTHESIS:
{synthesis}

NEW CONCLUSIONS:
{conclusions}

CURRENT BELIEFS AND KNOWLEDGE:
Principles: {principles}
Core Knowledge: {core_knowledge}
Experiences: {experiences}
Current Mindset: {mindset}

Reflect deeply on how this new knowledge relates to existing understanding:
1. What does it CONFIRM that you already believed?
2. What does it EXPAND or add to your understanding?
3. What does it CHALLENGE or contradict?
4. What REVOLUTIONARY new paradigms does it introduce?

Be intellectually honest. Not all new information is valuable.

Return JSON:
{{
    "overall_assessment": "high_value|moderate_value|low_value",
    "intellectual_growth": "description of how this advances understanding",
    "confirmations": [
        {{
            "existing_belief": "what was confirmed",
            "evidence": "how it was confirmed"
        }}
    ],
    "expansions": [
        {{
            "existing_understanding": "what you knew",
            "new_perspective": "what's added",
            "integration": "how to integrate"
        }}
    ],
    "challenges": [
        {{
            "challenged_belief": "what's questioned",
            "challenge": "the counter-argument",
            "severity": "minor|moderate|fundamental",
            "resolution_needed": "what to investigate"
        }}
    ],
    "revolutionary": [
        {{
            "new_paradigm": "the new idea",
            "implications": "what it means",
            "adoption": "adopt|investigate|monitor"
        }}
    ]
}}
"""

UPDATE_DECISION_PROMPT = """You are deciding how to update your knowledge base based on reflection.

REFLECTION RESULTS:
{reflection}

CONFIRMATIONS: {confirmations}
EXPANSIONS: {expansions}
CHALLENGES: {challenges}
REVOLUTIONARY IDEAS: {revolutionary}

Decide what updates to make. These will be PERSISTED to the actor's knowledge files.

Knowledge Types:
- "principle": Fundamental truths, laws, or guidelines (e.g., "First Law of Thermodynamics")
- "experience": Practical insights, observations, case studies learned
- "knowledge": Core facts, methods, techniques, definitions
- "challenge": Open questions, unresolved issues, areas needing investigation

Guidelines:
1. Be selective - only add genuinely valuable insights
2. Be specific - vague statements are not useful
3. Be concise - each item should be 1-2 sentences
4. Include source attribution when relevant

Return JSON:
{{
    "mindset_updates": [
        "specific mindset change to make"
    ],
    "knowledge_to_add": [
        {{
            "type": "principle|experience|knowledge|challenge",
            "content": "clear, specific statement to add",
            "source": "paper/topic that provided this insight"
        }}
    ],
    "beliefs_to_investigate": [
        {{
            "belief": "what to investigate",
            "reason": "why it needs investigation"
        }}
    ],
    "update_summary": "1-2 sentence summary of what was learned and updated"
}}

Be conservative. Quality over quantity. Only include items that provide genuine value.
"""

EXPANSION_PROMPT = """You are identifying knowledge gaps and planning further exploration.

TOPIC: {topic}
GOAL: {goal}

CURRENT STATE:
- Papers studied: {papers_studied}
- Knowledge bits extracted: {knowledge_bits_count}
- Conclusions reached: {conclusions}
- Uncertainties identified: {uncertainties}

EMERGING QUESTIONS FROM STUDY:
{emerging_questions}

CHALLENGES REQUIRING INVESTIGATION:
{challenges}

Identify what to explore next:
1. What fundamental questions remain unanswered?
2. What knowledge gaps need filling?
3. What contradictions need resolution?
4. What practical applications need exploration?
5. When is the learning goal sufficiently achieved?

Return JSON:
{{
    "goal_progress": "0-100 percent toward goal",
    "continue_learning": true/false,
    "reasoning": "why continue or stop",
    "knowledge_gaps": [
        {{
            "gap": "description of gap",
            "importance": "high|medium|low"
        }}
    ],
    "next_queries": [
        {{
            "query": "search query",
            "purpose": "what to find",
            "priority": "high|medium|low"
        }}
    ],
    "recommended_depth": "broad|focused|deep"
}}
"""


class CognitiveLoop:
    """
    The Cognitive Loop - Silvertongue's autonomous learning engine.
    
    Orchestrates the complete learning cycle:
    Search → Study → Think → Reflect → Update → Expand → (repeat)
    
    Usage:
        loop = CognitiveLoop('student')
        
        # Start a learning journey
        result = loop.learn(
            topic="machine learning in chemical engineering",
            goal="Understand how ML is transforming process optimization",
            max_iterations=3
        )
        
        # Or run step by step
        loop.start("quantum computing", "Understand basics of quantum algorithms")
        loop.step()  # Search
        loop.step()  # Study
        loop.step()  # Think
        loop.step()  # Reflect
        loop.step()  # Update
        loop.step()  # Expand
        result = loop.get_state()
    """
    
    def __init__(self, actor_name: str = 'template', verbose: bool = True):
        """
        Initialize the Cognitive Loop.
        
        Args:
            actor_name: Name of the actor whose knowledge base to use
            verbose: Whether to print progress
        """
        self.actor_name = actor_name
        self.verbose = verbose
        
        # Initialize components
        self.kb = KnowledgeBase(actor_name)
        self.conscious = SilvertongueConscious(self.kb)
        self.subconscious = SilvertongueSubconscious(self.kb)
        self.study_processor = StudyProcessor(actor_name)
        self.schedule_runner = LearningScheduleRunner(actor_name)
        self.scholar = ScholarSearcher()
        self.researcher = ResearchManager()
        
        # Learning session manager for LoRA training data
        self.session_manager = LearningSessionManager(actor_name)
        
        # State
        self.state: Optional[CognitiveState] = None
        self.history: List[Dict] = []
        
        # Configuration
        self.config = {
            'max_papers_per_search': 20,
            'max_papers_to_study': 5,
            'search_depth': 2,
            'min_citations': 0,
            'year_from': None,
            'year_to': None,
            'auto_save': True,
            'save_session': True,  # Save learning session for LoRA
            'model_override': None  # LoRA adapter name to use (None = base model)
        }
        
        # Paths - use array_dir from config for data storage
        self.project_root = Path(__file__).parent.parent.parent
        try:
            from src.utils.chat_utils import load_config
            app_config = load_config()
            array_dir = Path(app_config.get('array_dir', ''))
            self.loops_dir = array_dir / 'data' / actor_name / 'cognitive_loops'
        except Exception:
            # Fallback to project dir
            self.loops_dir = self.project_root / 'data' / actor_name / 'cognitive_loops'
        self.loops_dir.mkdir(parents=True, exist_ok=True)
    
    def configure(self, **kwargs):
        """Update configuration."""
        self.config.update(kwargs)
        return self
    
    def start(self, topic: str, goal: str) -> CognitiveState:
        """
        Start a new cognitive loop.
        
        Args:
            topic: The subject to learn about
            goal: The learning objective
            
        Returns:
            Initial cognitive state
        """
        self.state = CognitiveState(
            loop_id=str(uuid.uuid4())[:8],
            phase=LoopPhase.SEARCH,
            iteration=1,
            topic=topic,
            goal=goal,
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self._log(f"🧠 Starting Cognitive Loop: {topic}")
        self._log(f"   Goal: {goal}")
        self._log(f"   Loop ID: {self.state.loop_id}")
        
        # Start learning session for LoRA training data
        if self.config.get('save_session', True):
            self.session_manager.start_session(
                topic=topic,
                goal=goal,
                session_id=self.state.loop_id
            )
            self._log(f"   📝 Learning session started for LoRA data")
        
        self._record_history("start", {"topic": topic, "goal": goal})
        
        if self.config['auto_save']:
            self._save_state()
        
        return self.state
    
    def step(self) -> CognitiveState:
        """
        Execute one phase of the cognitive loop.
        
        Returns:
            Updated cognitive state
        """
        if not self.state:
            raise ValueError("No active loop. Call start() first.")
        
        if self.state.phase == LoopPhase.COMPLETE:
            self._log("✅ Loop already complete")
            return self.state
        
        phase = self.state.phase
        self._log(f"\n{'='*60}")
        self._log(f"📍 Phase: {phase.value.upper()} (Iteration {self.state.iteration})")
        self._log('='*60)
        
        # Execute phase
        if phase == LoopPhase.SEARCH:
            self._phase_search()
        elif phase == LoopPhase.STUDY:
            self._phase_study()
        elif phase == LoopPhase.THINK:
            self._phase_think()
        elif phase == LoopPhase.REFLECT:
            self._phase_reflect()
        elif phase == LoopPhase.UPDATE:
            self._phase_update()
        elif phase == LoopPhase.EXPAND:
            self._phase_expand()
        
        self.state.updated_at = datetime.now().isoformat()
        
        if self.config['auto_save']:
            self._save_state()
        
        return self.state
    
    def learn(self, topic: str, goal: str, max_iterations: int = 3) -> Dict:
        """
        Run a complete learning journey.
        
        Args:
            topic: Subject to learn
            goal: Learning objective
            max_iterations: Maximum search-study-think-reflect-update-expand cycles
            
        Returns:
            Final state and summary
        """
        self.start(topic, goal)
        
        while self.state.phase != LoopPhase.COMPLETE:
            self.step()
            
            # Check iteration limit
            if (self.state.phase == LoopPhase.SEARCH and 
                self.state.iteration > max_iterations):
                self._log(f"\n🛑 Max iterations ({max_iterations}) reached")
                self.state.phase = LoopPhase.COMPLETE
                self._finalize_learning_session()
                break
        
        # Finalize session if not already done (in case loop ended naturally)
        if self.session_manager.current_session and not self.session_manager.current_session.completed_at:
            self._finalize_learning_session()
        
        return self._generate_summary()
    
    def learn_nonstop(
        self, 
        topic: str, 
        goal: str, 
        goal_threshold: int = 90,
        max_iterations: int = 100,
        pause_between_iterations: float = 0,
        checkpoint_interval: int = 1,
        on_iteration_complete: callable = None,
        stop_condition: callable = None
    ) -> Dict:
        """
        Run a non-stop learning journey until goal is achieved.
        
        The loop continues autonomously until:
        - Goal progress reaches the threshold
        - Max iterations reached
        - stop_condition callback returns True
        - KeyboardInterrupt (Ctrl+C) - gracefully saves and exits
        
        Args:
            topic: Subject to learn
            goal: Learning objective
            goal_threshold: Stop when goal progress reaches this % (default: 90)
            max_iterations: Safety limit on iterations (default: 100)
            pause_between_iterations: Seconds to pause between iterations (default: 0)
            checkpoint_interval: Save checkpoint every N iterations (default: 1)
            on_iteration_complete: Callback(state, iteration) called after each iteration
            stop_condition: Callback(state) -> bool, return True to stop
            
        Returns:
            Final state and summary
        """
        import time
        
        self.start(topic, goal)
        
        self._log(f"\n🔄 NON-STOP MODE ACTIVATED")
        self._log(f"   Goal threshold: {goal_threshold}%")
        self._log(f"   Max iterations: {max_iterations}")
        self._log(f"   Press Ctrl+C to gracefully stop\n")
        
        try:
            while self.state.phase != LoopPhase.COMPLETE:
                # Execute one phase with error handling
                try:
                    self.step()
                except Exception as step_error:
                    import traceback
                    self._log(f"\n❌ ERROR in phase {self.state.phase.value}: {step_error}")
                    self._log(f"   Traceback: {traceback.format_exc()}")
                    self._log(f"   Attempting to continue to next phase...")
                    # Try to advance to next phase to avoid infinite loop on same error
                    try:
                        self._advance_phase_on_error()
                    except:
                        self._log(f"   ❌ Cannot recover, stopping loop")
                        self.state.phase = LoopPhase.COMPLETE
                        break
                
                # Check if we completed an iteration (just finished EXPAND)
                if self.state.phase == LoopPhase.SEARCH and self.state.iteration > 1:
                    iteration_just_completed = self.state.iteration - 1
                    
                    self._log(f"\n{'─'*60}")
                    self._log(f"✅ Iteration {iteration_just_completed} complete")
                    self._log(f"   Papers studied: {self.state.total_papers_studied}")
                    self._log(f"   Knowledge bits: {self.state.total_knowledge_bits}")
                    self._log(f"{'─'*60}\n")
                    
                    # Callback
                    if on_iteration_complete:
                        try:
                            on_iteration_complete(self.state, iteration_just_completed)
                        except Exception as e:
                            self._log(f"   ⚠️ Callback error: {e}")
                    
                    # Checkpoint save
                    if checkpoint_interval and iteration_just_completed % checkpoint_interval == 0:
                        self._save_state()
                        self._log(f"   💾 Checkpoint saved")
                    
                    # Pause between iterations
                    if pause_between_iterations > 0:
                        self._log(f"   ⏸️ Pausing {pause_between_iterations}s...")
                        time.sleep(pause_between_iterations)
                
                # Check stop conditions
                if stop_condition and stop_condition(self.state):
                    self._log(f"\n🛑 Stop condition met")
                    self.state.phase = LoopPhase.COMPLETE
                    self._finalize_learning_session()
                    break
                
                # Check iteration limit
                if self.state.phase == LoopPhase.SEARCH and self.state.iteration > max_iterations:
                    self._log(f"\n🛑 Max iterations ({max_iterations}) reached")
                    self.state.phase = LoopPhase.COMPLETE
                    self._finalize_learning_session()
                    break
                    
        except KeyboardInterrupt:
            self._log(f"\n\n⚠️ Interrupted by user (Ctrl+C)")
            self._log(f"   Saving current state and session...")
            self._finalize_learning_session()  # Save session data before exit
            self.state.phase = LoopPhase.PAUSED
            self._save_state()
            self._log(f"   State saved. Resume with: resume_learning('{self.actor_name}', '{self.state.loop_id}')")
        
        # Finalize session if not already done
        if self.session_manager.current_session and not self.session_manager.current_session.completed_at:
            self._finalize_learning_session()
        
        return self._generate_summary()
    
    def pause(self):
        """Pause the learning loop gracefully."""
        if self.state.phase != LoopPhase.PAUSED:
            self.state.phase = LoopPhase.PAUSED
            self._save_state()
            self._log("⏸️ Loop paused")
    
    def run_forever(
        self,
        topics: List[str] = None,
        goal_generator: callable = None,
        idle_search_interval: float = 3600,
        **nonstop_kwargs
    ) -> None:
        """
        Run the cognitive loop forever, cycling through topics.
        
        This is the ultimate autonomous learning mode. The agent will:
        - Learn about each topic in the list
        - Generate new topics based on emerging questions
        - Never stop until manually interrupted
        
        Args:
            topics: Initial list of topics to learn about
            goal_generator: Callback(topic) -> goal string
            idle_search_interval: Seconds between topic cycles (default: 1 hour)
            **nonstop_kwargs: Additional args passed to learn_nonstop
        """
        import time
        
        if topics is None:
            topics = []
        
        topic_queue = list(topics)
        completed_topics = []
        
        self._log(f"\n{'='*60}")
        self._log(f"🌀 FOREVER MODE - Autonomous Learning Agent")
        self._log(f"{'='*60}")
        self._log(f"   Initial topics: {len(topic_queue)}")
        self._log(f"   Press Ctrl+C to stop\n")
        
        try:
            cycle = 0
            while True:
                cycle += 1
                
                # Get next topic
                if topic_queue:
                    topic = topic_queue.pop(0)
                elif self.state and self.state.emerging_questions:
                    # Generate topic from emerging questions
                    topic = self.state.emerging_questions.pop(0)
                    self._log(f"\n🔮 Generated topic from emerging questions: {topic}")
                else:
                    # Idle - wait for new topics
                    self._log(f"\n😴 No topics in queue. Waiting {idle_search_interval}s...")
                    time.sleep(idle_search_interval)
                    continue
                
                # Generate goal
                if goal_generator:
                    goal = goal_generator(topic)
                else:
                    goal = f"Develop comprehensive understanding of {topic}"
                
                self._log(f"\n{'='*60}")
                self._log(f"🎯 CYCLE {cycle}: {topic}")
                self._log(f"{'='*60}")
                
                # Learn about this topic
                result = self.learn_nonstop(topic, goal, **nonstop_kwargs)
                
                completed_topics.append({
                    "topic": topic,
                    "goal": goal,
                    "result": result,
                    "completed_at": datetime.now().isoformat()
                })
                
                # Add knowledge gaps and emerging questions to queue
                if self.state:
                    for gap in self.state.knowledge_gaps[:3]:
                        gap_str = gap.get('gap', gap) if isinstance(gap, dict) else gap
                        if gap_str not in topic_queue:
                            topic_queue.append(gap_str)
                    
                    for q in self.state.emerging_questions[:2]:
                        if q not in topic_queue:
                            topic_queue.append(q)
                
                self._log(f"\n✅ Topic complete. Queue: {len(topic_queue)} topics")
                
        except KeyboardInterrupt:
            self._log(f"\n\n🛑 Forever mode stopped by user")
            self._log(f"   Completed {len(completed_topics)} topics")
            self._log(f"   Remaining in queue: {len(topic_queue)}")
            if self.state:
                self._finalize_learning_session()  # Save any ongoing session
                self._save_state()
    
    def _phase_search(self):
        """SEARCH phase: Discover relevant papers and resources."""
        self._log("🔍 Searching for knowledge sources...")
        
        # Generate search strategy
        current_knowledge = self._get_current_knowledge_summary()
        previous_searches = "\n".join(self.state.search_queries) or "None"
        
        prompt = SEARCH_STRATEGY_PROMPT.format(
            topic=self.state.topic,
            goal=self.state.goal,
            current_knowledge=current_knowledge,
            previous_searches=previous_searches
        )
        
        response = self._llm_call(prompt, phase="search")
        strategy = self._parse_json(response)
        
        self._log(f"   Strategy: {strategy.get('reasoning', 'Generated queries')[:100]}...")
        
        # Store focus areas in session
        if self.config.get('save_session', True) and self.session_manager.current_session:
            self.session_manager.current_session.focus_areas = strategy.get('focus_areas', [])
        
        # Execute searches
        all_papers = []
        queries = strategy.get('queries', [{'query': self.state.topic}])
        
        for q in queries[:3]:  # Limit queries
            query_str = q.get('query', q) if isinstance(q, dict) else q
            purpose = q.get('purpose', 'general search') if isinstance(q, dict) else 'general search'
            self.state.search_queries.append(query_str)
            
            self._log(f"   🔎 Query: {query_str}")
            
            try:
                papers = self.scholar.search(
                    query_str,
                    depth=self.config['search_depth'],
                    year_from=self.config.get('year_from'),
                    year_to=self.config.get('year_to')
                )
                
                # Filter by citations if configured
                if self.config['min_citations']:
                    papers = [p for p in papers if p.citation_count >= self.config['min_citations']]
                
                self._log(f"      Found {len(papers)} papers")
                
                paper_dicts = [paper.to_dict() for paper in papers[:self.config['max_papers_per_search']]]
                all_papers.extend(paper_dicts)
                
                # Record search in learning session
                if self.config.get('save_session', True) and self.session_manager.current_session:
                    self.session_manager.record_search(query_str, purpose, paper_dicts)
                    
            except Exception as e:
                self._log(f"      ⚠️ Search error: {e}")
        
        # Deduplicate by title
        seen_titles = set()
        unique_papers = []
        for p in all_papers:
            title = p.get('title', '').lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(p)
        
        self.state.discovered_papers = unique_papers
        self._log(f"\n   📚 Total unique papers discovered: {len(unique_papers)}")
        
        # Select papers for study
        if unique_papers:
            self._select_papers_for_study(unique_papers, strategy.get('focus_areas', []))
        
        self._record_history("search", {
            "queries": self.state.search_queries[-3:],
            "papers_found": len(unique_papers),
            "papers_selected": len(self.state.selected_papers)
        })
        
        # Advance phase
        if self.state.selected_papers:
            self.state.phase = LoopPhase.STUDY
        else:
            self._log("   ⚠️ No papers to study, skipping to THINK")
            self.state.phase = LoopPhase.THINK
    
    def _select_papers_for_study(self, papers: List[Dict], focus_areas: List[str]):
        """Select the most valuable papers for deep study."""
        
        # Format papers for prompt
        papers_str = "\n".join([
            f"- [{p.get('citation_count', 0)} cites] {p.get('title', 'Unknown')}\n"
            f"  {p.get('publication_info', '')}"
            for p in papers[:30]
        ])
        
        prompt = PAPER_SELECTION_PROMPT.format(
            topic=self.state.topic,
            goal=self.state.goal,
            focus_areas=", ".join(focus_areas) or "general understanding",
            papers=papers_str
        )
        
        response = self._llm_call(prompt)
        selection = self._parse_json(response)
        
        # Match selected titles to papers
        selected = []
        for sel in selection.get('selected', [])[:self.config['max_papers_to_study']]:
            sel_title = sel.get('title', '').lower()
            for paper in papers:
                if sel_title in paper.get('title', '').lower():
                    paper['selection_reason'] = sel.get('reason', '')
                    paper['expected_value'] = sel.get('expected_value', '')
                    paper['priority'] = sel.get('priority', 5)
                    selected.append(paper)
                    break
        
        # If no matches, take top cited
        if not selected:
            selected = sorted(papers, key=lambda x: x.get('citation_count', 0), reverse=True)[:self.config['max_papers_to_study']]
        
        self.state.selected_papers = selected
        self._log(f"   📑 Selected {len(selected)} papers for study")
    
    def _phase_study(self):
        """STUDY phase: Deep dive into selected materials."""
        self._log("📖 Studying selected materials...")
        
        if not self.state.selected_papers:
            self._log("   No papers to study")
            self.state.phase = LoopPhase.THINK
            return
        
        # Check if PDF fetching is enabled and available
        use_pdf = self.config.get('fetch_pdfs', True)
        
        if use_pdf:
            try:
                from src.actions.study_processor import StudyProcessor, HAS_PDF_FETCHER, HAS_SMART_FETCHER
                if HAS_SMART_FETCHER:
                    self._log("   📄 Smart Paper Fetcher enabled (LLM-guided PDF extraction)")
                elif HAS_PDF_FETCHER:
                    self._log("   📄 PDF fetching enabled - will try to get full papers")
                else:
                    self._log("   ⚠️ PDF fetcher not available - using abstracts only")
                    use_pdf = False
            except ImportError:
                use_pdf = False
        
        study_config = {
            'extraction_targets': [
                {'type': 'concept', 'description': 'Key concepts and definitions'},
                {'type': 'method', 'description': 'Methods and approaches'},
                {'type': 'finding', 'description': 'Key findings and results'},
                {'type': 'application', 'description': 'Practical applications'}
            ],
            'focus_areas': [self.state.topic, self.state.goal],
            'generate_summary': True,
            'analyze_enlightenment': True
        }
        
        for i, paper in enumerate(self.state.selected_papers[:self.config.get('max_papers_to_study', 3)]):
            self._log(f"\n   [{i+1}] Studying: {paper.get('title', 'Unknown')[:60]}...")
            
            if use_pdf:
                # Try full PDF study
                try:
                    processor = StudyProcessor(self.actor_name)
                    study_result = processor.study_from_url(paper, study_config)
                    content_source = study_result.get('paper', {}).get('content_source', 'unknown')
                    self._log(f"       Content source: {content_source}")
                except Exception as e:
                    self._log(f"       ⚠️ PDF study failed: {e}")
                    study_result = self._simulate_paper_study(paper)
            else:
                # Fallback to simulated study from title/metadata
                study_result = self._simulate_paper_study(paper)
            
            self.state.studied_papers.append({
                "paper": paper,
                "study_result": study_result,
                "studied_at": datetime.now().isoformat()
            })
            
            # Extract knowledge bits based on result format
            if 'extraction' in study_result:
                # New format from StudyProcessor
                knowledge_bits = study_result.get('extraction', {}).get('knowledge_bits', [])
                notes = study_result.get('study_notes', {})
                questions = study_result.get('extraction', {}).get('questions_raised', [])
            else:
                # Old format from simulate_paper_study
                knowledge_bits = study_result.get('knowledge_bits', [])
                notes = study_result.get('notes', {})
                questions = study_result.get('questions', [])
            
            self.state.knowledge_bits.extend(knowledge_bits)
            
            if notes:
                self.state.study_notes.append(notes)
            
            self.state.emerging_questions.extend(questions)
            
            # Record in learning session for LoRA
            if self.config.get('save_session', True) and self.session_manager.current_session:
                self.session_manager.record_study_note(
                    title=paper.get('title', 'Unknown'),
                    url=paper.get('link', paper.get('url', '')),
                    source_type='paper',
                    key_points=notes.get('key_concepts', []) if isinstance(notes, dict) else [],
                    summary=notes.get('summary', notes.get('paper_summary', '')) if isinstance(notes, dict) else '',
                    questions=questions,
                    relevance=0.7 if paper.get('citation_count', 0) > 50 else 0.5
                )
                
                # Record each knowledge bit
                for kb in knowledge_bits:
                    kb['source'] = paper.get('title', 'Unknown')
                    self.session_manager.record_knowledge_bit(kb)
        
        self.state.total_papers_studied += len(self.state.studied_papers)
        self.state.total_knowledge_bits += len(self.state.knowledge_bits)
        
        self._log(f"\n   ✅ Studied {len(self.state.studied_papers)} papers")
        self._log(f"   📝 Extracted {len(self.state.knowledge_bits)} knowledge bits")
        
        self._record_history("study", {
            "papers_studied": len(self.state.studied_papers),
            "knowledge_bits": len(self.state.knowledge_bits)
        })
        
        self.state.phase = LoopPhase.THINK
    
    def _simulate_paper_study(self, paper: Dict) -> Dict:
        """Simulate studying a paper by analyzing its title and metadata."""
        
        prompt = f"""Analyze this academic paper based on its title and metadata.
Extract potential knowledge and insights.

PAPER:
Title: {paper.get('title', 'Unknown')}
Authors: {paper.get('authors', 'Unknown')}
Publication: {paper.get('publication_info', '')}
Citations: {paper.get('citation_count', 0)}
Year: {paper.get('year', 'Unknown')}

CONTEXT:
Topic: {self.state.topic}
Goal: {self.state.goal}

Return JSON:
{{
    "knowledge_bits": [
        {{
            "title": "knowledge title",
            "type": "concept|method|finding|application",
            "description": "what was learned",
            "confidence": "high|medium|low"
        }}
    ],
    "notes": {{
        "summary": "what this paper likely contributes",
        "key_concepts": ["concept1", "concept2"],
        "methodology_hints": "apparent methodology",
        "relevance": "how it relates to goal"
    }},
    "questions": ["question raised by this paper"]
}}
"""
        
        response = self._llm_call(prompt, temperature=0.4)
        return self._parse_json(response)
    
    def _phase_think(self):
        """THINK phase: Synthesize and reason about learned information."""
        self._log("🤔 Thinking and synthesizing...")
        
        # Format knowledge for synthesis
        kb_str = json.dumps(self.state.knowledge_bits[:20], indent=2)
        notes_str = json.dumps(self.state.study_notes[:10], indent=2)
        
        prompt = SYNTHESIS_PROMPT.format(
            topic=self.state.topic,
            goal=self.state.goal,
            knowledge_bits=kb_str,
            study_notes=notes_str
        )
        
        response = self._llm_call(prompt, temperature=0.5)
        synthesis = self._parse_json(response)
        
        self.state.synthesis = synthesis.get('synthesis', '')
        self.state.conclusions = synthesis.get('conclusions', [])
        
        # Add uncertainties to emerging questions
        self.state.emerging_questions.extend(synthesis.get('uncertainties', []))
        
        self.state.thoughts.append({
            "iteration": self.state.iteration,
            "synthesis": synthesis,
            "thought_at": datetime.now().isoformat()
        })
        
        self._log(f"   💡 Key themes: {synthesis.get('key_themes', [])}")
        self._log(f"   📌 Conclusions: {len(self.state.conclusions)}")
        
        self._record_history("think", {
            "themes": synthesis.get('key_themes', []),
            "conclusions": len(self.state.conclusions)
        })
        
        self.state.phase = LoopPhase.REFLECT
    
    def _phase_reflect(self):
        """REFLECT phase: Evaluate new knowledge against existing beliefs."""
        self._log("🪞 Reflecting on new knowledge...")
        
        # Get current beliefs
        principles = self.kb.get_principles()
        core_knowledge = self.kb.get_core_knowledge()
        experiences = self.kb.get_experiences()
        mindset = self.kb.get_current_mindset()
        
        prompt = REFLECTION_PROMPT.format(
            topic=self.state.topic,
            synthesis=self.state.synthesis,
            conclusions="\n".join(f"- {c}" for c in self.state.conclusions),
            principles="\n".join(f"- {p}" for p in principles[:10]),
            core_knowledge="\n".join(f"- {k}" for k in core_knowledge[:10]),
            experiences="\n".join(f"- {e}" for e in experiences[:10]),
            mindset=json.dumps(mindset, indent=2) if mindset else "Not yet defined"
        )
        
        response = self._llm_call(prompt, temperature=0.8, phase="reflect")
        reflection = self._parse_json(response)
        
        # Store reflection results
        self.state.enlightenments.append({
            "iteration": self.state.iteration,
            "assessment": reflection.get('overall_assessment', 'unknown'),
            "growth": reflection.get('intellectual_growth', ''),
            "reflection_at": datetime.now().isoformat()
        })
        
        self.state.confirmations.extend(reflection.get('confirmations', []))
        self.state.expansions.extend(reflection.get('expansions', []))
        self.state.challenges.extend(reflection.get('challenges', []))
        self.state.revolutionary_ideas.extend(reflection.get('revolutionary', []))
        
        # Record insights in learning session for LoRA
        if self.config.get('save_session', True) and self.session_manager.current_session:
            for conf in reflection.get('confirmations', []):
                self.session_manager.record_insight(
                    insight_type='confirmation',
                    content=conf.get('confirmed_belief', str(conf)),
                    evidence=conf.get('supporting_evidence', []) if isinstance(conf, dict) else [],
                    confidence=0.9
                )
            
            for exp in reflection.get('expansions', []):
                self.session_manager.record_insight(
                    insight_type='expansion',
                    content=exp.get('new_dimension', str(exp)),
                    prior=exp.get('original_belief', '') if isinstance(exp, dict) else '',
                    new_understanding=exp.get('expanded_understanding', '') if isinstance(exp, dict) else '',
                    confidence=0.8
                )
            
            for chal in reflection.get('challenges', []):
                self.session_manager.record_insight(
                    insight_type='challenge',
                    content=chal.get('challenged_belief', str(chal)),
                    evidence=chal.get('challenging_evidence', []) if isinstance(chal, dict) else [],
                    prior=chal.get('challenged_belief', '') if isinstance(chal, dict) else '',
                    new_understanding=chal.get('new_perspective', '') if isinstance(chal, dict) else '',
                    confidence=0.85
                )
            
            for rev in reflection.get('revolutionary', []):
                self.session_manager.record_insight(
                    insight_type='revolutionary',
                    content=rev.get('paradigm_shift', str(rev)),
                    implications=rev.get('implications', []) if isinstance(rev, dict) else [],
                    confidence=0.75
                )
            
            # Record synthesis
            self.session_manager.record_synthesis(
                self.state.synthesis,
                self.state.conclusions
            )
        
        # Log summary
        assessment = reflection.get('overall_assessment', 'unknown')
        emoji = {'high_value': '🌟', 'moderate_value': '👍', 'low_value': '📉'}.get(assessment, '❓')
        
        self._log(f"   {emoji} Assessment: {assessment}")
        self._log(f"   ✅ Confirmations: {len(reflection.get('confirmations', []))}")
        self._log(f"   🌱 Expansions: {len(reflection.get('expansions', []))}")
        self._log(f"   ⚡ Challenges: {len(reflection.get('challenges', []))}")
        self._log(f"   🚀 Revolutionary: {len(reflection.get('revolutionary', []))}")
        
        self._record_history("reflect", {
            "assessment": assessment,
            "confirmations": len(self.state.confirmations),
            "expansions": len(self.state.expansions),
            "challenges": len(self.state.challenges),
            "revolutionary": len(self.state.revolutionary_ideas)
        })
        
        self.state.phase = LoopPhase.UPDATE
    
    def _phase_update(self):
        """UPDATE phase: Integrate new knowledge into knowledge base and persist to files."""
        self._log("📝 Updating knowledge base...")
        
        prompt = UPDATE_DECISION_PROMPT.format(
            reflection=json.dumps(self.state.enlightenments[-1] if self.state.enlightenments else {}, indent=2),
            confirmations=json.dumps(self.state.confirmations[-5:], indent=2),
            expansions=json.dumps(self.state.expansions[-5:], indent=2),
            challenges=json.dumps(self.state.challenges[-5:], indent=2),
            revolutionary=json.dumps(self.state.revolutionary_ideas[-3:], indent=2)
        )
        
        response = self._llm_call(prompt, temperature=0.3)
        updates = self._parse_json(response)
        
        # Apply updates
        mindset_updates = updates.get('mindset_updates', [])
        self.state.mindset_updates.extend(mindset_updates)
        
        for update in mindset_updates:
            self._log(f"   🔄 Mindset: {update[:60]}...")
        
        # Track what was persisted
        persisted = {
            'principles': 0,
            'experiences': 0,
            'knowledge': 0,
            'challenges': 0
        }
        
        # PERSIST knowledge to actor's YAML files (with archiving)
        source = f"cognitive_loop_{self.state.topic[:30]}"
        
        for knowledge in updates.get('knowledge_to_add', []):
            k_type = knowledge.get('type', 'experience')
            content = knowledge.get('content', '')
            k_source = knowledge.get('source', self.state.topic)
            
            # Persist based on type
            if k_type == 'principle':
                if self.kb.add_principle(content, source=k_source):
                    persisted['principles'] += 1
                    self._log(f"   💾 Persisted principle: {content[:50]}...")
            elif k_type == 'experience':
                if self.kb.add_experience(content, source=k_source):
                    persisted['experiences'] += 1
                    self._log(f"   💾 Persisted experience: {content[:50]}...")
            elif k_type == 'knowledge':
                if self.kb.add_core_knowledge(content, source=k_source):
                    persisted['knowledge'] += 1
                    self._log(f"   💾 Persisted knowledge: {content[:50]}...")
            elif k_type == 'challenge':
                if self.kb.add_challenge(content, source=k_source):
                    persisted['challenges'] += 1
                    self._log(f"   💾 Persisted challenge: {content[:50]}...")
            else:
                # Default to experience
                if self.kb.add_experience(content, source=k_source):
                    persisted['experiences'] += 1
            
            self.state.knowledge_evolved.append(knowledge)
        
        # Handle beliefs that need investigation (add as challenges)
        for belief in updates.get('beliefs_to_investigate', []):
            belief_text = belief.get('belief', belief) if isinstance(belief, dict) else belief
            reason = belief.get('reason', 'needs investigation') if isinstance(belief, dict) else 'needs investigation'
            if self.kb.add_challenge(f"{belief_text} - {reason}", source=source):
                persisted['challenges'] += 1
                self._log(f"   💾 Added investigation: {belief_text[:50]}...")
        
        # Process revolutionary ideas as new principles
        for rev in self.state.revolutionary_ideas[-3:]:
            paradigm = rev.get('new_paradigm', '')
            adoption = rev.get('adoption', 'investigate')
            if adoption == 'adopt' and paradigm:
                if self.kb.add_principle(paradigm, source=f"revolutionary_{source}"):
                    persisted['principles'] += 1
                    self._log(f"   🚀 Adopted revolutionary principle: {paradigm[:50]}...")
        
        # Process expansions as new knowledge/experiences
        for exp in self.state.expansions[-5:]:
            new_persp = exp.get('new_perspective', '')
            if new_persp:
                if self.kb.add_experience(new_persp, source=source):
                    persisted['experiences'] += 1
        
        # Log persistence summary
        total_persisted = sum(persisted.values())
        if total_persisted > 0:
            self._log(f"\n   💾 PERSISTED TO ACTOR FILES:")
            self._log(f"      Principles: {persisted['principles']}")
            self._log(f"      Experiences: {persisted['experiences']}")
            self._log(f"      Knowledge: {persisted['knowledge']}")
            self._log(f"      Challenges: {persisted['challenges']}")
            self._log(f"      (Previous versions archived)")
        
        # Record to learning session for LoRA
        if self.config.get('save_session', True) and self.session_manager.current_session:
            for update in mindset_updates:
                self.session_manager.record_knowledge_update('mindset', update)
            
            for knowledge in updates.get('knowledge_to_add', []):
                k_type = knowledge.get('type', 'experience')
                content = knowledge.get('content', '')
                self.session_manager.record_knowledge_update(k_type, content)
        
        # Record learning event
        self.kb.record_learning_event(
            event_type='cognitive_loop_update',
            description=f"Cognitive loop iteration {self.state.iteration} on: {self.state.topic}",
            insights=[
                f"Studied {len(self.state.studied_papers)} papers",
                f"Extracted {len(self.state.knowledge_bits)} knowledge bits",
                f"Persisted {total_persisted} items to actor files",
                updates.get('update_summary', 'Knowledge updated')
            ],
            mindset_changes=mindset_updates
        )
        
        self._log(f"   ✅ Update complete: {updates.get('update_summary', 'Applied')[:60]}")
        
        self._record_history("update", {
            "mindset_updates": len(mindset_updates),
            "knowledge_added": len(updates.get('knowledge_to_add', [])),
            "persisted": persisted
        })
        
        self.state.phase = LoopPhase.EXPAND
    
    def _phase_expand(self):
        """EXPAND phase: Identify gaps and plan next exploration."""
        self._log("🌐 Expanding - identifying knowledge gaps...")
        
        prompt = EXPANSION_PROMPT.format(
            topic=self.state.topic,
            goal=self.state.goal,
            papers_studied=self.state.total_papers_studied,
            knowledge_bits_count=self.state.total_knowledge_bits,
            conclusions="\n".join(f"- {c}" for c in self.state.conclusions[:5]),
            uncertainties="\n".join(f"- {q}" for q in self.state.emerging_questions[:10]),
            emerging_questions="\n".join(f"- {q}" for q in self.state.emerging_questions[:10]),
            challenges="\n".join(f"- {c.get('challenged_belief', '')}" for c in self.state.challenges[:5])
        )
        
        response = self._llm_call(prompt, temperature=0.4)
        expansion = self._parse_json(response)
        
        goal_progress = expansion.get('goal_progress', 50)
        continue_learning = expansion.get('continue_learning', True)
        
        self._log(f"   📊 Goal progress: {goal_progress}%")
        self._log(f"   🔄 Continue learning: {continue_learning}")
        self._log(f"   💭 Reasoning: {expansion.get('reasoning', 'N/A')[:100]}...")
        
        # Store gaps and next queries
        for gap in expansion.get('knowledge_gaps', []):
            gap_text = gap.get('gap', gap) if isinstance(gap, dict) else gap
            self.state.knowledge_gaps.append(gap_text)
        
        for query in expansion.get('next_queries', []):
            q = query.get('query', query) if isinstance(query, dict) else query
            self.state.next_search_queries.append(q)
        
        # Record in learning session
        if self.config.get('save_session', True) and self.session_manager.current_session:
            self.session_manager.current_session.emerging_questions = self.state.emerging_questions
            self.session_manager.current_session.knowledge_gaps = self.state.knowledge_gaps
        
        self._record_history("expand", {
            "goal_progress": goal_progress,
            "continue": continue_learning,
            "gaps_identified": len(expansion.get('knowledge_gaps', [])),
            "next_queries": len(expansion.get('next_queries', []))
        })
        
        # Decide whether to continue or complete
        if continue_learning and self.state.next_search_queries:
            self._log("\n   🔄 Continuing to next iteration...")
            self.state.iteration += 1
            
            # Reset per-iteration state (but keep cumulative data like studied_papers)
            self.state.discovered_papers = []
            self.state.selected_papers = []
            # Note: Don't clear studied_papers - keep cumulative record for study tracking
            # self.state.studied_papers = []  
            self.state.knowledge_bits = []
            self.state.study_notes = []
            
            # Use next queries for search
            self.state.search_queries = self.state.next_search_queries[:3]
            self.state.next_search_queries = self.state.next_search_queries[3:]
            
            self.state.phase = LoopPhase.SEARCH
        else:
            self._log("\n   ✅ Learning goal achieved!")
            self.state.phase = LoopPhase.COMPLETE
            
            # Finalize learning session and save for LoRA
            self._finalize_learning_session()
    
    def _finalize_learning_session(self):
        """Finalize the learning session and save all data for LoRA training."""
        if not self.config.get('save_session', True):
            return
        
        if not self.session_manager.current_session:
            return
        
        session = self.session_manager.current_session
        
        # Update session with final state data
        session.iterations = self.state.iteration
        session.total_papers_studied = self.state.total_papers_studied
        session.total_knowledge_bits = self.state.total_knowledge_bits
        session.conclusions = self.state.conclusions
        session.synthesis = self.state.synthesis
        
        # Generate Q&A pairs for training
        qa_pairs = self.session_manager.generate_qa_pairs()
        
        # Add custom Q&A pairs based on learning outcomes
        self._generate_custom_qa_pairs()
        
        # Finalize and save
        tags = [
            f"topic:{self.state.topic[:30]}",
            f"iterations:{self.state.iteration}",
            f"papers:{self.state.total_papers_studied}"
        ]
        
        session_dir = self.session_manager.finalize_session(tags=tags)
        
        self._log(f"\n   📝 LEARNING SESSION SAVED FOR LORA:")
        self._log(f"      Location: {session_dir}")
        self._log(f"      Q&A pairs: {len(session.qa_pairs)}")
        self._log(f"      Thought traces: {len(session.thought_traces)}")
        self._log(f"      Study notes: {len(session.study_notes)}")
        self._log(f"      Insights: {len(session.insights)}")
    
    def _generate_custom_qa_pairs(self):
        """Generate additional Q&A pairs from the learning session."""
        if not self.session_manager.current_session:
            return
        
        session = self.session_manager.current_session
        
        # Q&A from each conclusion
        for i, conclusion in enumerate(self.state.conclusions):
            self.session_manager.add_qa_pair(
                question=f"Based on your study of {self.state.topic}, what is conclusion #{i+1}?",
                answer=conclusion,
                context=f"Goal: {self.state.goal}",
                category="synthesis",
                confidence=0.85
            )
        
        # Q&A from mindset updates (important for alignment)
        for update in self.state.mindset_updates:
            self.session_manager.add_qa_pair(
                question=f"How has studying {self.state.topic} changed your thinking?",
                answer=update,
                category="conceptual",
                confidence=0.9
            )
        
        # Q&A from challenges (critical for alignment/correction)
        for challenge in self.state.challenges:
            old_belief = challenge.get('challenged_belief', '')
            new_view = challenge.get('new_perspective', '')
            evidence = challenge.get('challenging_evidence', '')
            
            if old_belief and new_view:
                self.session_manager.add_qa_pair(
                    question=f"What misconception about {self.state.topic} did you correct?",
                    answer=f"I previously believed: {old_belief}\n\nAfter studying the evidence, I now understand: {new_view}",
                    reasoning=f"Evidence: {evidence}" if evidence else "",
                    category="correction",
                    confidence=0.95  # High confidence for corrections
                )
        
        # Q&A about knowledge gaps (for future learning)
        if self.state.knowledge_gaps:
            gaps_text = "\n".join(f"- {gap}" for gap in self.state.knowledge_gaps[:5])
            self.session_manager.add_qa_pair(
                question=f"What gaps remain in your understanding of {self.state.topic}?",
                answer=gaps_text,
                category="metacognitive",
                confidence=0.8
            )

    def _get_current_knowledge_summary(self) -> str:
        """Get a summary of current knowledge for context."""
        principles = self.kb.get_principles()[:5]
        knowledge = self.kb.get_core_knowledge()[:5]
        
        summary = "Current Understanding:\n"
        
        if principles:
            summary += "Principles:\n" + "\n".join(f"- {p}" for p in principles) + "\n"
        
        if knowledge:
            summary += "Core Knowledge:\n" + "\n".join(f"- {k}" for k in knowledge) + "\n"
        
        if self.state.conclusions:
            summary += "Recent Conclusions:\n" + "\n".join(f"- {c}" for c in self.state.conclusions[:5])
        
        return summary or "No prior knowledge recorded"
    
    def _llm_call(self, prompt: str, temperature: float = 0.3, phase: str = "") -> str:
        """Make an LLM call and record for learning session."""
        messages = [
            {"role": "system", "content": "You are a cognitive assistant helping with learning and knowledge synthesis. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        # Use model_override if configured (for LoRA adapter support)
        model_override = self.config.get('model_override')
        
        try:
            response = get_chat_completion(messages, temperature=temperature, model_override=model_override)
        except Exception as e:
            error_msg = str(e)
            self._log(f"   ⚠️ LLM call failed: {error_msg[:100]}")
            
            # If using a LoRA adapter that doesn't exist, fall back to base model
            if model_override and ('does not exist' in error_msg or '404' in error_msg):
                self._log(f"   🔄 Falling back to base model (adapter '{model_override}' not available)")
                try:
                    response = get_chat_completion(messages, temperature=temperature, model_override=None)
                except Exception as e2:
                    raise RuntimeError(f"LLM call failed even with base model: {e2}")
            else:
                raise
        
        # Record thought trace for learning session
        if self.config.get('save_session', True) and self.session_manager.current_session:
            self.session_manager.record_thought(
                phase=phase or (self.state.phase.value if self.state else "unknown"),
                prompt=prompt,
                response=response,
                temperature=temperature
            )
        
        return response
    
    def _parse_json(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # Try finding JSON object
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        return {"error": "Could not parse JSON", "raw": response[:500]}
    
    def _advance_phase_on_error(self):
        """Advance to the next phase after an error to avoid getting stuck."""
        phase_order = [
            LoopPhase.SEARCH, LoopPhase.STUDY, LoopPhase.THINK,
            LoopPhase.REFLECT, LoopPhase.UPDATE, LoopPhase.EXPAND
        ]
        current_idx = phase_order.index(self.state.phase) if self.state.phase in phase_order else -1
        if current_idx >= 0 and current_idx < len(phase_order) - 1:
            self.state.phase = phase_order[current_idx + 1]
            self._log(f"   ⏭️ Skipping to phase: {self.state.phase.value}")
        elif self.state.phase == LoopPhase.EXPAND:
            # Finished iteration, go back to SEARCH
            self.state.iteration += 1
            self.state.phase = LoopPhase.SEARCH
            self._log(f"   ⏭️ Starting new iteration {self.state.iteration}")
        else:
            # Unknown state, complete the loop
            self.state.phase = LoopPhase.COMPLETE
    
    def _log(self, message: str):
        """Log a message if verbose."""
        if self.verbose:
            print(message)
    
    def _record_history(self, action: str, data: Dict):
        """Record an action in history."""
        self.history.append({
            "action": action,
            "phase": self.state.phase.value,
            "iteration": self.state.iteration,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
    
    def _save_state(self):
        """Save current state to disk."""
        if not self.state:
            return
        
        state_file = self.loops_dir / f"loop_{self.state.loop_id}.json"
        
        data = {
            "state": self.state.to_dict(),
            "history": self.history,
            "config": self.config
        }
        
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_state(self, loop_id: str) -> CognitiveState:
        """Load a saved cognitive loop state."""
        state_file = self.loops_dir / f"loop_{loop_id}.json"
        
        if not state_file.exists():
            raise FileNotFoundError(f"Loop not found: {loop_id}")
        
        with open(state_file, 'r') as f:
            data = json.load(f)
        
        self.state = CognitiveState.from_dict(data['state'])
        self.history = data.get('history', [])
        self.config.update(data.get('config', {}))
        
        return self.state
    
    def list_loops(self) -> List[Dict]:
        """List all saved cognitive loops."""
        loops = []
        
        for loop_file in self.loops_dir.glob('loop_*.json'):
            try:
                with open(loop_file, 'r') as f:
                    data = json.load(f)
                    state = data['state']
                    loops.append({
                        "loop_id": state['loop_id'],
                        "topic": state['topic'],
                        "phase": state['phase'],
                        "iteration": state['iteration'],
                        "started_at": state['started_at'],
                        "papers_studied": state['total_papers_studied']
                    })
            except:
                continue
        
        return sorted(loops, key=lambda x: x['started_at'], reverse=True)
    
    def get_state(self) -> Optional[CognitiveState]:
        """Get current state."""
        return self.state
    
    def list_learning_sessions(self) -> List[Dict]:
        """List all saved learning sessions."""
        return self.session_manager.list_sessions()
    
    def export_all_training_data(self, output_file: Optional[Path] = None) -> Path:
        """Export all Q&A pairs from all sessions into one training file."""
        return self.session_manager.export_all_training_data(output_file)
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of the learning journey."""
        if not self.state:
            return {"error": "No active loop"}
        
        return {
            "loop_id": self.state.loop_id,
            "topic": self.state.topic,
            "goal": self.state.goal,
            "iterations": self.state.iteration,
            "phase": self.state.phase.value,
            "statistics": {
                "total_papers_studied": self.state.total_papers_studied,
                "total_knowledge_bits": self.state.total_knowledge_bits,
                "confirmations": len(self.state.confirmations),
                "expansions": len(self.state.expansions),
                "challenges": len(self.state.challenges),
                "revolutionary_ideas": len(self.state.revolutionary_ideas),
                "mindset_updates": len(self.state.mindset_updates)
            },
            "conclusions": self.state.conclusions,
            "synthesis": self.state.synthesis,
            "knowledge_gaps": self.state.knowledge_gaps,
            "emerging_questions": self.state.emerging_questions[:10],
            "duration": {
                "started": self.state.started_at,
                "completed": self.state.updated_at
            },
            "learning_session": {
                "session_id": self.state.loop_id,
                "qa_pairs_generated": len(self.session_manager.current_session.qa_pairs) if self.session_manager.current_session else 0,
                "thought_traces": len(self.session_manager.current_session.thought_traces) if self.session_manager.current_session else 0
            } if self.config.get('save_session', True) else None
        }
    
    def get_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        summary = self._generate_summary()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                   COGNITIVE LOOP SUMMARY                      ║
╚══════════════════════════════════════════════════════════════╝

📚 Topic: {summary['topic']}
🎯 Goal: {summary['goal']}
🔄 Iterations: {summary['iterations']}
📍 Status: {summary['phase']}

📊 STATISTICS:
   • Papers Studied: {summary['statistics']['total_papers_studied']}
   • Knowledge Bits: {summary['statistics']['total_knowledge_bits']}
   • Confirmations: {summary['statistics']['confirmations']}
   • Expansions: {summary['statistics']['expansions']}
   • Challenges: {summary['statistics']['challenges']}
   • Revolutionary Ideas: {summary['statistics']['revolutionary_ideas']}
   • Mindset Updates: {summary['statistics']['mindset_updates']}

💡 KEY CONCLUSIONS:
{chr(10).join(f'   • {c}' for c in summary['conclusions'][:5]) or '   (none)'}

🔍 SYNTHESIS:
{summary['synthesis'][:500] if summary['synthesis'] else '(none)'}...

❓ REMAINING QUESTIONS:
{chr(10).join(f'   • {q}' for q in summary['emerging_questions'][:5]) or '   (none)'}

📝 LEARNING SESSION (for LoRA):
   • Session ID: {summary.get('learning_session', {}).get('session_id', 'N/A') if summary.get('learning_session') else 'disabled'}
   • Q&A Pairs: {summary.get('learning_session', {}).get('qa_pairs_generated', 0) if summary.get('learning_session') else 0}
   • Thought Traces: {summary.get('learning_session', {}).get('thought_traces', 0) if summary.get('learning_session') else 0}

⏱️ Duration: {summary['duration']['started']} to {summary['duration']['completed']}
"""
        return report


# Convenience functions

def start_learning(actor_name: str, topic: str, goal: str, **config) -> CognitiveLoop:
    """Start a new cognitive learning loop."""
    loop = CognitiveLoop(actor_name)
    loop.configure(**config)
    loop.start(topic, goal)
    return loop


def learn(actor_name: str, topic: str, goal: str, max_iterations: int = 3, **config) -> Dict:
    """Run a complete learning journey and return summary."""
    loop = CognitiveLoop(actor_name)
    loop.configure(**config)
    return loop.learn(topic, goal, max_iterations)


def learn_nonstop(
    actor_name: str, 
    topic: str, 
    goal: str,
    goal_threshold: int = 90,
    max_iterations: int = 100,
    **config
) -> Dict:
    """Run a non-stop learning journey until goal is achieved."""
    loop = CognitiveLoop(actor_name)
    loop.configure(**config)
    return loop.learn_nonstop(
        topic, goal, 
        goal_threshold=goal_threshold,
        max_iterations=max_iterations
    )


def run_forever(actor_name: str, topics: List[str] = None, **config) -> None:
    """Run the cognitive loop forever, cycling through topics."""
    loop = CognitiveLoop(actor_name)
    loop.configure(**config)
    loop.run_forever(topics=topics)


def resume_learning(actor_name: str, loop_id: str) -> CognitiveLoop:
    """Resume a saved cognitive loop."""
    loop = CognitiveLoop(actor_name)
    loop.load_state(loop_id)
    return loop
