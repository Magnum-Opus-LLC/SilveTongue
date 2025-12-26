"""
Learning Session Manager

Captures and preserves all data from a learning session for later LoRA fine-tuning.
Each session creates a complete package with:
- Raw materials (papers, sources)
- Processing traces (thoughts, reasoning)
- Insights (knowledge bits, conclusions)
- Q&A pairs for training
- Metadata for filtering and organization
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
import hashlib


@dataclass
class LearningQAPair:
    """A question-answer pair extracted from learning for fine-tuning."""
    question: str
    answer: str
    context: str = ""  # What knowledge/papers informed this
    reasoning: str = ""  # Chain of thought
    confidence: float = 0.8
    category: str = "general"  # factual, conceptual, analytical, synthesis
    source_papers: List[str] = field(default_factory=list)

    def to_training_format(self, include_reasoning: bool = True) -> Dict:
        """Convert to format suitable for fine-tuning."""
        if include_reasoning:
            return {
                "instruction": self.question,
                "input": self.context if self.context else "",
                "output": f"{self.reasoning}\n\nAnswer: {self.answer}" if self.reasoning else self.answer,
                "metadata": {
                    "category": self.category,
                    "confidence": self.confidence,
                    "sources": self.source_papers
                }
            }
        else:
            return {
                "instruction": self.question,
                "input": self.context if self.context else "",
                "output": self.answer
            }


@dataclass
class ThoughtTrace:
    """A trace of thinking/reasoning during learning."""
    phase: str  # search, study, think, reflect, update, expand
    timestamp: str
    prompt: str  # What was asked
    response: str  # What was generated
    parsed_output: Dict = field(default_factory=dict)  # Structured output
    model: str = ""
    temperature: float = 0.3
    tokens_used: int = 0


@dataclass
class StudyNote:
    """Notes from studying a paper or source."""
    source_title: str
    source_url: str
    source_type: str  # paper, article, documentation
    key_points: List[str] = field(default_factory=list)
    # {"text": ..., "context": ...}
    quotes: List[Dict] = field(default_factory=list)
    questions_raised: List[str] = field(default_factory=list)
    connections_made: List[str] = field(
        default_factory=list)  # Links to other knowledge
    summary: str = ""
    relevance_score: float = 0.0
    studied_at: str = ""


@dataclass
class InsightRecord:
    """A recorded insight or realization during learning."""
    insight_type: str  # confirmation, expansion, challenge, revolutionary
    content: str
    evidence: List[str] = field(default_factory=list)
    prior_belief: str = ""  # What was believed before
    new_understanding: str = ""  # What is understood now
    confidence: float = 0.8
    implications: List[str] = field(default_factory=list)


@dataclass
class LearningSession:
    """
    Complete record of a learning session for LoRA fine-tuning.

    Captures everything that happened during learning so it can be
    used to train a model to learn similarly.
    """
    # Identity (required fields first)
    session_id: str
    actor_name: str
    created_at: str
    topic: str
    goal: str

    # Optional fields with defaults
    completed_at: str = ""

    # Focus areas
    focus_areas: List[str] = field(default_factory=list)

    # Sources
    # {"query": ..., "purpose": ..., "results_count": ...}
    search_queries: List[Dict] = field(default_factory=list)
    papers_discovered: List[Dict] = field(default_factory=list)
    papers_studied: List[Dict] = field(default_factory=list)

    # Study outputs
    study_notes: List[StudyNote] = field(default_factory=list)
    knowledge_bits: List[Dict] = field(default_factory=list)

    # Thinking traces
    thought_traces: List[ThoughtTrace] = field(default_factory=list)
    synthesis: str = ""
    conclusions: List[str] = field(default_factory=list)

    # Insights & Reflections
    insights: List[InsightRecord] = field(default_factory=list)
    confirmations: List[Dict] = field(default_factory=list)
    expansions: List[Dict] = field(default_factory=list)
    challenges: List[Dict] = field(default_factory=list)
    revolutionary_ideas: List[Dict] = field(default_factory=list)

    # Knowledge updates
    mindset_updates: List[str] = field(default_factory=list)
    principles_added: List[str] = field(default_factory=list)
    experiences_added: List[str] = field(default_factory=list)
    knowledge_added: List[str] = field(default_factory=list)

    # Questions for exploration
    emerging_questions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)

    # Training pairs generated
    qa_pairs: List[LearningQAPair] = field(default_factory=list)

    # Metadata
    iterations: int = 0
    total_papers_studied: int = 0
    total_knowledge_bits: int = 0
    total_tokens_used: int = 0
    duration_seconds: float = 0
    model_used: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, list):
                data[key] = [
                    item.to_dict() if hasattr(item, 'to_dict') else
                    asdict(item) if hasattr(
                        item, '__dataclass_fields__') else item
                    for item in value
                ]
            elif hasattr(value, 'to_dict'):
                data[key] = value.to_dict()
            elif hasattr(value, '__dataclass_fields__'):
                data[key] = asdict(value)
            else:
                data[key] = value
        return data

    def get_content_hash(self) -> str:
        """Get a hash of the learning content for deduplication."""
        content = f"{self.topic}:{self.goal}:{json.dumps(sorted(self.conclusions))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class LearningSessionManager:
    """
    Manages learning sessions - creating, saving, loading, and exporting.
    """

    def __init__(self, actor_name: str = "student", data_dir: Optional[Path] = None):
        self.actor_name = actor_name

        if data_dir is None:
            # Use array_dir from config.json for data storage
            try:
                from src.utils.chat_utils import load_config
                config = load_config()
                data_dir = Path(config.get('array_dir', '')) / "data"
            except Exception:
                # Fallback to project data dir
                project_root = Path(__file__).parent.parent.parent
                data_dir = project_root / "data"

        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / actor_name / "learning_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.current_session: Optional[LearningSession] = None
        self._start_time: Optional[datetime] = None

    def start_session(self, topic: str, goal: str, session_id: Optional[str] = None) -> LearningSession:
        """Start a new learning session."""
        import uuid

        self._start_time = datetime.now()

        if session_id is None:
            session_id = f"{self._start_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        self.current_session = LearningSession(
            session_id=session_id,
            actor_name=self.actor_name,
            created_at=self._start_time.isoformat(),
            topic=topic,
            goal=goal
        )

        # Create session directory
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        return self.current_session

    def record_search(self, query: str, purpose: str, results: List[Dict]):
        """Record a search operation."""
        if not self.current_session:
            return

        self.current_session.search_queries.append({
            "query": query,
            "purpose": purpose,
            "results_count": len(results),
            "timestamp": datetime.now().isoformat()
        })

        # Add discovered papers
        for paper in results:
            if paper not in self.current_session.papers_discovered:
                self.current_session.papers_discovered.append(paper)

    def record_thought(self, phase: str, prompt: str, response: str,
                       parsed: Optional[Dict] = None, model: str = "",
                       temperature: float = 0.3, tokens: int = 0):
        """Record a thought/reasoning trace."""
        if not self.current_session:
            return

        trace = ThoughtTrace(
            phase=phase,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            parsed_output=parsed or {},
            model=model,
            temperature=temperature,
            tokens_used=tokens
        )

        self.current_session.thought_traces.append(trace)
        self.current_session.total_tokens_used += tokens

    def record_study_note(self, title: str, url: str, source_type: str,
                          key_points: List[str], summary: str,
                          questions: Optional[List[str]] = None,
                          relevance: float = 0.5):
        """Record notes from studying a source."""
        if not self.current_session:
            return

        note = StudyNote(
            source_title=title,
            source_url=url,
            source_type=source_type,
            key_points=key_points,
            summary=summary,
            questions_raised=questions or [],
            relevance_score=relevance,
            studied_at=datetime.now().isoformat()
        )

        self.current_session.study_notes.append(note)

        # Also track paper as studied
        self.current_session.papers_studied.append({
            "title": title,
            "url": url,
            "type": source_type
        })
        self.current_session.total_papers_studied += 1

    def record_knowledge_bit(self, bit: Dict):
        """Record extracted knowledge bit."""
        if not self.current_session:
            return

        self.current_session.knowledge_bits.append(bit)
        self.current_session.total_knowledge_bits += 1

    def record_insight(self, insight_type: str, content: str,
                       evidence: List[str] = None, prior: str = "",
                       new_understanding: str = "", confidence: float = 0.8,
                       implications: List[str] = None):
        """Record an insight or realization."""
        if not self.current_session:
            return

        insight = InsightRecord(
            insight_type=insight_type,
            content=content,
            evidence=evidence or [],
            prior_belief=prior,
            new_understanding=new_understanding,
            confidence=confidence,
            implications=implications or []
        )

        self.current_session.insights.append(insight)

        # Also categorize
        if insight_type == "confirmation":
            self.current_session.confirmations.append(asdict(insight))
        elif insight_type == "expansion":
            self.current_session.expansions.append(asdict(insight))
        elif insight_type == "challenge":
            self.current_session.challenges.append(asdict(insight))
        elif insight_type == "revolutionary":
            self.current_session.revolutionary_ideas.append(asdict(insight))

    def record_synthesis(self, synthesis: str, conclusions: List[str]):
        """Record synthesis and conclusions."""
        if not self.current_session:
            return

        self.current_session.synthesis = synthesis
        self.current_session.conclusions = conclusions

    def record_knowledge_update(self, update_type: str, content: str):
        """Record a knowledge update that was persisted."""
        if not self.current_session:
            return

        if update_type == "principle":
            self.current_session.principles_added.append(content)
        elif update_type == "experience":
            self.current_session.experiences_added.append(content)
        elif update_type == "knowledge":
            self.current_session.knowledge_added.append(content)
        elif update_type == "mindset":
            self.current_session.mindset_updates.append(content)

    def add_qa_pair(self, question: str, answer: str, context: str = "",
                    reasoning: str = "", category: str = "general",
                    sources: List[str] = None, confidence: float = 0.8):
        """Add a Q&A pair for training."""
        if not self.current_session:
            return

        qa = LearningQAPair(
            question=question,
            answer=answer,
            context=context,
            reasoning=reasoning,
            confidence=confidence,
            category=category,
            source_papers=sources or []
        )

        self.current_session.qa_pairs.append(qa)

    def generate_qa_pairs(self) -> List[LearningQAPair]:
        """
        Auto-generate Q&A pairs from the session for training.
        Creates different types of training examples.
        """
        if not self.current_session:
            return []

        qa_pairs = []
        session = self.current_session

        # 1. Factual Q&A from knowledge bits
        for bit in session.knowledge_bits:
            content = bit.get('content', bit.get('key_point', ''))
            source = bit.get('source', '')
            if content:
                qa_pairs.append(LearningQAPair(
                    question=f"What is known about {session.topic} regarding: {content[:50]}...?",
                    answer=content,
                    context=f"Based on research into {session.topic}",
                    category="factual",
                    source_papers=[source] if source else [],
                    confidence=bit.get('confidence', 0.7)
                ))

        # 2. Synthesis Q&A
        if session.synthesis:
            qa_pairs.append(LearningQAPair(
                question=f"What are the key insights from studying {session.topic}?",
                answer=session.synthesis,
                context=f"Goal: {session.goal}",
                category="synthesis",
                confidence=0.85
            ))

        # 3. Conclusion Q&A
        for conclusion in session.conclusions:
            qa_pairs.append(LearningQAPair(
                question=f"What conclusions can be drawn about {session.topic}?",
                answer=conclusion,
                category="analytical",
                confidence=0.8
            ))

        # 4. Challenge/update Q&A (most valuable for alignment)
        for challenge in session.challenges:
            prior = challenge.get(
                'prior_belief', challenge.get('challenged_belief', ''))
            new = challenge.get('new_understanding',
                                challenge.get('new_perspective', ''))
            if prior and new:
                qa_pairs.append(LearningQAPair(
                    question=f"How has understanding of {session.topic} evolved?",
                    answer=f"Previous understanding: {prior}\n\nUpdated understanding: {new}",
                    reasoning=challenge.get('evidence', ''),
                    category="conceptual",
                    confidence=0.9
                ))

        # 5. Question exploration pairs
        for question in session.emerging_questions[:5]:
            qa_pairs.append(LearningQAPair(
                question=f"What questions remain about {session.topic}?",
                answer=question,
                context="Areas requiring further exploration",
                category="exploratory",
                confidence=0.7
            ))

        return qa_pairs

    def finalize_session(self, tags: List[str] = None) -> Path:
        """Finalize and save the session."""
        if not self.current_session:
            raise ValueError("No active session")

        session = self.current_session

        # Complete metadata
        session.completed_at = datetime.now().isoformat()
        if self._start_time:
            session.duration_seconds = (
                datetime.now() - self._start_time).total_seconds()
        session.tags = tags or []

        # Auto-generate Q&A pairs if not already done
        if not session.qa_pairs:
            session.qa_pairs = self.generate_qa_pairs()

        # Save session
        session_dir = self.sessions_dir / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save main session file
        session_file = session_dir / "session.json"
        with open(session_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2, default=str)

        # Save individual components for easier access

        # Study notes
        notes_file = session_dir / "study_notes.json"
        with open(notes_file, 'w') as f:
            json.dump([asdict(n)
                      for n in session.study_notes], f, indent=2, default=str)

        # Thought traces (for understanding reasoning)
        traces_file = session_dir / "thought_traces.json"
        with open(traces_file, 'w') as f:
            json.dump([asdict(t) for t in session.thought_traces],
                      f, indent=2, default=str)

        # Insights
        insights_file = session_dir / "insights.json"
        with open(insights_file, 'w') as f:
            json.dump([asdict(i)
                      for i in session.insights], f, indent=2, default=str)

        # Q&A pairs for training (multiple formats)
        qa_file = session_dir / "qa_pairs.json"
        with open(qa_file, 'w') as f:
            json.dump([asdict(qa)
                      for qa in session.qa_pairs], f, indent=2, default=str)

        # Training-ready format (alpaca style)
        training_file = session_dir / "training_data.jsonl"
        with open(training_file, 'w') as f:
            for qa in session.qa_pairs:
                f.write(json.dumps(qa.to_training_format()) + "\n")

        # Summary markdown
        summary_file = session_dir / "summary.md"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary_markdown(session))

        return session_dir

    def _generate_summary_markdown(self, session: LearningSession) -> str:
        """Generate a human-readable summary."""
        md = f"""# Learning Session: {session.topic}

**Session ID:** {session.session_id}
**Created:** {session.created_at}
**Completed:** {session.completed_at}
**Duration:** {session.duration_seconds / 60:.1f} minutes

## Goal
{session.goal}

## Statistics
- Papers discovered: {len(session.papers_discovered)}
- Papers studied: {session.total_papers_studied}
- Knowledge bits extracted: {session.total_knowledge_bits}
- Insights recorded: {len(session.insights)}
- Q&A pairs generated: {len(session.qa_pairs)}
- Tokens used: {session.total_tokens_used}

## Key Conclusions
"""
        for conclusion in session.conclusions:
            md += f"- {conclusion}\n"

        md += "\n## Synthesis\n"
        md += session.synthesis + "\n" if session.synthesis else "No synthesis recorded.\n"

        if session.challenges:
            md += "\n## Beliefs Challenged\n"
            for challenge in session.challenges:
                md += f"- **Prior:** {challenge.get('prior_belief', challenge.get('challenged_belief', 'N/A'))}\n"
                md += f"  **New:** {challenge.get('new_understanding', challenge.get('new_perspective', 'N/A'))}\n\n"

        if session.emerging_questions:
            md += "\n## Questions for Future Exploration\n"
            for q in session.emerging_questions[:10]:
                md += f"- {q}\n"

        if session.knowledge_gaps:
            md += "\n## Knowledge Gaps Identified\n"
            for gap in session.knowledge_gaps[:10]:
                md += f"- {gap}\n"

        md += f"\n## Files in This Session\n"
        md += f"- `session.json` - Complete session data\n"
        md += f"- `study_notes.json` - Notes from studying papers\n"
        md += f"- `thought_traces.json` - Reasoning traces (prompts/responses)\n"
        md += f"- `insights.json` - Recorded insights and realizations\n"
        md += f"- `qa_pairs.json` - Generated Q&A pairs\n"
        md += f"- `training_data.jsonl` - Ready for fine-tuning (Alpaca format)\n"

        return md

    def export_all_training_data(self, output_file: Optional[Path] = None) -> Path:
        """Export all Q&A pairs from all sessions into one training file."""
        if output_file is None:
            output_file = self.data_dir / self.actor_name / "all_training_data.jsonl"

        all_qa = []

        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                qa_file = session_dir / "qa_pairs.json"
                if qa_file.exists():
                    with open(qa_file, 'r') as f:
                        qa_data = json.load(f)
                        for qa in qa_data:
                            qa_pair = LearningQAPair(**qa)
                            all_qa.append(qa_pair.to_training_format())

        with open(output_file, 'w') as f:
            for qa in all_qa:
                f.write(json.dumps(qa) + "\n")

        return output_file

    def list_sessions(self) -> List[Dict]:
        """List all learning sessions."""
        sessions = []

        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    try:
                        with open(session_file, 'r') as f:
                            data = json.load(f)
                            sessions.append({
                                "session_id": data['session_id'],
                                "topic": data['topic'],
                                "goal": data['goal'],
                                "created_at": data['created_at'],
                                "papers_studied": data['total_papers_studied'],
                                "qa_pairs": len(data.get('qa_pairs', []))
                            })
                    except Exception:
                        continue

        return sorted(sessions, key=lambda x: x['created_at'], reverse=True)

    def load_session(self, session_id: str) -> LearningSession:
        """Load a session by ID."""
        session_file = self.sessions_dir / session_id / "session.json"

        if not session_file.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(session_file, 'r') as f:
            data = json.load(f)

        # Convert back to dataclasses (simplified - just use dict for now)
        session = LearningSession(
            session_id=data['session_id'],
            actor_name=data['actor_name'],
            created_at=data['created_at'],
            topic=data['topic'],
            goal=data['goal']
        )

        # Copy remaining fields
        for key, value in data.items():
            if hasattr(session, key):
                setattr(session, key, value)

        return session
