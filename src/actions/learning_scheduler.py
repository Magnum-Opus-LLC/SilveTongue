"""
Learning Schedule Runner

Manages and executes learning schedules - processing papers,
tracking progress, and integrating knowledge into the actor's knowledge base.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import shutil

from src.utils.paper_parser import paper_parser
from src.utils.knowledge_base import KnowledgeBase
from src.actions.study_processor import StudyProcessor


class LearningScheduleRunner:
    """
    Runs learning schedules to process papers and integrate knowledge.

    Usage:
        runner = LearningScheduleRunner('student')
        runner.create_schedule('ai_papers', '/path/to/papers', focus_areas=['machine learning'])
        runner.run_schedule('ai_papers')
    """

    def __init__(self, actor_name: str = 'template'):
        self.actor_name = actor_name
        self.project_root = Path(__file__).parent.parent.parent
        self.actor_root = self.project_root / 'actors' / actor_name
        self.schedule_root = self.actor_root / 'learning_schedule'
        self.schedules_dir = self.schedule_root / 'schedules'

        # Ensure directories exist
        self.schedules_dir.mkdir(parents=True, exist_ok=True)
        (self.schedule_root / 'notes').mkdir(exist_ok=True)
        (self.schedule_root / 'knowledge').mkdir(exist_ok=True)

        self.kb = KnowledgeBase(actor_name)
        self.study_processor = StudyProcessor(actor_name)

    def create_schedule(self, schedule_id: str, papers_directory: str,
                        name: str = None, description: str = None,
                        file_types: List[str] = None,
                        focus_areas: List[str] = None,
                        extraction_targets: List[Dict] = None,
                        depth: str = 'comprehensive',
                        recursive: bool = True) -> Dict:
        """
        Create a new learning schedule.

        Args:
            schedule_id: Unique identifier for this schedule
            papers_directory: Path to directory containing papers
            name: Human-readable name
            description: Description of what this schedule covers
            file_types: File extensions to process (default: pdf, txt, md)
            focus_areas: Areas to focus on during extraction
            extraction_targets: Types of knowledge to extract
            depth: Study depth (quick, standard, comprehensive)
            recursive: Whether to scan subdirectories

        Returns:
            Created schedule configuration
        """
        papers_path = Path(papers_directory)
        if not papers_path.exists():
            raise FileNotFoundError(
                f"Papers directory not found: {papers_directory}")

        # Scan for papers
        if file_types is None:
            file_types = ['.pdf', '.txt', '.md']

        papers = paper_parser.scan_directory(
            papers_path, file_types, recursive)

        if not papers:
            raise ValueError(
                f"No papers found in {papers_directory} with types {file_types}")

        # Default extraction targets
        if extraction_targets is None:
            extraction_targets = [
                {"type": "concept",
                    "description": "Core concepts, definitions, and terminology"},
                {"type": "method",
                    "description": "Methods, techniques, algorithms, and approaches"},
                {"type": "finding", "description": "Key findings, results, and conclusions"},
                {"type": "application",
                    "description": "Practical applications and use cases"},
                {"type": "reference", "description": "Important citations and related work"}
            ]

        if focus_areas is None:
            focus_areas = ["key concepts", "methodology",
                           "findings", "applications"]

        schedule = {
            "meta": {
                "schedule_id": schedule_id,
                "name": name or f"Learning Schedule: {schedule_id}",
                "description": description or f"Study papers from {papers_directory}",
                "created_at": datetime.now().isoformat(),
                "status": "created"
            },
            "source": {
                "directory": str(papers_path.absolute()),
                "file_types": file_types,
                "recursive": recursive
            },
            "study_config": {
                "depth": depth,
                "focus_areas": focus_areas,
                "extraction_targets": extraction_targets,
                "max_knowledge_bits_per_paper": 20,
                "generate_summary": True,
                "generate_questions": True,
                "analyze_enlightenment": True
            },
            "output": {
                "notes_dir": f"notes/{schedule_id}",
                "knowledge_dir": f"knowledge/{schedule_id}",
                "format": "json"
            },
            "progress": {
                "total_papers": len(papers),
                "papers_processed": 0,
                "papers_remaining": [str(p) for p in papers],
                "papers_completed": [],
                "papers_failed": [],
                "last_processed": None
            }
        }

        # Save schedule
        schedule_path = self.schedules_dir / f"{schedule_id}.json"
        with open(schedule_path, 'w') as f:
            json.dump(schedule, f, indent=2)

        print(f"✅ Created schedule '{schedule_id}' with {len(papers)} papers")
        print(f"   Saved to: {schedule_path}")

        return schedule

    def load_schedule(self, schedule_id: str) -> Dict:
        """Load a schedule by ID."""
        schedule_path = self.schedules_dir / f"{schedule_id}.json"

        if not schedule_path.exists():
            raise FileNotFoundError(f"Schedule not found: {schedule_id}")

        with open(schedule_path, 'r') as f:
            return json.load(f)

    def save_schedule(self, schedule: Dict):
        """Save schedule to disk."""
        schedule_id = schedule['meta']['schedule_id']
        schedule_path = self.schedules_dir / f"{schedule_id}.json"

        with open(schedule_path, 'w') as f:
            json.dump(schedule, f, indent=2)

    def list_schedules(self) -> List[Dict]:
        """List all available schedules."""
        schedules = []

        for schedule_file in self.schedules_dir.glob('*.json'):
            if schedule_file.name == 'example.json':
                continue
            try:
                with open(schedule_file, 'r') as f:
                    schedule = json.load(f)
                    schedules.append({
                        "id": schedule['meta']['schedule_id'],
                        "name": schedule['meta']['name'],
                        "status": schedule['meta']['status'],
                        "progress": f"{schedule['progress']['papers_processed']}/{schedule['progress']['total_papers']}"
                    })
            except Exception as e:
                print(f"Warning: Could not load {schedule_file}: {e}")

        return schedules

    def run_schedule(self, schedule_id: str, max_papers: int = None,
                     continue_on_error: bool = True) -> Dict:
        """
        Run a learning schedule.

        Args:
            schedule_id: ID of schedule to run
            max_papers: Maximum number of papers to process (None = all)
            continue_on_error: Whether to continue if a paper fails

        Returns:
            Summary of the run
        """
        schedule = self.load_schedule(schedule_id)

        print(f"\n{'='*60}")
        print(f"📚 Running Learning Schedule: {schedule['meta']['name']}")
        print(f"{'='*60}")

        schedule['meta']['status'] = 'running'
        self.save_schedule(schedule)

        papers_to_process = schedule['progress']['papers_remaining']

        if max_papers:
            papers_to_process = papers_to_process[:max_papers]

        if not papers_to_process:
            print("✅ All papers already processed!")
            return {"status": "complete", "processed": 0}

        print(f"📄 Papers to process: {len(papers_to_process)}")
        print()

        results = []

        for i, paper_path in enumerate(papers_to_process):
            print(
                f"\n[{i+1}/{len(papers_to_process)}] Processing: {Path(paper_path).name}")

            try:
                # Study the paper
                study_result = self.study_processor.study_paper(
                    Path(paper_path),
                    schedule['study_config']
                )

                # Save study results
                saved_paths = self.study_processor.save_study_result(
                    study_result,
                    schedule_id
                )

                # Integrate into knowledge base
                self._integrate_knowledge(study_result, schedule_id)

                # Get enlightenment info for progress tracking
                enlightenment = study_result.get('enlightenment', {})
                enlightenment_category = enlightenment.get(
                    'enlightenment_category') if enlightenment else None
                value_score = enlightenment.get(
                    'value_score') if enlightenment else None

                # Update progress
                schedule['progress']['papers_remaining'].remove(paper_path)
                schedule['progress']['papers_completed'].append({
                    "path": paper_path,
                    "processed_at": datetime.now().isoformat(),
                    "knowledge_bits": len(study_result['extraction']['knowledge_bits']),
                    "enlightenment_category": enlightenment_category,
                    "value_score": value_score,
                    "notes_file": str(saved_paths['notes']),
                    "knowledge_file": str(saved_paths['knowledge'])
                })
                schedule['progress']['papers_processed'] += 1
                schedule['progress']['last_processed'] = datetime.now().isoformat()

                self.save_schedule(schedule)

                results.append({
                    "paper": paper_path,
                    "status": "success",
                    "knowledge_bits": len(study_result['extraction']['knowledge_bits']),
                    "enlightenment": enlightenment_category,
                    "value_score": value_score
                })

                print(f"   ✅ Completed!")

            except Exception as e:
                print(f"   ❌ Error: {e}")

                schedule['progress']['papers_remaining'].remove(paper_path)
                schedule['progress']['papers_failed'].append({
                    "path": paper_path,
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                })
                self.save_schedule(schedule)

                results.append({
                    "paper": paper_path,
                    "status": "failed",
                    "error": str(e)
                })

                if not continue_on_error:
                    break

        # Update status
        if not schedule['progress']['papers_remaining']:
            schedule['meta']['status'] = 'completed'
        else:
            schedule['meta']['status'] = 'paused'

        self.save_schedule(schedule)

        # Calculate enlightenment stats
        enlightenment_counts = {}
        for r in results:
            if r.get('enlightenment'):
                cat = r['enlightenment']
                enlightenment_counts[cat] = enlightenment_counts.get(
                    cat, 0) + 1

        avg_value = 0
        value_scores = [r.get('value_score', 0)
                        for r in results if r.get('value_score')]
        if value_scores:
            avg_value = sum(value_scores) / len(value_scores)

        # Record learning event
        enlightenment_summary = ", ".join([f"{v} {k}" for k, v in enlightenment_counts.items(
        )]) if enlightenment_counts else "no analysis"
        self.kb.record_learning_event(
            event_type='study_session',
            description=f"Completed study session for schedule '{schedule_id}'",
            insights=[
                f"Processed {len([r for r in results if r['status'] == 'success'])} papers",
                f"Extracted {sum(r.get('knowledge_bits', 0) for r in results)} knowledge bits",
                f"Enlightenment breakdown: {enlightenment_summary}",
                f"Average value score: {avg_value:.1f}/10"
            ]
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 Study Session Complete")
        print(f"{'='*60}")
        print(
            f"   ✅ Successful: {len([r for r in results if r['status'] == 'success'])}")
        print(
            f"   ❌ Failed: {len([r for r in results if r['status'] == 'failed'])}")
        print(
            f"   📝 Total knowledge bits: {sum(r.get('knowledge_bits', 0) for r in results)}")
        print(
            f"   📈 Schedule progress: {schedule['progress']['papers_processed']}/{schedule['progress']['total_papers']}")

        # Print enlightenment summary
        if enlightenment_counts:
            print(f"\n   💡 Enlightenment Analysis:")
            emoji_map = {'CONFIRMS': '✅', 'EXPANDS': '🌱', 'CHALLENGES': '⚡',
                         'REVOLUTIONARY': '🚀', 'LOW_VALUE': '📉'}
            for cat, count in sorted(enlightenment_counts.items()):
                emoji = emoji_map.get(cat, '❓')
                print(f"      {emoji} {cat}: {count}")
            print(f"      📊 Avg Value Score: {avg_value:.1f}/10")

        return {
            "status": schedule['meta']['status'],
            "processed": len(results),
            "successful": len([r for r in results if r['status'] == 'success']),
            "failed": len([r for r in results if r['status'] == 'failed']),
            "enlightenment_summary": enlightenment_counts,
            "avg_value_score": avg_value,
            "results": results
        }

    def _integrate_knowledge(self, study_result: Dict, schedule_id: str):
        """
        Integrate extracted knowledge into the actor's knowledge base.

        This updates:
        - Learning book with the study session
        - Knowledge evolution records
        - Enlightenment-based updates (confirmations, challenges, expansions)
        """
        bits = study_result['extraction']['knowledge_bits']
        notes = study_result.get('study_notes', {})
        enlightenment = study_result.get('enlightenment', {})
        paper_name = study_result['paper']['file_name']

        # Record in learning book
        insights = []
        mindset_changes = []

        # Extract key concepts as insights
        if notes:
            for concept in notes.get('key_concepts', [])[:5]:
                insights.append(
                    f"Learned concept: {concept.get('concept', 'Unknown')}")

            for contrib in notes.get('main_contributions', [])[:3]:
                insights.append(f"Key contribution: {contrib}")

        # Count by type
        type_counts = {}
        for bit in bits:
            t = bit.get('type', 'other')
            type_counts[t] = type_counts.get(t, 0) + 1

        type_summary = ", ".join([f"{v} {k}s" for k, v in type_counts.items()])
        mindset_changes.append(
            f"Extracted {len(bits)} knowledge bits ({type_summary})")

        # Process enlightenment analysis
        if enlightenment:
            category = enlightenment.get('enlightenment_category', 'UNKNOWN')
            value_score = enlightenment.get('value_score', 0)

            insights.append(
                f"Enlightenment: {category} (value: {value_score}/10)")

            # Handle confirmations - strengthen existing beliefs
            for conf in enlightenment.get('confirmations', []):
                existing = conf.get('existing_belief', '')
                evidence = conf.get('paper_evidence', '')
                if existing:
                    insights.append(f"✅ Confirmed: {existing[:80]}")
                    self.kb.record_knowledge_evolution(
                        'principles_refined',
                        f"[CONFIRMED by {paper_name}] {existing[:100]}"
                    )

            # Handle expansions - add to understanding
            for exp in enlightenment.get('expansions', []):
                new_persp = exp.get('new_perspective', '')
                if new_persp:
                    insights.append(f"🌱 New perspective: {new_persp[:80]}")
                    self.kb.record_knowledge_evolution(
                        'experiences_learned',
                        f"[EXPANDED by {paper_name}] {new_persp[:100]}"
                    )

            # Handle challenges - note beliefs under question
            for chal in enlightenment.get('challenges', []):
                challenged = chal.get('challenged_belief', '')
                severity = chal.get('severity', 'minor')
                if challenged:
                    insights.append(
                        f"⚡ Challenge ({severity}): {challenged[:80]}")
                    self.kb.record_knowledge_evolution(
                        'challenges_encountered',
                        f"[CHALLENGED by {paper_name}] ({severity}) {challenged[:100]}"
                    )

            # Handle revolutionary ideas
            for rev in enlightenment.get('revolutionary_ideas', []):
                paradigm = rev.get('new_paradigm', '')
                recommendation = rev.get('adoption_recommendation', '')
                if paradigm:
                    insights.append(
                        f"🚀 Revolutionary: {paradigm[:80]} - {recommendation}")
                    self.kb.record_knowledge_evolution(
                        'principles_added',
                        f"[NEW PARADIGM from {paper_name}] ({recommendation}) {paradigm[:100]}"
                    )

            # Handle low value papers
            if category == 'LOW_VALUE':
                reasons = enlightenment.get('low_value_reasons', [])
                if reasons:
                    insights.append(f"📉 Low value: {reasons[0][:80]}")

            # Apply mindset updates if recommended
            mindset_update = enlightenment.get('mindset_update', {})
            if mindset_update.get('should_update'):
                for change in mindset_update.get('suggested_changes', [])[:3]:
                    mindset_changes.append(f"Mindset shift: {change}")

        # Record the learning event
        self.kb.record_learning_event(
            event_type='paper_study',
            description=f"Studied paper: {paper_name}",
            query=f"Schedule: {schedule_id}",
            insights=insights,
            mindset_changes=mindset_changes
        )

        # Record knowledge evolution for high-confidence findings
        for bit in bits:
            if bit.get('confidence') == 'high' and bit.get('type') in ['finding', 'method']:
                self.kb.record_knowledge_evolution(
                    'experiences_learned',
                    f"[{paper_name}] {bit.get('title', 'Unknown')}: {bit.get('description', '')[:100]}"
                )

    def get_schedule_knowledge(self, schedule_id: str) -> List[Dict]:
        """Get all knowledge bits from a schedule."""
        knowledge_dir = self.schedule_root / 'knowledge' / schedule_id

        if not knowledge_dir.exists():
            return []

        all_bits = []

        for knowledge_file in knowledge_dir.glob('*_knowledge.json'):
            with open(knowledge_file, 'r') as f:
                data = json.load(f)
                all_bits.extend(data.get('knowledge_bits', []))

        return all_bits

    def search_knowledge(self, schedule_id: str, query: str = None,
                         knowledge_type: str = None,
                         confidence: str = None) -> List[Dict]:
        """
        Search knowledge bits with filters.

        Args:
            schedule_id: Schedule to search
            query: Text to search in title/description
            knowledge_type: Filter by type (concept, method, finding, etc.)
            confidence: Filter by confidence level
        """
        all_bits = self.get_schedule_knowledge(schedule_id)

        results = []

        for bit in all_bits:
            # Apply filters
            if knowledge_type and bit.get('type') != knowledge_type:
                continue

            if confidence and bit.get('confidence') != confidence:
                continue

            if query:
                query_lower = query.lower()
                searchable = f"{bit.get('title', '')} {bit.get('description', '')}".lower(
                )
                if query_lower not in searchable:
                    continue

            results.append(bit)

        return results

    def export_knowledge_summary(self, schedule_id: str, output_path: str = None) -> str:
        """
        Export a markdown summary of all knowledge from a schedule.
        """
        schedule = self.load_schedule(schedule_id)
        all_bits = self.get_schedule_knowledge(schedule_id)

        # Group by type
        by_type = {}
        for bit in all_bits:
            t = bit.get('type', 'other')
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(bit)

        # Generate markdown
        lines = [
            f"# Knowledge Summary: {schedule['meta']['name']}",
            f"\n*Generated: {datetime.now().isoformat()}*",
            f"\n**Total Knowledge Bits:** {len(all_bits)}",
            f"\n**Papers Processed:** {schedule['progress']['papers_processed']}",
            "\n---\n"
        ]

        for knowledge_type, bits in sorted(by_type.items()):
            lines.append(f"\n## {knowledge_type.title()}s ({len(bits)})\n")

            for bit in bits:
                confidence_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(
                    bit.get('confidence', 'unknown'), '⚪'
                )

                lines.append(
                    f"### {confidence_emoji} {bit.get('title', 'Untitled')}")
                lines.append(f"\n{bit.get('description', 'No description')}\n")

                if bit.get('key_terms'):
                    lines.append(
                        f"**Key terms:** {', '.join(bit['key_terms'])}\n")

                if bit.get('quote'):
                    lines.append(f"> {bit['quote']}\n")

                lines.append(
                    f"*Source: {Path(bit.get('source_file', 'Unknown')).name}*\n")
                lines.append("---\n")

        content = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            print(f"Exported to: {output_path}")

        return content

    def discover_papers(self, query: str, depth: int = 2,
                        year_from: int = None, year_to: int = None,
                        min_citations: int = None) -> List[Dict]:
        """
        Discover papers on a topic using Google Scholar.

        Args:
            query: Search query
            depth: Number of pages to search (10 results each)
            year_from: Filter papers from this year
            year_to: Filter papers up to this year
            min_citations: Minimum citation count

        Returns:
            List of discovered papers with metadata
        """
        from src.scholar_search import ScholarSearcher

        print(f"\n🔍 Discovering papers: {query}")

        searcher = ScholarSearcher()
        papers = searcher.search_advanced(
            query=query,
            year_from=year_from,
            year_to=year_to,
            depth=depth
        )

        # Filter by citations if specified
        if min_citations:
            papers = [p for p in papers if p.citation_count >= min_citations]

        print(f"   Found {len(papers)} papers")

        # Record the discovery
        self.kb.record_learning_event(
            event_type='paper_discovery',
            description=f"Discovered {len(papers)} papers on: {query}",
            insights=[
                f"Top cited: {papers[0].title if papers else 'None'} ({papers[0].citation_count if papers else 0} citations)",
                f"Year range: {year_from or 'any'}-{year_to or 'any'}",
                f"Min citations filter: {min_citations or 'none'}"
            ]
        )

        return [p.to_dict() for p in papers]

    def get_research_survey(self, query: str, depth: int = 2) -> Dict:
        """
        Get a research survey for a topic using Google Scholar.

        Returns:
            Survey with top papers, word analysis, and citation distribution
        """
        from src.scholar_search import ScholarSearcher

        print(f"\n📊 Research survey: {query}")

        searcher = ScholarSearcher()
        survey = searcher.get_survey(query, depth=depth)

        # Convert papers to dicts
        survey['top_papers'] = [p.to_dict()
                                for p in survey.get('top_papers', [])]

        print(f"   Found {len(survey.get('top_papers', []))} top papers")
        print(
            f"   Identified {len(survey.get('word_analysis', []))} key terms")

        return survey


# Convenience function
def create_learning_schedule(actor_name: str, schedule_id: str,
                             papers_directory: str, **kwargs) -> Dict:
    """Quick helper to create a learning schedule."""
    runner = LearningScheduleRunner(actor_name)
    return runner.create_schedule(schedule_id, papers_directory, **kwargs)


def run_learning_schedule(actor_name: str, schedule_id: str, **kwargs) -> Dict:
    """Quick helper to run a learning schedule."""
    runner = LearningScheduleRunner(actor_name)
    return runner.run_schedule(schedule_id, **kwargs)


def discover_papers(actor_name: str, query: str, **kwargs) -> List[Dict]:
    """Quick helper to discover papers via Scholar search."""
    runner = LearningScheduleRunner(actor_name)
    return runner.discover_papers(query, **kwargs)
