#!/usr/bin/env python3
"""
Learning Session Analyzer
=========================
Comprehensive analysis of an actor's learning sessions, topics, papers, and insights.

Usage:
    python -m src.report_tool.session_analyzer [actor_name] [--output report.md] [--json]

Examples:
    python -m src.report_tool.session_analyzer steve
    python -m src.report_tool.session_analyzer steve --output steve_report.md
    python -m src.report_tool.session_analyzer steve --json > analysis.json
"""

import json
import os
import re
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any


class LearningSessionAnalyzer:
    """Analyzes learning sessions from an actor's data directory"""

    # Common buzzwords to track in AI/ML research
    BUZZWORDS = [
        'transformer', 'attention', 'edge', 'ai', 'latency', 'model',
        'quantization', 'pruning', 'sparsity', 'drift', 'continual',
        'learning', 'neural', 'network', 'deep', 'inference', 'training',
        'optimization', 'efficiency', 'hardware', 'accelerator', 'gpu',
        'tensor', 'graph', 'robustness', 'reliability', 'scalability',
        'trade-off', 'trade‑off', 'benchmark', 'evaluation', 'metric',
        'long-term', 'long‑term', 'optimal', 'impact', 'effects',
        'llm', 'gpt', 'bert', 'vision', 'multimodal', 'embedding',
        'fine-tuning', 'pre-training', 'self-supervised', 'contrastive'
    ]

    # Topic cluster definitions
    TOPIC_CLUSTERS = {
        'edge_ai': ['edge', 'iot', 'embedded', 'mobile', 'device', 'on-device'],
        'transformers': ['transformer', 'attention', 'bert', 'gpt', 'language model', 'llm'],
        'optimization': ['quantization', 'pruning', 'compression', 'sparsity', 'efficient'],
        'reliability': ['reliability', 'robustness', 'fault', 'drift', 'stability', 'safety'],
        'latency': ['latency', 'real-time', 'real‑time', 'inference', 'speed', 'throughput'],
        'continual_learning': ['continual', 'lifelong', 'incremental', 'online learning', 'streaming'],
        'hardware': ['hardware', 'accelerator', 'gpu', 'tpu', 'fpga', 'tensor', 'asic'],
        'benchmarks': ['benchmark', 'evaluation', 'metric', 'comparison', 'standardized'],
        'vision': ['vision', 'image', 'video', 'visual', 'cnn', 'detection'],
        'nlp': ['nlp', 'language', 'text', 'sentiment', 'translation', 'summarization']
    }

    def __init__(self, actor_name: str, base_dir: str = "array/data"):
        self.actor_name = actor_name
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / actor_name / "learning_sessions"
        self.sessions: List[Dict] = []
        self.analysis_results: Dict[str, Any] = {}

    def load_sessions(self) -> int:
        """Load all learning sessions. Returns count of loaded sessions."""
        if not self.sessions_dir.exists():
            raise FileNotFoundError(
                f"Sessions directory not found: {self.sessions_dir}")

        session_dirs = sorted([
            d for d in self.sessions_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        for session_dir in session_dirs:
            session_file = session_dir / 'session.json'
            insights_file = session_dir / 'insights.json'

            if session_file.exists():
                try:
                    with open(session_file) as f:
                        session = json.load(f)

                    session['session_dir'] = str(session_dir.name)

                    if insights_file.exists():
                        with open(insights_file) as f:
                            session['insights'] = json.load(f)

                    self.sessions.append(session)
                except Exception as e:
                    print(f"Warning: Error loading {session_file}: {e}")

        return len(self.sessions)

    def analyze_all(self) -> Dict[str, Any]:
        """Run all analyses and return combined results"""
        self.analysis_results = {
            'actor_name': self.actor_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': self._analyze_summary(),
            'topics': self._analyze_topics(),
            'papers': self._analyze_papers(),
            'insights': self._analyze_insights(),
            'temporal': self._analyze_temporal(),
            'clustering': self._analyze_clustering(),
            'feedback_loops': self._analyze_feedback_loops(),
            'knowledge_growth': self._analyze_knowledge_growth()
        }
        return self.analysis_results

    def _analyze_summary(self) -> Dict:
        """High-level summary statistics"""
        total_papers = sum(len(s.get('papers_discovered', []))
                           for s in self.sessions)
        total_studied = sum(len(s.get('papers_studied', []))
                            for s in self.sessions)
        total_insights = sum(len(s.get('insights', [])) for s in self.sessions)

        return {
            'total_sessions': len(self.sessions),
            'total_papers_discovered': total_papers,
            'total_papers_studied': total_studied,
            'total_insights': total_insights,
            'study_rate': f"{(total_studied/total_papers*100):.1f}%" if total_papers > 0 else "N/A"
        }

    def _analyze_topics(self) -> Dict:
        """Analyze topic patterns and buzzword distribution"""
        topics = [s.get('topic', '') for s in self.sessions if s.get('topic')]

        # Buzzword frequency
        buzzword_counts = Counter()
        for topic in topics:
            topic_lower = topic.lower()
            for bw in self.BUZZWORDS:
                if bw in topic_lower:
                    buzzword_counts[bw] += 1

        # Topic length analysis
        topic_lengths = [len(t.split()) for t in topics]

        # Common starting phrases
        starting_phrases = Counter()
        for topic in topics:
            words = topic.split()[:3]
            if len(words) >= 3:
                starting_phrases[' '.join(words).lower()] += 1

        return {
            'total_unique_topics': len(set(topics)),
            'avg_topic_length_words': sum(topic_lengths) / len(topic_lengths) if topic_lengths else 0,
            'buzzword_frequency': dict(buzzword_counts.most_common(25)),
            'top_starting_phrases': dict(starting_phrases.most_common(10)),
            'buzzword_concentration': sum(buzzword_counts.values()) / len(topics) if topics else 0
        }

    def _analyze_papers(self) -> Dict:
        """Analyze discovered and studied papers"""
        all_papers = []
        sources = Counter()
        citations = []
        years = Counter()

        for session in self.sessions:
            for paper in session.get('papers_discovered', []):
                all_papers.append(paper)
                sources[paper.get('source', 'unknown')] += 1

                if paper.get('citation_count'):
                    try:
                        citations.append(int(paper['citation_count']))
                    except (ValueError, TypeError):
                        pass

                year = paper.get('year', '')
                if year and str(year).isdigit():
                    years[str(year)] += 1

        # Title word analysis
        title_words = Counter()
        stop_words = {'the', 'a', 'an', 'of', 'for', 'in', 'on',
                      'and', 'to', 'with', 'from', 'by', 'is', 'are', 'using'}

        for paper in all_papers:
            title = paper.get('title', '').lower()
            words = re.findall(r'\b[a-z]{3,}\b', title)
            for word in words:
                if word not in stop_words:
                    title_words[word] += 1

        return {
            'total_papers': len(all_papers),
            'unique_sources': len(sources),
            'top_sources': dict(sources.most_common(15)),
            'citation_stats': {
                'total': sum(citations),
                'average': sum(citations) / len(citations) if citations else 0,
                'max': max(citations) if citations else 0,
                'min': min(citations) if citations else 0,
                'papers_with_citations': len(citations)
            },
            'publication_years': dict(years.most_common(10)),
            'top_title_words': dict(title_words.most_common(30))
        }

    def _analyze_insights(self) -> Dict:
        """Analyze insight types and content"""
        insight_types = Counter()
        challenges = []
        expansions = []
        revolutionary = []
        confirmations = []

        for session in self.sessions:
            for insight in session.get('insights', []):
                insight_type = insight.get('insight_type', 'unknown')
                insight_types[insight_type] += 1

                content = insight.get('content', '')
                if isinstance(content, dict):
                    if insight_type == 'challenge':
                        challenges.append(insight.get(
                            'prior_belief', content.get('challenged_belief', '')))
                    elif insight_type == 'expansion':
                        expansions.append(content.get('new_perspective', ''))
                    elif insight_type == 'revolutionary':
                        revolutionary.append(content.get('new_paradigm', ''))
                    elif insight_type == 'confirmation':
                        confirmations.append(
                            content.get('existing_belief', ''))
                elif insight_type == 'challenge':
                    challenges.append(insight.get(
                        'prior_belief', str(content)))

        total = sum(insight_types.values())

        return {
            'total': total,
            'distribution': dict(insight_types),
            'percentages': {k: f"{v/total*100:.1f}%" for k, v in insight_types.items()} if total > 0 else {},
            'challenges_count': len([c for c in challenges if c]),
            'expansions_count': len([e for e in expansions if e]),
            'revolutionary_count': len([r for r in revolutionary if r]),
            'sample_challenges': [c[:100] for c in challenges if c][:5],
            'sample_expansions': [e[:100] for e in expansions if e][:5],
            'sample_revolutionary': [r[:100] for r in revolutionary if r][:5]
        }

    def _analyze_temporal(self) -> Dict:
        """Analyze temporal patterns"""
        timestamps = []
        durations = []

        for session in self.sessions:
            created = session.get('created_at')
            completed = session.get('completed_at')

            if created:
                try:
                    ts = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    pass

            if created and completed:
                try:
                    start = datetime.fromisoformat(
                        created.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(
                        completed.replace('Z', '+00:00'))
                    duration = (end - start).total_seconds() / 60
                    if 0 < duration < 60:  # Reasonable range
                        durations.append(duration)
                except (ValueError, TypeError):
                    pass

        sessions_by_hour = Counter(ts.hour for ts in timestamps)
        sessions_by_day = Counter(ts.strftime('%Y-%m-%d') for ts in timestamps)

        return {
            'sessions_with_timestamps': len(timestamps),
            'duration_stats': {
                'average_minutes': sum(durations) / len(durations) if durations else 0,
                'min_minutes': min(durations) if durations else 0,
                'max_minutes': max(durations) if durations else 0,
                'total_hours': sum(durations) / 60 if durations else 0
            },
            'sessions_by_hour': dict(sorted(sessions_by_hour.items())),
            'sessions_by_day': dict(sorted(sessions_by_day.items())[-10:]),
            'time_range': {
                'first': timestamps[0].isoformat() if timestamps else None,
                'last': timestamps[-1].isoformat() if timestamps else None
            }
        }

    def _analyze_clustering(self) -> Dict:
        """Analyze topic clustering"""
        cluster_counts = defaultdict(list)

        for session in self.sessions:
            topic = session.get('topic', '').lower()
            matched = False

            for cluster_name, keywords in self.TOPIC_CLUSTERS.items():
                if any(kw in topic for kw in keywords):
                    cluster_counts[cluster_name].append(
                        session.get('session_dir', ''))
                    matched = True
                    break

            if not matched:
                cluster_counts['other'].append(session.get('session_dir', ''))

        return {
            'cluster_sizes': {k: len(v) for k, v in sorted(cluster_counts.items(), key=lambda x: -len(x[1]))},
            'cluster_percentages': {k: f"{len(v)/len(self.sessions)*100:.1f}%" for k, v in cluster_counts.items()},
            'dominant_cluster': max(cluster_counts.items(), key=lambda x: len(x[1]))[0] if cluster_counts else None
        }

    def _analyze_feedback_loops(self) -> Dict:
        """Detect feedback loops in topic generation"""
        all_focus_areas = []
        feedback_count = 0
        feedback_examples = []

        for session in self.sessions:
            all_focus_areas.extend(session.get('focus_areas', []))

        # Check for topic continuity
        for i, session in enumerate(self.sessions[:-1]):
            focus_areas = session.get('focus_areas', [])
            next_topic = self.sessions[i+1].get('topic', '').lower()

            for area in focus_areas:
                area_words = set(re.findall(r'\b\w{4,}\b', area.lower()))
                topic_words = set(re.findall(r'\b\w{4,}\b', next_topic))
                overlap = area_words & topic_words

                if len(overlap) >= 2:
                    feedback_count += 1
                    if len(feedback_examples) < 5:
                        feedback_examples.append({
                            'focus_area': area[:80],
                            'next_topic': self.sessions[i+1].get('topic', '')[:80],
                            'overlap_words': list(overlap)[:5]
                        })
                    break

        focus_area_counts = Counter(all_focus_areas)

        return {
            'total_focus_areas': len(all_focus_areas),
            'unique_focus_areas': len(set(all_focus_areas)),
            'feedback_loops_detected': feedback_count,
            'feedback_rate': f"{feedback_count/len(self.sessions)*100:.1f}%" if self.sessions else "0%",
            'top_recurring_focus_areas': [{'area': area[:60], 'count': cnt} for area, cnt in focus_area_counts.most_common(10)],
            'feedback_examples': feedback_examples
        }

    def _analyze_knowledge_growth(self) -> Dict:
        """Analyze knowledge growth over time"""
        cumulative_papers = 0
        cumulative_insights = 0
        growth_timeline = []

        for i, session in enumerate(self.sessions):
            papers = len(session.get('papers_discovered', []))
            insights = len(session.get('insights', []))
            cumulative_papers += papers
            cumulative_insights += insights

            if i % 10 == 0 or i == len(self.sessions) - 1:
                growth_timeline.append({
                    'session_index': i,
                    'cumulative_papers': cumulative_papers,
                    'cumulative_insights': cumulative_insights
                })

        if len(self.sessions) >= 2:
            first_half = self.sessions[:len(self.sessions)//2]
            second_half = self.sessions[len(self.sessions)//2:]

            first_half_insights = sum(len(s.get('insights', []))
                                      for s in first_half)
            second_half_insights = sum(
                len(s.get('insights', [])) for s in second_half)

            growth_pattern = 'accelerating' if second_half_insights > first_half_insights else 'decelerating'
        else:
            growth_pattern = 'insufficient data'

        return {
            'growth_timeline': growth_timeline,
            'growth_pattern': growth_pattern,
            'avg_papers_per_session': cumulative_papers / len(self.sessions) if self.sessions else 0,
            'avg_insights_per_session': cumulative_insights / len(self.sessions) if self.sessions else 0
        }

    def generate_report(self, format: str = 'text') -> str:
        """Generate a formatted report"""
        if not self.analysis_results:
            self.analyze_all()

        if format == 'json':
            return json.dumps(self.analysis_results, indent=2, default=str)
        elif format == 'markdown':
            return self._generate_markdown_report()
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate text report"""
        r = self.analysis_results
        lines = []

        lines.append("=" * 100)
        lines.append(f"LEARNING SESSION ANALYSIS: {r['actor_name'].upper()}")
        lines.append(f"   Generated: {r['analysis_timestamp']}")
        lines.append("=" * 100)

        # Summary
        s = r['summary']
        lines.append(f"\nSUMMARY")
        lines.append("-" * 100)
        lines.append(f"  Sessions: {s['total_sessions']}")
        lines.append(f"  Papers Discovered: {s['total_papers_discovered']}")
        lines.append(
            f"  Papers Studied: {s['total_papers_studied']} ({s['study_rate']})")
        lines.append(f"  Total Insights: {s['total_insights']}")

        # Topics
        t = r['topics']
        lines.append(f"\nTOPIC ANALYSIS")
        lines.append("-" * 100)
        lines.append(f"  Unique Topics: {t['total_unique_topics']}")
        lines.append(
            f"  Avg Topic Length: {t['avg_topic_length_words']:.1f} words")
        lines.append(
            f"  Buzzword Density: {t['buzzword_concentration']:.2f} per topic")
        lines.append(f"\n  Top Buzzwords:")
        for bw, count in list(t['buzzword_frequency'].items())[:15]:
            bar = '#' * (count // 2)
            lines.append(f"    {bw:20s}: {count:4d} {bar}")

        # Papers
        p = r['papers']
        lines.append(f"\nPAPER ANALYSIS")
        lines.append("-" * 100)
        lines.append(f"  Total Papers: {p['total_papers']}")
        lines.append(f"  Unique Sources: {p['unique_sources']}")
        lines.append(
            f"  Citation Stats: avg={p['citation_stats']['average']:.0f}, max={p['citation_stats']['max']:,}")
        lines.append(f"\n  Top Sources:")
        for src, count in list(p['top_sources'].items())[:10]:
            bar = '#' * (count // 20)
            lines.append(f"    {src:35s}: {count:4d} {bar}")

        # Insights
        i = r['insights']
        lines.append(f"\nINSIGHT ANALYSIS")
        lines.append("-" * 100)
        lines.append(f"  Total Insights: {i['total']}")
        for insight_type, pct in i['percentages'].items():
            count = i['distribution'].get(insight_type, 0)
            lines.append(f"    {insight_type:15s}: {count:5d} ({pct})")

        # Clustering
        c = r['clustering']
        lines.append(f"\nTOPIC CLUSTERING")
        lines.append("-" * 100)
        for cluster, size in c['cluster_sizes'].items():
            pct = c['cluster_percentages'].get(cluster, '0%')
            bar = '#' * (size // 3)
            lines.append(f"  {cluster:20s}: {size:4d} ({pct}) {bar}")

        # Knowledge Growth
        kg = r['knowledge_growth']
        lines.append(f"\nKNOWLEDGE GROWTH")
        lines.append("-" * 100)
        lines.append(
            f"  Avg Papers/Session: {kg['avg_papers_per_session']:.1f}")
        lines.append(
            f"  Avg Insights/Session: {kg['avg_insights_per_session']:.1f}")
        lines.append(f"  Growth Pattern: {kg['growth_pattern']}")

        lines.append("=" * 100)
        return '\n'.join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate markdown report"""
        r = self.analysis_results
        s = r['summary']
        t = r['topics']
        p = r['papers']
        i = r['insights']
        c = r['clustering']
        f = r['feedback_loops']
        tm = r['temporal']
        kg = r['knowledge_growth']

        return f"""# Learning Session Analysis: {r['actor_name']}

**Generated:** {r['analysis_timestamp']}

## Summary

| Metric | Value |
|--------|-------|
| Total Sessions | {s['total_sessions']} |
| Papers Discovered | {s['total_papers_discovered']} |
| Papers Studied | {s['total_papers_studied']} ({s['study_rate']}) |
| Total Insights | {s['total_insights']} |
| Learning Time | {tm['duration_stats']['total_hours']:.1f} hours |

## Topic Analysis

- **Unique Topics:** {t['total_unique_topics']}
- **Avg Topic Length:** {t['avg_topic_length_words']:.1f} words
- **Buzzword Density:** {t['buzzword_concentration']:.2f} per topic

### Top Buzzwords

| Buzzword | Count |
|----------|-------|
{chr(10).join([f"| {bw} | {cnt} |" for bw, cnt in list(t['buzzword_frequency'].items())[:15]])}

## Paper Analysis

- **Total Papers:** {p['total_papers']}
- **Unique Sources:** {p['unique_sources']}

### Top Sources

| Source | Papers |
|--------|--------|
{chr(10).join([f"| {src} | {cnt} |" for src, cnt in list(p['top_sources'].items())[:10]])}

## Insight Analysis

| Type | Count | Percentage |
|------|-------|------------|
{chr(10).join([f"| {k} | {v} | {i['percentages'].get(k, 'N/A')} |" for k, v in i['distribution'].items()])}

## Topic Clustering

| Cluster | Sessions | Percentage |
|---------|----------|------------|
{chr(10).join([f"| {cluster} | {size} | {c['cluster_percentages'].get(cluster, 'N/A')} |" for cluster, size in c['cluster_sizes'].items()])}

## Knowledge Growth

- **Avg Papers/Session:** {kg['avg_papers_per_session']:.1f}
- **Avg Insights/Session:** {kg['avg_insights_per_session']:.1f}
- **Growth Pattern:** {kg['growth_pattern']}

---
*Analysis complete*
"""


def main():
    parser = argparse.ArgumentParser(
        description='Analyze learning sessions for an actor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.report_tool.session_analyzer steve
  python -m src.report_tool.session_analyzer steve --output report.md --format markdown
  python -m src.report_tool.session_analyzer steve --json
        """
    )
    parser.add_argument('actor', help='Actor name (e.g., steve, student)')
    parser.add_argument('--base-dir', default='array/data',
                        help='Base data directory')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--format', '-f', choices=['text', 'markdown', 'json'], default='text',
                        help='Output format')
    parser.add_argument('--json', action='store_true',
                        help='Shortcut for --format json')

    args = parser.parse_args()

    if args.json:
        args.format = 'json'

    try:
        analyzer = LearningSessionAnalyzer(args.actor, args.base_dir)
        count = analyzer.load_sessions()
        print(
            f"Loaded {count} sessions for actor '{args.actor}'", file=sys.stderr)

        analyzer.analyze_all()
        report = analyzer.generate_report(args.format)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}", file=sys.stderr)
        else:
            print(report)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
