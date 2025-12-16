#!/usr/bin/env python3
"""
Knowledge Report Generator
==========================
Generates comprehensive markdown reports with wordcloud evolution.

Usage:
    python -m src.report_tool.report_generator [actor_name] [--output-dir dir]
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("Warning: wordcloud not installed. Install with: pip install wordcloud")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class KnowledgeReportGenerator:
    """Generate comprehensive knowledge reports with visualizations"""
    
    BUZZWORDS = [
        'novel', 'innovative', 'cutting-edge', 'state-of-the-art', 'groundbreaking',
        'revolutionary', 'paradigm', 'synergy', 'leverage', 'optimize', 'scalable',
        'robust', 'efficient', 'framework', 'methodology', 'architecture', 'pipeline',
        'end-to-end', 'holistic', 'comprehensive', 'systematic', 'empirical',
        'theoretical', 'practical', 'real-world', 'benchmark', 'baseline',
        'outperform', 'achieve', 'demonstrate', 'propose', 'present', 'introduce',
        'neural', 'deep', 'learning', 'model', 'network', 'layer', 'attention',
        'transformer', 'embedding', 'representation', 'feature', 'latent',
        'training', 'inference', 'optimization', 'gradient', 'loss', 'accuracy',
        'precision', 'recall', 'performance', 'evaluation', 'analysis', 'study',
        'research', 'contribution', 'insight', 'finding', 'result', 'conclusion',
        'future', 'direction', 'limitation', 'challenge', 'opportunity', 'potential',
        'integration', 'adaptation', 'transfer', 'generalization', 'specialization',
        'knowledge', 'understanding', 'reasoning', 'inference', 'prediction',
        'classification', 'regression', 'clustering', 'segmentation', 'detection',
        'generation', 'synthesis', 'reconstruction', 'compression', 'quantization',
        'pruning', 'distillation', 'augmentation', 'regularization', 'normalization',
        'edge', 'device', 'mobile', 'embedded', 'real-time', 'latency', 'throughput',
        'memory', 'computation', 'resource', 'constraint', 'trade-off', 'balance',
        'evidence', 'validation', 'verification', 'testing', 'deployment', 'production'
    ]
    
    DOMAINS = {
        'Edge AI': ['edge', 'iot', 'embedded', 'mobile', 'device', 'on-device', 'tinyml'],
        'Transformers/LLM': ['transformer', 'attention', 'bert', 'gpt', 'llm', 'language model'],
        'Optimization': ['quantization', 'pruning', 'compression', 'sparsity', 'efficient', 'distillation'],
        'Reliability': ['reliability', 'robustness', 'fault', 'drift', 'stability', 'safety', 'failure'],
        'Performance': ['latency', 'throughput', 'inference', 'speed', 'real-time'],
        'Continual Learning': ['continual', 'lifelong', 'incremental', 'online learning', 'streaming'],
        'Hardware': ['hardware', 'accelerator', 'gpu', 'tpu', 'fpga', 'tensor', 'asic'],
        'Computer Vision': ['vision', 'image', 'video', 'visual', 'cnn', 'detection'],
        'NLP': ['nlp', 'text', 'sentiment', 'translation', 'summarization'],
        'Graph/Tensor': ['graph', 'tensor', 'gnn', 'network', 'contraction']
    }
    
    def __init__(self, actor_name: str, base_dir: str = "array/data"):
        self.actor_name = actor_name
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / actor_name / "learning_sessions"
        self.sessions = []
        self.all_keywords = []
        self.timeline_data = []
        self.domain_first_seen = {}
        
    def load_data(self):
        """Load all learning sessions"""
        session_dirs = sorted([
            d for d in self.sessions_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        for idx, session_dir in enumerate(session_dirs):
            session_file = session_dir / 'session.json'
            insights_file = session_dir / 'insights.json'
            
            if session_file.exists():
                try:
                    with open(session_file) as f:
                        session = json.load(f)
                    
                    session['index'] = idx
                    
                    if insights_file.exists():
                        with open(insights_file) as f:
                            session['insights'] = json.load(f)
                    
                    self._extract_keywords(session)
                    self.sessions.append(session)
                except Exception as e:
                    print(f"Warning: {e}")
        
        self._build_timeline()
        return len(self.sessions)
    
    def _extract_keywords(self, session: dict):
        """Extract keywords from session"""
        text_parts = []
        
        text_parts.append(session.get('topic', ''))
        
        for paper in session.get('papers_discovered', []):
            text_parts.append(paper.get('title', ''))
            text_parts.append(paper.get('snippet', ''))
        
        for insight in session.get('insights', []):
            content = insight.get('content', '')
            if isinstance(content, dict):
                text_parts.extend(str(v) for v in content.values())
            else:
                text_parts.append(str(content))
        
        text = ' '.join(text_parts).lower()
        
        for word in self.BUZZWORDS:
            count = text.count(word.lower())
            self.all_keywords.extend([word] * count)
    
    def _classify_domains(self, text: str) -> list:
        """Classify text into domains"""
        text_lower = text.lower()
        domains = []
        for domain, keywords in self.DOMAINS.items():
            if any(kw in text_lower for kw in keywords):
                domains.append(domain)
        return domains if domains else ['General']
    
    def _build_timeline(self):
        """Build timeline data"""
        cumulative_domains = set()
        cumulative_papers = 0
        cumulative_insights = 0
        
        for session in self.sessions:
            idx = session['index']
            topic = session.get('topic', '')
            
            session_domains = self._classify_domains(topic)
            for d in session_domains:
                if d not in cumulative_domains:
                    self.domain_first_seen[d] = idx
                cumulative_domains.add(d)
            
            papers = len(session.get('papers_discovered', []))
            insights = len(session.get('insights', []))
            cumulative_papers += papers
            cumulative_insights += insights
            
            insight_types = Counter()
            for ins in session.get('insights', []):
                insight_types[ins.get('insight_type', 'unknown')] += 1
            
            self.timeline_data.append({
                'index': idx,
                'topic': topic[:60],
                'domains': list(cumulative_domains),
                'domain_count': len(cumulative_domains),
                'papers_cumulative': cumulative_papers,
                'insights_cumulative': cumulative_insights,
                'insight_types': dict(insight_types)
            })
    
    def generate_wordcloud_evolution(self, output_dir: str, num_frames: int = 8):
        """Generate wordcloud evolution frames"""
        if not HAS_WORDCLOUD or not HAS_MATPLOTLIB:
            print("wordcloud or matplotlib not available")
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_files = []
        frame_indices = np.linspace(0, len(self.sessions)-1, num_frames, dtype=int)
        
        cumulative_keywords = []
        keyword_idx = 0
        
        for frame_num, end_session_idx in enumerate(frame_indices):
            while keyword_idx <= end_session_idx:
                session = self.sessions[keyword_idx]
                text_parts = []
                text_parts.append(session.get('topic', ''))
                for paper in session.get('papers_discovered', []):
                    text_parts.append(paper.get('title', ''))
                for insight in session.get('insights', []):
                    content = insight.get('content', '')
                    if isinstance(content, dict):
                        text_parts.extend(str(v) for v in content.values())
                    else:
                        text_parts.append(str(content))
                
                text = ' '.join(text_parts).lower()
                for word in self.BUZZWORDS:
                    count = text.count(word.lower())
                    cumulative_keywords.extend([word] * count)
                
                keyword_idx += 1
            
            if cumulative_keywords:
                word_freq = Counter(cumulative_keywords)
                
                wc = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    colormap='viridis',
                    relative_scaling=0.5
                ).generate_from_frequencies(word_freq)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                
                session_info = self.sessions[end_session_idx]
                topic = session_info.get('topic', '')[:50]
                ax.set_title(
                    f"Session {end_session_idx + 1}/{len(self.sessions)}: {topic}...\n"
                    f"Keywords: {len(word_freq)} unique | Total mentions: {len(cumulative_keywords):,}",
                    fontsize=12, pad=10
                )
                
                frame_path = output_path / f"wordcloud_frame_{frame_num:02d}.png"
                plt.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                frame_files.append(str(frame_path))
                print(f"Generated frame {frame_num + 1}/{num_frames}")
        
        return frame_files
    
    def generate_markdown_report(self, output_path: str, wordcloud_frames: list = None):
        """Generate comprehensive markdown report"""
        lines = []
        
        # Header
        lines.append(f"# Knowledge Evolution Report: {self.actor_name.upper()}")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")
        
        # Executive Summary
        lines.append("---")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        
        total_papers = sum(len(s.get('papers_discovered', [])) for s in self.sessions)
        total_insights = sum(len(s.get('insights', [])) for s in self.sessions)
        
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Sessions | {len(self.sessions)} |")
        lines.append(f"| Papers Discovered | {total_papers:,} |")
        lines.append(f"| Insights Generated | {total_insights:,} |")
        lines.append(f"| Domains Explored | {len(self.domain_first_seen)} |")
        lines.append(f"| Unique Keywords | {len(set(self.all_keywords))} |")
        lines.append("")
        
        # Domain Discovery
        lines.append("---")
        lines.append("")
        lines.append("## Domain Discovery Timeline")
        lines.append("")
        lines.append("```")
        lines.append("Session  Domain")
        lines.append("-------  ------")
        for domain, idx in sorted(self.domain_first_seen.items(), key=lambda x: x[1]):
            bar = '*' * min(idx + 1, 30)
            lines.append(f"{idx:>4}     {domain:<25} {bar}")
        lines.append("```")
        lines.append("")
        
        # Learning Pattern Analysis
        lines.append("---")
        lines.append("")
        lines.append("## Learning Pattern Analysis")
        lines.append("")
        
        early_sessions = len(self.sessions) // 4
        late_start = len(self.sessions) * 3 // 4
        
        early_domains = set()
        for d in self.timeline_data[:early_sessions]:
            early_domains.update(d['domains'])
        
        late_new_domains = set()
        for d in self.timeline_data[late_start:]:
            for domain in d['domains']:
                if self.domain_first_seen.get(domain, 0) >= late_start:
                    late_new_domains.add(domain)
        
        lines.append(f"### Early Phase (Sessions 1-{early_sessions})")
        lines.append(f"- Domains discovered: {len(early_domains)}")
        lines.append(f"- Pattern: **Rapid breadth expansion**")
        lines.append("")
        
        lines.append(f"### Late Phase (Sessions {late_start}-{len(self.sessions)})")
        lines.append(f"- New domains: {len(late_new_domains)}")
        lines.append(f"- Pattern: **Deep specialization**")
        lines.append("")
        
        # Insight Distribution
        lines.append("---")
        lines.append("")
        lines.append("## Insight Type Distribution")
        lines.append("")
        
        all_insight_types = Counter()
        for session in self.sessions:
            for insight in session.get('insights', []):
                all_insight_types[insight.get('insight_type', 'unknown')] += 1
        
        total = sum(all_insight_types.values())
        lines.append("| Type | Count | Percentage |")
        lines.append("|------|-------|------------|")
        for itype, count in all_insight_types.most_common():
            pct = count / total * 100 if total > 0 else 0
            bar = '=' * int(pct / 2)
            lines.append(f"| {itype} | {count:,} | {pct:.1f}% {bar} |")
        lines.append("")
        
        # Top Keywords
        lines.append("---")
        lines.append("")
        lines.append("## Top Keywords")
        lines.append("")
        
        keyword_counts = Counter(self.all_keywords)
        lines.append("| Rank | Keyword | Frequency |")
        lines.append("|------|---------|-----------|")
        for rank, (word, count) in enumerate(keyword_counts.most_common(20), 1):
            lines.append(f"| {rank} | {word} | {count:,} |")
        lines.append("")
        
        # Wordcloud Evolution
        if wordcloud_frames:
            lines.append("---")
            lines.append("")
            lines.append("## Knowledge Evolution (Wordcloud)")
            lines.append("")
            lines.append("*Visual representation of keyword emergence over time:*")
            lines.append("")
            
            for i, frame in enumerate(wordcloud_frames):
                frame_name = Path(frame).name
                session_idx = int(len(self.sessions) * (i + 1) / len(wordcloud_frames))
                lines.append(f"### Stage {i + 1}: Session {session_idx}")
                lines.append(f"![Wordcloud Stage {i + 1}](wordcloud_frames/{frame_name})")
                lines.append("")
        
        # Session Highlights
        lines.append("---")
        lines.append("")
        lines.append("## Notable Sessions")
        lines.append("")
        
        sessions_by_insights = sorted(self.sessions, 
                                      key=lambda x: len(x.get('insights', [])), 
                                      reverse=True)[:5]
        
        for session in sessions_by_insights:
            idx = session['index']
            topic = session.get('topic', 'Unknown')[:80]
            papers = len(session.get('papers_discovered', []))
            insights = len(session.get('insights', []))
            lines.append(f"**Session {idx + 1}**: {topic}")
            lines.append(f"- Papers: {papers} | Insights: {insights}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Silvertongue Knowledge Report Generator*")
        
        report_content = '\n'.join(lines)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved: {output_path}")
        return report_content
    
    def generate_full_report(self, output_dir: str, num_wordcloud_frames: int = 8):
        """Generate complete report with all visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating knowledge report for {self.actor_name}...")
        print(f"Sessions: {len(self.sessions)}")
        
        # Generate wordcloud evolution
        wordcloud_dir = output_path / "wordcloud_frames"
        wordcloud_frames = self.generate_wordcloud_evolution(
            str(wordcloud_dir), 
            num_wordcloud_frames
        )
        
        # Generate markdown report
        report_path = output_path / f"{self.actor_name}_knowledge_report.md"
        self.generate_markdown_report(str(report_path), wordcloud_frames)
        
        print(f"\nReport generated in: {output_dir}/")
        print(f"  - Markdown: {report_path.name}")
        print(f"  - Wordcloud frames: {len(wordcloud_frames)} images")
        
        return str(report_path)


def main():
    parser = argparse.ArgumentParser(description='Generate knowledge evolution report')
    parser.add_argument('actor', help='Actor name (e.g., steve)')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory')
    parser.add_argument('--frames', type=int, default=8, help='Number of wordcloud frames')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or f"runs/{args.actor}_report"
    
    generator = KnowledgeReportGenerator(args.actor)
    count = generator.load_data()
    
    if count == 0:
        print(f"No sessions found for actor '{args.actor}'")
        return 1
    
    generator.generate_full_report(output_dir, args.frames)
    return 0


if __name__ == "__main__":
    exit(main())
