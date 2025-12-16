#!/usr/bin/env python3
"""
Knowledge Report Generator
===========================
Comprehensive report with visualizations and wordcloud evolution.

Usage:
    python knowledge_report_generator.py [actor_name] [--output-dir runs]
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from wordcloud import WordCloud
from PIL import Image
import io
import base64

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


class KnowledgeReportGenerator:
    """Generate comprehensive knowledge evolution report with visualizations"""
    
    # Domain classification keywords
    DOMAINS = {
        'Edge AI': ['edge', 'iot', 'embedded', 'mobile', 'device', 'on-device', 'tinyml'],
        'Transformers/LLM': ['transformer', 'attention', 'bert', 'gpt', 'llm', 'language model'],
        'Optimization': ['quantization', 'pruning', 'compression', 'sparsity', 'efficient', 'distillation'],
        'Reliability': ['reliability', 'robustness', 'fault', 'drift', 'stability', 'safety', 'failure'],
        'Performance': ['latency', 'throughput', 'inference', 'speed', 'real-time', 'real‑time'],
        'Continual Learning': ['continual', 'lifelong', 'incremental', 'online learning', 'streaming', 'forgetting'],
        'Hardware': ['hardware', 'accelerator', 'gpu', 'tpu', 'fpga', 'tensor', 'asic', 'npu'],
        'Computer Vision': ['vision', 'image', 'video', 'visual', 'cnn', 'detection', 'segmentation'],
        'NLP': ['nlp', 'text', 'sentiment', 'translation', 'summarization', 'embedding'],
        'Graph/Tensor': ['graph', 'tensor', 'gnn', 'network', 'contraction']
    }
    
    # Stopwords to exclude from wordclouds
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'using', 'based', 'this', 'that', 'these', 'those', 'it', 'its',
        'how', 'what', 'which', 'who', 'whom', 'why', 'where', 'when', 'while'
    }
    
    def __init__(self, actor_name: str, base_dir: str = "array/data"):
        self.actor_name = actor_name
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / actor_name / "learning_sessions"
        self.sessions = []
        self.timeline_data = []
        self.domain_first_seen = {}
        
    def load_data(self):
        """Load learning sessions"""
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
                    session['session_dir'] = str(session_dir.name)
                    
                    if insights_file.exists():
                        with open(insights_file) as f:
                            session['insights'] = json.load(f)
                    
                    self.sessions.append(session)
                except Exception as e:
                    print(f"Warning: {e}")
        
        self._build_timeline()
        return len(self.sessions)
    
    def _classify_domains(self, text: str) -> list:
        """Classify text into domains"""
        text_lower = text.lower()
        domains = []
        for domain, keywords in self.DOMAINS.items():
            if any(kw in text_lower for kw in keywords):
                domains.append(domain)
        return domains if domains else ['General']
    
    def _extract_words(self, text: str) -> list:
        """Extract meaningful words from text"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in self.STOPWORDS]
    
    def _build_timeline(self):
        """Build timeline data structure"""
        cumulative_domains = set()
        cumulative_papers = 0
        cumulative_insights = 0
        cumulative_words = Counter()
        
        for session in self.sessions:
            idx = session['index']
            topic = session.get('topic', '')
            
            # Get domains for this session
            session_domains = self._classify_domains(topic)
            for d in session_domains:
                if d not in cumulative_domains:
                    self.domain_first_seen[d] = idx
                cumulative_domains.add(d)
            
            # Count papers and insights
            papers = len(session.get('papers_discovered', []))
            insights_list = session.get('insights', [])
            insights = len(insights_list)
            cumulative_papers += papers
            cumulative_insights += insights
            
            # Extract words from topic
            topic_words = self._extract_words(topic)
            cumulative_words.update(topic_words)
            
            # Extract words from paper titles
            for paper in session.get('papers_discovered', []):
                title = paper.get('title', '')
                cumulative_words.update(self._extract_words(title))
            
            # Extract words from insights
            for ins in insights_list:
                content = ins.get('content', '')
                if isinstance(content, dict):
                    for v in content.values():
                        if isinstance(v, str):
                            cumulative_words.update(self._extract_words(v))
                elif isinstance(content, str):
                    cumulative_words.update(self._extract_words(content))
            
            # Insight type breakdown
            insight_types = Counter()
            for ins in insights_list:
                insight_types[ins.get('insight_type', 'unknown')] += 1
            
            self.timeline_data.append({
                'index': idx,
                'timestamp': session.get('created_at', ''),
                'topic': topic[:60],
                'domains': list(cumulative_domains),
                'new_domains': [d for d in session_domains if self.domain_first_seen.get(d) == idx],
                'domain_count': len(cumulative_domains),
                'papers_session': papers,
                'papers_cumulative': cumulative_papers,
                'insights_session': insights,
                'insights_cumulative': cumulative_insights,
                'word_counts': dict(cumulative_words.most_common(200)),
                'insight_types': dict(insight_types)
            })
    
    def _generate_wordcloud(self, word_counts: dict, title: str = "") -> plt.Figure:
        """Generate a wordcloud figure"""
        if not word_counts:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        wc = WordCloud(
            width=1200, 
            height=600, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            min_font_size=10,
            max_font_size=100
        ).generate_from_frequencies(word_counts)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        return fig
    
    def generate_wordcloud_evolution(self, output_dir: str, num_frames: int = 10):
        """Generate wordcloud evolution frames"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select evenly spaced snapshots
        frame_indices = np.linspace(0, len(self.timeline_data)-1, num_frames, dtype=int)
        
        frames_info = []
        for frame_num, idx in enumerate(frame_indices):
            data = self.timeline_data[idx]
            word_counts = data['word_counts']
            
            session_label = f"Session {idx+1}/{len(self.sessions)}"
            title = f"{session_label} | Domains: {data['domain_count']} | Papers: {data['papers_cumulative']:,}"
            
            fig = self._generate_wordcloud(word_counts, title)
            
            frame_path = output_path / f'wordcloud_frame_{frame_num:02d}.png'
            fig.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            frames_info.append({
                'frame': frame_num,
                'session': idx + 1,
                'domains': data['domain_count'],
                'papers': data['papers_cumulative'],
                'insights': data['insights_cumulative'],
                'top_words': list(Counter(word_counts).most_common(10)),
                'path': str(frame_path)
            })
        
        return frames_info
    
    def generate_domain_expansion_plot(self, output_path: str) -> dict:
        """Generate domain expansion visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Knowledge Domain Expansion: {self.actor_name}", fontsize=16, fontweight='bold')
        
        indices = [d['index'] for d in self.timeline_data]
        
        # 1. Domain count over time
        ax1 = axes[0, 0]
        domain_counts = [d['domain_count'] for d in self.timeline_data]
        ax1.fill_between(indices, domain_counts, alpha=0.3, color='blue')
        ax1.plot(indices, domain_counts, 'b-', linewidth=2, label='Domains Discovered')
        ax1.set_xlabel('Session Index')
        ax1.set_ylabel('Cumulative Domains')
        ax1.set_title('Domain Discovery Timeline')
        
        # Mark domain discoveries
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.domain_first_seen)))
        for (domain, first_idx), color in zip(self.domain_first_seen.items(), colors):
            ax1.axvline(x=first_idx, color=color, linestyle='--', alpha=0.5)
            ax1.annotate(domain, (first_idx, domain_counts[min(first_idx, len(domain_counts)-1)]), 
                        rotation=45, fontsize=8, alpha=0.8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Stacked area chart
        ax2 = axes[0, 1]
        domain_presence = defaultdict(list)
        for d in self.timeline_data:
            for domain in self.DOMAINS.keys():
                domain_presence[domain].append(1 if domain in d['domains'] else 0)
        
        y_stack = np.zeros(len(indices))
        domain_areas = [(dom, np.array(pres)) for dom, pres in domain_presence.items() if sum(pres) > 0]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(domain_areas)))
        for (domain, presence), color in zip(domain_areas, colors):
            ax2.fill_between(indices, y_stack, y_stack + presence, 
                           alpha=0.7, label=domain, color=color)
            y_stack = y_stack + presence
        
        ax2.set_xlabel('Session Index')
        ax2.set_ylabel('Domains Active')
        ax2.set_title('Domain Coverage Over Time')
        ax2.legend(loc='upper left', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        # 3. Knowledge accumulation
        ax3 = axes[1, 0]
        papers = [d['papers_cumulative'] for d in self.timeline_data]
        insights = [d['insights_cumulative'] for d in self.timeline_data]
        
        ax3.plot(indices, papers, 'g-', linewidth=2, label='Papers')
        ax3.plot(indices, insights, 'r-', linewidth=2, label='Insights')
        ax3.fill_between(indices, papers, alpha=0.2, color='green')
        ax3.fill_between(indices, insights, alpha=0.2, color='red')
        ax3.set_xlabel('Session Index')
        ax3.set_ylabel('Cumulative Count')
        ax3.set_title('Knowledge Accumulation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Insight type heatmap
        ax4 = axes[1, 1]
        insight_types = ['confirmation', 'expansion', 'challenge', 'revolutionary']
        
        window_size = max(1, len(self.timeline_data) // 10)
        windows = []
        window_labels = []
        
        for i in range(0, len(self.timeline_data), window_size):
            window = self.timeline_data[i:i+window_size]
            totals = {t: 0 for t in insight_types}
            for d in window:
                for t in insight_types:
                    totals[t] += d['insight_types'].get(t, 0)
            total = sum(totals.values())
            if total > 0:
                windows.append([totals[t]/total*100 for t in insight_types])
            else:
                windows.append([0]*len(insight_types))
            window_labels.append(f'{i}-{min(i+window_size, len(self.timeline_data))}')
        
        windows = np.array(windows).T
        
        im = ax4.imshow(windows, aspect='auto', cmap='YlOrRd')
        ax4.set_yticks(range(len(insight_types)))
        ax4.set_yticklabels(insight_types)
        ax4.set_xticks(range(len(window_labels)))
        ax4.set_xticklabels(window_labels, rotation=45, ha='right')
        ax4.set_xlabel('Session Window')
        ax4.set_title('Insight Type Distribution (%)')
        plt.colorbar(im, ax=ax4, label='%')
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return {
            'total_domains': len(self.domain_first_seen),
            'domain_discovery': dict(sorted(self.domain_first_seen.items(), key=lambda x: x[1])),
            'final_papers': papers[-1] if papers else 0,
            'final_insights': insights[-1] if insights else 0
        }
    
    def generate_waterfall_plot(self, output_path: str):
        """Create domain discovery waterfall chart"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sorted_domains = sorted(self.domain_first_seen.items(), key=lambda x: x[1])
        
        domains = [d[0] for d in sorted_domains]
        first_seen = [d[1] for d in sorted_domains]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(domains)))
        
        y_pos = range(len(domains))
        bars = ax.barh(y_pos, first_seen, color=colors, edgecolor='white', linewidth=1)
        
        for i, (domain, session_idx) in enumerate(sorted_domains):
            ax.text(session_idx + 2, i, f'Session {session_idx}', 
                   va='center', fontsize=9, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(domains)
        ax.set_xlabel('Session Index (when first discovered)')
        ax.set_title(f'Domain Discovery Waterfall: {self.actor_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
        
        ax.text(0.95, 0.05, f'Total: {len(domains)} domains\nSessions: {len(self.sessions)}',
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def compute_statistics(self) -> dict:
        """Compute comprehensive statistics"""
        total_papers = sum(d['papers_session'] for d in self.timeline_data)
        total_insights = sum(d['insights_session'] for d in self.timeline_data)
        
        # Compute learning phases
        mid_point = len(self.timeline_data) // 2
        early_papers = sum(d['papers_session'] for d in self.timeline_data[:mid_point])
        late_papers = sum(d['papers_session'] for d in self.timeline_data[mid_point:])
        
        early_domains = self.timeline_data[mid_point]['domain_count'] if mid_point < len(self.timeline_data) else 0
        late_domains = self.timeline_data[-1]['domain_count'] - early_domains if self.timeline_data else 0
        
        # Insight type totals
        insight_totals = Counter()
        for d in self.timeline_data:
            for itype, count in d['insight_types'].items():
                insight_totals[itype] += count
        
        # Top words overall
        final_words = self.timeline_data[-1]['word_counts'] if self.timeline_data else {}
        
        return {
            'total_sessions': len(self.sessions),
            'total_papers': total_papers,
            'total_insights': total_insights,
            'total_domains': len(self.domain_first_seen),
            'domain_discovery': dict(sorted(self.domain_first_seen.items(), key=lambda x: x[1])),
            'learning_phases': {
                'early': {
                    'sessions': mid_point,
                    'papers': early_papers,
                    'domains_discovered': early_domains
                },
                'late': {
                    'sessions': len(self.timeline_data) - mid_point,
                    'papers': late_papers,
                    'domains_discovered': late_domains
                }
            },
            'insight_distribution': dict(insight_totals.most_common()),
            'top_keywords': Counter(final_words).most_common(30),
            'growth_pattern': 'accelerating' if late_papers > early_papers else 'decelerating',
            'domain_saturation_session': max(self.domain_first_seen.values()) if self.domain_first_seen else 0
        }
    
    def generate_markdown_report(self, output_dir: str) -> str:
        """Generate comprehensive markdown report with embedded visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all visualizations
        print("Generating domain expansion plot...")
        expansion_stats = self.generate_domain_expansion_plot(str(output_path / f'{self.actor_name}_expansion.png'))
        
        print("Generating waterfall plot...")
        self.generate_waterfall_plot(str(output_path / f'{self.actor_name}_waterfall.png'))
        
        print("Generating wordcloud evolution...")
        wordcloud_frames = self.generate_wordcloud_evolution(str(output_path / 'wordcloud_frames'), num_frames=8)
        
        print("Computing statistics...")
        stats = self.compute_statistics()
        
        # Build report
        report = f"""# Knowledge Evolution Report: {self.actor_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Sessions | {stats['total_sessions']} |
| Papers Discovered | {stats['total_papers']:,} |
| Insights Generated | {stats['total_insights']:,} |
| Domains Explored | {stats['total_domains']} |
| Growth Pattern | {stats['growth_pattern'].title()} |
| Domain Saturation | Session {stats['domain_saturation_session']} |

---

## Domain Expansion Analysis

![Domain Expansion]({self.actor_name}_expansion.png)

### Domain Discovery Timeline

The learning agent discovered **{stats['total_domains']} distinct knowledge domains** over {stats['total_sessions']} sessions.

| Domain | First Discovered (Session) |
|--------|---------------------------|
"""
        for domain, session in stats['domain_discovery'].items():
            report += f"| {domain} | {session} |\n"
        
        report += f"""
### Key Finding: T-Shaped Learning Pattern

- **Breadth Phase (Sessions 0-{stats['learning_phases']['early']['sessions']}):**
  - {stats['learning_phases']['early']['domains_discovered']} domains discovered
  - {stats['learning_phases']['early']['papers']:,} papers explored
  
- **Depth Phase (Sessions {stats['learning_phases']['early']['sessions']}-{stats['total_sessions']}):**
  - {stats['learning_phases']['late']['domains_discovered']} new domains discovered
  - {stats['learning_phases']['late']['papers']:,} papers explored
  - Focus shifted to deepening existing knowledge

---

## Domain Discovery Waterfall

![Domain Waterfall]({self.actor_name}_waterfall.png)

This visualization shows when each domain was first encountered, demonstrating the **rapid initial exploration** followed by **knowledge consolidation**.

---

## Insight Distribution

| Type | Count | Percentage |
|------|-------|------------|
"""
        total_insights = sum(stats['insight_distribution'].values())
        for itype, count in stats['insight_distribution'].items():
            pct = (count / total_insights * 100) if total_insights > 0 else 0
            report += f"| {itype.title()} | {count:,} | {pct:.1f}% |\n"
        
        report += f"""
### Interpretation

"""
        conf_pct = stats['insight_distribution'].get('confirmation', 0) / total_insights * 100 if total_insights > 0 else 0
        exp_pct = stats['insight_distribution'].get('expansion', 0) / total_insights * 100 if total_insights > 0 else 0
        chal_pct = stats['insight_distribution'].get('challenge', 0) / total_insights * 100 if total_insights > 0 else 0
        rev_pct = stats['insight_distribution'].get('revolutionary', 0) / total_insights * 100 if total_insights > 0 else 0

        if conf_pct > 30:
            report += f"- **High Confirmation Rate ({conf_pct:.1f}%):** Indicates knowledge reinforcement and consolidation\n"
        if exp_pct > 25:
            report += f"- **Strong Expansion ({exp_pct:.1f}%):** Active discovery of new perspectives\n"
        if chal_pct > 20:
            report += f"- **Healthy Challenge Rate ({chal_pct:.1f}%):** Engaging with contradictory information\n"
        if rev_pct > 15:
            report += f"- **Notable Revolutionary Insights ({rev_pct:.1f}%):** Paradigm-shifting discoveries occurring\n"

        report += f"""
---

## Knowledge Vocabulary Evolution

The following wordclouds show how the knowledge vocabulary evolved across learning sessions:

### Wordcloud Timeline

"""
        for frame in wordcloud_frames:
            report += f"""
#### Session {frame['session']} / {stats['total_sessions']} — {frame['domains']} Domains, {frame['papers']:,} Papers

![Wordcloud Frame {frame['frame']}](wordcloud_frames/wordcloud_frame_{frame['frame']:02d}.png)

**Top Keywords:** {', '.join([f"{w[0]} ({w[1]})" for w in frame['top_words'][:5]])}

"""

        report += f"""
---

## Top Knowledge Keywords

The most frequent terms across all sessions reflect the core knowledge areas:

| Rank | Keyword | Frequency |
|------|---------|-----------|
"""
        for rank, (word, count) in enumerate(stats['top_keywords'][:20], 1):
            report += f"| {rank} | {word} | {count:,} |\n"
        
        report += f"""
---

## Conclusions

1. **Learning Pattern:** The agent exhibits a **{stats['growth_pattern']}** learning pattern with domain saturation occurring at session {stats['domain_saturation_session']}.

2. **Domain Coverage:** {stats['total_domains']} domains were explored, with most ({len([d for d, s in stats['domain_discovery'].items() if s < stats['total_sessions']//4])}) discovered in the first quarter of sessions.

3. **Knowledge Depth vs Breadth:** After initial broad exploration, the focus shifted to deepening understanding within established domains.

4. **Insight Quality:** {conf_pct + exp_pct:.1f}% of insights are confirmatory or expansive, while {chal_pct + rev_pct:.1f}% challenge or revolutionize existing knowledge.

---

*Report generated by Knowledge Report Generator*
"""
        
        # Save report
        report_path = output_path / f'{self.actor_name}_knowledge_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Report saved to: {report_path}")
        return str(report_path)


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive knowledge evolution report')
    parser.add_argument('actor', help='Actor name (e.g., steve)')
    parser.add_argument('--output-dir', '-o', default='runs', help='Output directory')
    
    args = parser.parse_args()
    
    generator = KnowledgeReportGenerator(args.actor)
    count = generator.load_data()
    
    if count == 0:
        print(f"No sessions found for actor '{args.actor}'")
        return 1
    
    print(f"Loaded {count} sessions for '{args.actor}'")
    
    report_path = generator.generate_markdown_report(args.output_dir)
    
    print(f"\n📊 Report generation complete!")
    print(f"   View: {report_path}")


if __name__ == "__main__":
    main()
