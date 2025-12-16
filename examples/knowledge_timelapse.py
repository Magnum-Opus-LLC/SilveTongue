#!/usr/bin/env python3
"""
Knowledge Time-Lapse Visualization
===================================
Creates visual time-lapse of knowledge/domain/principle expansion.

Usage:
    python knowledge_timelapse.py [actor_name] [--output prefix]
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


class KnowledgeTimelapse:
    """Generate time-lapse visualizations of knowledge expansion"""
    
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
    
    # Principle keywords to detect
    PRINCIPLE_KEYWORDS = [
        'trade-off', 'trade‑off', 'principle', 'law', 'theorem', 'fundamental',
        'optimal', 'balance', 'constraint', 'guarantee', 'bound', 'limit'
    ]
    
    def __init__(self, actor_name: str, base_dir: str = "array/data"):
        self.actor_name = actor_name
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / actor_name / "learning_sessions"
        self.actor_dir = Path("array/actors") / actor_name
        self.sessions = []
        self.timeline_data = []
        
    def load_data(self):
        """Load learning sessions and knowledge files"""
        # Load sessions
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
    
    def _extract_principles(self, session: dict) -> list:
        """Extract principle-like statements from insights"""
        principles = []
        for insight in session.get('insights', []):
            content = insight.get('content', '')
            if isinstance(content, dict):
                text = str(content)
            else:
                text = str(content)
            
            text_lower = text.lower()
            if any(kw in text_lower for kw in self.PRINCIPLE_KEYWORDS):
                # Extract a short summary
                if isinstance(content, dict):
                    principle = content.get('new_perspective', content.get('new_paradigm', ''))[:100]
                else:
                    principle = text[:100]
                if principle:
                    principles.append(principle)
        
        return principles
    
    def _build_timeline(self):
        """Build timeline data structure"""
        cumulative_domains = set()
        cumulative_papers = 0
        cumulative_insights = 0
        cumulative_principles = []
        domain_first_seen = {}
        
        for session in self.sessions:
            idx = session['index']
            topic = session.get('topic', '')
            
            # Get domains for this session
            session_domains = self._classify_domains(topic)
            for d in session_domains:
                if d not in cumulative_domains:
                    domain_first_seen[d] = idx
                cumulative_domains.add(d)
            
            # Count papers and insights
            papers = len(session.get('papers_discovered', []))
            insights = len(session.get('insights', []))
            cumulative_papers += papers
            cumulative_insights += insights
            
            # Extract principles
            principles = self._extract_principles(session)
            cumulative_principles.extend(principles)
            
            # Insight type breakdown
            insight_types = Counter()
            for ins in session.get('insights', []):
                insight_types[ins.get('insight_type', 'unknown')] += 1
            
            self.timeline_data.append({
                'index': idx,
                'timestamp': session.get('created_at', ''),
                'topic': topic[:60],
                'domains': list(cumulative_domains),
                'new_domains': [d for d in session_domains if domain_first_seen.get(d) == idx],
                'domain_count': len(cumulative_domains),
                'papers_session': papers,
                'papers_cumulative': cumulative_papers,
                'insights_session': insights,
                'insights_cumulative': cumulative_insights,
                'principles_count': len(cumulative_principles),
                'insight_types': dict(insight_types)
            })
        
        self.domain_first_seen = domain_first_seen
    
    def plot_domain_expansion(self, output_path: str = None):
        """Plot domain expansion over time"""
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
        
        # Mark when each domain was discovered
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.domain_first_seen)))
        for (domain, first_idx), color in zip(self.domain_first_seen.items(), colors):
            ax1.axvline(x=first_idx, color=color, linestyle='--', alpha=0.5)
            ax1.annotate(domain, (first_idx, domain_counts[first_idx]), 
                        rotation=45, fontsize=8, alpha=0.8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Stacked area chart of domain coverage
        ax2 = axes[0, 1]
        domain_presence = defaultdict(list)
        for d in self.timeline_data:
            for domain in self.DOMAINS.keys():
                domain_presence[domain].append(1 if domain in d['domains'] else 0)
        
        # Create stacked area
        y_stack = np.zeros(len(indices))
        domain_areas = []
        for domain, presence in domain_presence.items():
            if sum(presence) > 0:  # Only show domains that appear
                domain_areas.append((domain, np.array(presence)))
        
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
        
        # 3. Knowledge accumulation (papers + insights)
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
        
        # Add growth rate annotations
        mid = len(indices) // 2
        papers_rate_1 = papers[mid] / mid if mid > 0 else 0
        papers_rate_2 = (papers[-1] - papers[mid]) / (len(indices) - mid) if len(indices) > mid else 0
        ax3.annotate(f'Early rate: {papers_rate_1:.1f}/session', 
                    xy=(mid//2, papers[mid//2]), fontsize=9, alpha=0.7)
        ax3.annotate(f'Late rate: {papers_rate_2:.1f}/session', 
                    xy=(mid + (len(indices)-mid)//2, papers[mid + (len(indices)-mid)//2]), 
                    fontsize=9, alpha=0.7)
        
        # 4. Insight type distribution over time (heatmap style)
        ax4 = axes[1, 1]
        insight_types = ['confirmation', 'expansion', 'challenge', 'revolutionary']
        
        # Aggregate by windows of 20 sessions
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
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig
    
    def plot_discovery_waterfall(self, output_path: str = None):
        """Create a waterfall chart showing when each domain was discovered"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort domains by discovery order
        sorted_domains = sorted(self.domain_first_seen.items(), key=lambda x: x[1])
        
        domains = [d[0] for d in sorted_domains]
        first_seen = [d[1] for d in sorted_domains]
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(domains)))
        
        y_pos = range(len(domains))
        bars = ax.barh(y_pos, first_seen, color=colors, edgecolor='white', linewidth=1)
        
        # Add session markers
        for i, (domain, session_idx) in enumerate(sorted_domains):
            ax.text(session_idx + 2, i, f'Session {session_idx}', 
                   va='center', fontsize=9, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(domains)
        ax.set_xlabel('Session Index (when first discovered)')
        ax.set_title(f'Domain Discovery Waterfall: {self.actor_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add annotation
        ax.text(0.95, 0.05, f'Total: {len(domains)} domains\nSessions: {len(self.sessions)}',
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig
    
    def plot_knowledge_spiral(self, output_path: str = None):
        """Create a spiral visualization of knowledge expansion"""
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        n_sessions = len(self.timeline_data)
        
        # Create spiral
        theta = np.linspace(0, 4 * np.pi, n_sessions)  # 2 full rotations
        r_base = np.linspace(1, 10, n_sessions)
        
        # Size based on insights gained
        sizes = [d['insights_session'] * 5 + 20 for d in self.timeline_data]
        
        # Color based on domain diversity
        colors = [d['domain_count'] for d in self.timeline_data]
        
        scatter = ax.scatter(theta, r_base, c=colors, s=sizes, 
                           cmap='plasma', alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # Mark domain discoveries
        for domain, session_idx in self.domain_first_seen.items():
            if session_idx < n_sessions:
                ax.annotate(domain, 
                           xy=(theta[session_idx], r_base[session_idx]),
                           fontsize=8, alpha=0.8,
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        
        ax.set_title(f'Knowledge Spiral: {self.actor_name}\n(Size=insights, Color=domains)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Number of Domains')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        return fig
    
    def plot_timelapse_frames(self, output_dir: str, num_frames: int = 20):
        """Generate individual frames for a time-lapse animation"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_indices = np.linspace(0, len(self.timeline_data)-1, num_frames, dtype=int)
        
        for frame_num, end_idx in enumerate(frame_indices):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Data up to this point
            data_slice = self.timeline_data[:end_idx+1]
            indices = [d['index'] for d in data_slice]
            
            # Panel 1: Domain count
            ax1 = axes[0]
            domain_counts = [d['domain_count'] for d in data_slice]
            ax1.fill_between(indices, domain_counts, alpha=0.3, color='blue')
            ax1.plot(indices, domain_counts, 'b-', linewidth=2)
            ax1.set_xlim(0, len(self.timeline_data))
            ax1.set_ylim(0, max(d['domain_count'] for d in self.timeline_data) + 1)
            ax1.set_xlabel('Session')
            ax1.set_ylabel('Domains')
            ax1.set_title(f'Domains Discovered: {domain_counts[-1]}')
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: Papers
            ax2 = axes[1]
            papers = [d['papers_cumulative'] for d in data_slice]
            ax2.fill_between(indices, papers, alpha=0.3, color='green')
            ax2.plot(indices, papers, 'g-', linewidth=2)
            ax2.set_xlim(0, len(self.timeline_data))
            ax2.set_ylim(0, max(d['papers_cumulative'] for d in self.timeline_data))
            ax2.set_xlabel('Session')
            ax2.set_ylabel('Papers')
            ax2.set_title(f'Papers Discovered: {papers[-1]:,}')
            ax2.grid(True, alpha=0.3)
            
            # Panel 3: Insights
            ax3 = axes[2]
            insights = [d['insights_cumulative'] for d in data_slice]
            ax3.fill_between(indices, insights, alpha=0.3, color='red')
            ax3.plot(indices, insights, 'r-', linewidth=2)
            ax3.set_xlim(0, len(self.timeline_data))
            ax3.set_ylim(0, max(d['insights_cumulative'] for d in self.timeline_data))
            ax3.set_xlabel('Session')
            ax3.set_ylabel('Insights')
            ax3.set_title(f'Insights Gained: {insights[-1]:,}')
            ax3.grid(True, alpha=0.3)
            
            fig.suptitle(f'Session {end_idx+1}/{len(self.timeline_data)}: {data_slice[-1]["topic"]}', 
                        fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            frame_path = output_path / f'frame_{frame_num:03d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Saved {num_frames} frames to {output_dir}/")
        print(f"To create GIF: convert -delay 50 -loop 0 {output_dir}/frame_*.png {output_dir}/timelapse.gif")
    
    def generate_all_visualizations(self, output_prefix: str = None):
        """Generate all visualizations"""
        if output_prefix is None:
            output_prefix = f"runs/{self.actor_name}_knowledge"
        
        print(f"\nGenerating visualizations for {self.actor_name}...")
        print(f"Sessions loaded: {len(self.sessions)}")
        print(f"Domains discovered: {len(self.domain_first_seen)}")
        
        # Main expansion plot
        self.plot_domain_expansion(f"{output_prefix}_expansion.png")
        
        # Waterfall
        self.plot_discovery_waterfall(f"{output_prefix}_waterfall.png")
        
        # Spiral
        self.plot_knowledge_spiral(f"{output_prefix}_spiral.png")
        
        print(f"\n✅ All visualizations saved with prefix: {output_prefix}")
        
        # Summary
        print(f"\n📊 Knowledge Expansion Summary:")
        print(f"   Total Sessions: {len(self.sessions)}")
        print(f"   Domains Discovered: {len(self.domain_first_seen)}")
        print(f"   Domain Discovery Timeline:")
        for domain, idx in sorted(self.domain_first_seen.items(), key=lambda x: x[1]):
            print(f"      Session {idx:3d}: {domain}")


def main():
    parser = argparse.ArgumentParser(description='Generate knowledge time-lapse visualizations')
    parser.add_argument('actor', help='Actor name (e.g., steve)')
    parser.add_argument('--output', '-o', default=None, help='Output prefix for files')
    parser.add_argument('--frames', action='store_true', help='Generate animation frames')
    parser.add_argument('--frames-dir', default='runs/timelapse_frames', help='Directory for frames')
    parser.add_argument('--num-frames', type=int, default=20, help='Number of frames to generate')
    
    args = parser.parse_args()
    
    timelapse = KnowledgeTimelapse(args.actor)
    count = timelapse.load_data()
    
    if count == 0:
        print(f"No sessions found for actor '{args.actor}'")
        return 1
    
    timelapse.generate_all_visualizations(args.output)
    
    if args.frames:
        timelapse.plot_timelapse_frames(args.frames_dir, args.num_frames)
    
    plt.show()


if __name__ == "__main__":
    main()
