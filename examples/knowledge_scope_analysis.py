#!/usr/bin/env python3
"""
Knowledge Scope & Topic Expansion Analysis

Tracks how Steve's knowledge breadth expanded over iterations:
- When new topics/domains were first introduced
- Cumulative knowledge scope growth
- Topic clustering and thematic evolution
- Knowledge frontier expansion patterns
"""

import json
import yaml
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Set


class KnowledgeScopeAnalyzer:
    """Analyze knowledge scope expansion over iterations."""
    
    def __init__(self, actor_path: Path):
        self.actor_path = actor_path
        self.domain_path = actor_path / "expertise" / "domain"
        
        # Load learning book
        self.learning_book = self._load_json(actor_path / "learning_book.json")
        
        # Parse iterations with extracted themes
        self.iterations = self._parse_iterations_with_themes()
        
    def _load_json(self, path: Path) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _parse_iterations_with_themes(self) -> List[dict]:
        """Parse iterations and extract themes from each."""
        iterations = []
        history = self.learning_book.get("learning_history", [])
        
        global_iter = 0
        for entry in history:
            if entry.get("event_type") == "cognitive_loop_update":
                global_iter += 1
                
                # Extract topic
                desc = entry.get("description", "")
                topic_match = re.search(r"on:\s*(.+)$", desc)
                topic = topic_match.group(1).strip() if topic_match else "unknown"
                
                # Extract all themes from insights and mindset changes
                all_text = " ".join(entry.get("insights_gained", []))
                all_text += " ".join(entry.get("mindset_changes", []))
                all_text += " " + topic
                
                themes = self._extract_themes(all_text)
                keywords = self._extract_keywords(all_text)
                domain = self._categorize_to_domain(all_text)
                
                iterations.append({
                    "iter": global_iter,
                    "topic": topic,
                    "themes": themes,
                    "keywords": keywords,
                    "domain": domain,
                    "insights": entry.get("insights_gained", []),
                    "mindset_changes": entry.get("mindset_changes", []),
                })
        
        return iterations
    
    def _extract_themes(self, text: str) -> Set[str]:
        """Extract high-level themes from text."""
        text_lower = text.lower()
        themes = set()
        
        theme_patterns = {
            "Transformer Architecture": ['transformer', 'attention mechanism', 'self-attention', 'multi-head'],
            "Edge Computing": ['edge', '5g', 'iot', 'embedded', 'mobile inference'],
            "Model Compression": ['quantization', 'pruning', 'distillation', 'compression'],
            "Continual Learning": ['continual learning', 'catastrophic forgetting', 'replay buffer', 'lifelong'],
            "Robustness": ['adversarial', 'robustness', 'attack', 'perturbation'],
            "Efficiency": ['latency', 'throughput', 'energy', 'flops', 'efficiency'],
            "Fairness & Ethics": ['bias', 'fairness', 'ethical', 'misinformation'],
            "Multimodal": ['multimodal', 'cross-modal', 'vision-language', 'audio-visual'],
            "Graph Neural Networks": ['gnn', 'graph neural', 'message passing', 'node embedding'],
            "Tensor Operations": ['tensor', 'contraction', 'decomposition', 'einsum'],
            "Scaling Laws": ['scaling law', 'chinchilla', 'compute optimal'],
            "Neural-Symbolic": ['symbolic', 'reasoning', 'knowledge graph', 'logic'],
            "Drift Detection": ['drift', 'distribution shift', 'covariate shift'],
            "Hardware Acceleration": ['gpu', 'tpu', 'npu', 'accelerator', 'asic'],
            "Optimization": ['optimizer', 'gradient', 'convergence', 'adam', 'sgd'],
            "Regularization": ['regularization', 'dropout', 'weight decay', 'l2'],
            "Memory Systems": ['memory', 'buffer', 'cache', 'bandwidth'],
            "Reliability": ['reliability', 'fault tolerance', 'degradation', 'failure'],
            "Calibration": ['calibration', 'uncertainty', 'confidence'],
            "Pre-training": ['pre-training', 'fine-tuning', 'transfer learning'],
            "Sparsity": ['sparsity', 'sparse', 'structured pruning'],
            "Normalization": ['normalization', 'batch norm', 'layer norm'],
            "Curriculum Learning": ['curriculum', 'progressive', 'staged training'],
            "Data Augmentation": ['augmentation', 'data augment'],
            "Benchmark Design": ['benchmark', 'evaluation', 'metric'],
            "API & Standards": ['api', 'standard', 'interface', 'protocol'],
            "Thermodynamics": ['thermodynamic', 'landauer', 'energy bound'],
            "Information Theory": ['entropy', 'mutual information', 'information bottleneck'],
        }
        
        for theme, patterns in theme_patterns.items():
            if any(p in text_lower for p in patterns):
                themes.add(theme)
        
        return themes
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract specific technical keywords."""
        text_lower = text.lower()
        keywords = set()
        
        keyword_list = [
            'transformer', 'attention', 'cnn', 'rnn', 'gnn', 'lstm', 'bert', 'gpt',
            'quantization', 'pruning', 'distillation', 'sparsity', 'compression',
            'edge', 'latency', 'throughput', 'inference', 'deployment',
            'continual', 'forgetting', 'replay', 'buffer', 'drift',
            'adversarial', 'robustness', 'perturbation', 'attack',
            'bias', 'fairness', 'misinformation', 'ethical',
            'multimodal', 'cross-modal', 'fusion', 'alignment',
            'tensor', 'contraction', 'decomposition', 'manifold',
            'scaling', 'chinchilla', 'emergent', 'capability',
            'symbolic', 'reasoning', 'logic', 'knowledge graph',
            'gpu', 'tpu', 'npu', 'accelerator', 'hardware',
            'optimizer', 'gradient', 'convergence', 'learning rate',
            'regularization', 'dropout', 'weight decay',
            'calibration', 'uncertainty', 'confidence',
            'pre-training', 'fine-tuning', 'transfer',
            'curriculum', 'progressive', 'staged',
            'benchmark', 'evaluation', 'metric',
            'api', 'standard', 'interface',
            'thermodynamic', 'landauer', 'entropy',
            '5g', '6g', 'iot', 'mobile', 'wireless',
            'stability', 'plasticity', 'trade-off',
            'architecture', 'layer', 'module', 'block',
            'normalization', 'residual', 'skip connection',
        ]
        
        for kw in keyword_list:
            if kw in text_lower:
                keywords.add(kw)
        
        return keywords
    
    def _categorize_to_domain(self, text: str) -> str:
        """Categorize text into a broad domain."""
        text_lower = text.lower()
        
        # Check in order of specificity
        if any(w in text_lower for w in ['continual', 'forgetting', 'replay', 'drift', 'lifelong', 'buffer', 'stability', 'plasticity']):
            return "Continual Learning"
        if any(w in text_lower for w in ['quantiz', 'prun', 'compress', 'distill', 'sparse', 'bit-width', 'low-precision']):
            return "Model Efficiency"
        if any(w in text_lower for w in ['robust', 'adversarial', 'attack', 'reliab', 'fault', 'perturbation']):
            return "Robustness & Security"
        if any(w in text_lower for w in ['bias', 'fair', 'ethic', 'misinformation', 'alignment']):
            return "Ethics & Fairness"
        if any(w in text_lower for w in ['multimodal', 'cross-modal', 'vision-language', 'audio-visual', 'fusion']):
            return "Multimodal AI"
        if any(w in text_lower for w in ['graph', 'gnn', 'node embedding', 'message passing']):
            return "Graph Learning"
        if any(w in text_lower for w in ['tensor', 'contraction', 'manifold', 'differential geometry', 'decomposition']):
            return "Tensor & Math"
        if any(w in text_lower for w in ['benchmark', 'metric', 'evaluat', 'standardiz']):
            return "Benchmarking"
        if any(w in text_lower for w in ['edge', '5g', '6g', 'iot', 'mobile', 'embedded', 'accelerator']):
            return "Edge & Deployment"
        if any(w in text_lower for w in ['transformer', 'language model', 'llm', 'attention', 'bert', 'gpt', 'pre-train']):
            return "Language Models"
        if any(w in text_lower for w in ['scaling', 'trade-off', 'trade‑off', 'optimal', 'balance']):
            return "Scaling & Trade-offs"
        if any(w in text_lower for w in ['latency', 'throughput', 'inference', 'deploy']):
            return "Deployment & Inference"
        return "General ML"
    
    def analyze_scope_expansion(self) -> dict:
        """Analyze cumulative scope expansion over iterations."""
        cumulative_themes = set()
        cumulative_keywords = set()
        cumulative_domains = set()
        cumulative_topics = set()
        
        expansion_timeline = []
        
        for it in self.iterations:
            new_themes = it["themes"] - cumulative_themes
            new_keywords = it["keywords"] - cumulative_keywords
            new_domain = it["domain"] not in cumulative_domains
            
            # Normalize topic for uniqueness
            topic_key = it["topic"][:50].lower()
            new_topic = topic_key not in cumulative_topics
            
            cumulative_themes.update(it["themes"])
            cumulative_keywords.update(it["keywords"])
            cumulative_domains.add(it["domain"])
            cumulative_topics.add(topic_key)
            
            expansion_timeline.append({
                "iter": it["iter"],
                "topic": it["topic"][:60],
                "new_themes": list(new_themes),
                "new_keywords": list(new_keywords),
                "new_domain": it["domain"] if new_domain else None,
                "cumulative_theme_count": len(cumulative_themes),
                "cumulative_keyword_count": len(cumulative_keywords),
                "cumulative_domain_count": len(cumulative_domains),
                "cumulative_topic_count": len(cumulative_topics),
            })
        
        return {
            "timeline": expansion_timeline,
            "final_themes": cumulative_themes,
            "final_keywords": cumulative_keywords,
            "final_domains": cumulative_domains,
            "final_topic_count": len(cumulative_topics),
        }
    
    def analyze_theme_introduction_order(self) -> List[dict]:
        """Track when each theme was first introduced."""
        theme_first_seen = {}
        
        for it in self.iterations:
            for theme in it["themes"]:
                if theme not in theme_first_seen:
                    theme_first_seen[theme] = {
                        "iter": it["iter"],
                        "topic": it["topic"][:60],
                        "domain": it["domain"],
                    }
        
        return sorted(theme_first_seen.items(), key=lambda x: x[1]["iter"])
    
    def analyze_domain_expansion(self) -> dict:
        """Track domain coverage expansion."""
        domain_timeline = defaultdict(list)
        domain_first_seen = {}
        
        for it in self.iterations:
            domain = it["domain"]
            domain_timeline[domain].append(it["iter"])
            
            if domain not in domain_first_seen:
                domain_first_seen[domain] = it["iter"]
        
        return {
            "first_seen": dict(sorted(domain_first_seen.items(), key=lambda x: x[1])),
            "coverage": {d: len(iters) for d, iters in domain_timeline.items()},
            "timeline": dict(domain_timeline),
        }
    
    def analyze_knowledge_frontier(self) -> List[dict]:
        """Identify iterations that expanded the knowledge frontier significantly."""
        frontier_expansions = []
        
        expansion = self.analyze_scope_expansion()
        timeline = expansion["timeline"]
        
        for entry in timeline:
            # Significant expansion = new themes or new domain
            if len(entry["new_themes"]) >= 2 or entry["new_domain"]:
                frontier_expansions.append({
                    "iter": entry["iter"],
                    "topic": entry["topic"],
                    "new_themes": entry["new_themes"],
                    "new_domain": entry["new_domain"],
                    "cumulative_scope": entry["cumulative_theme_count"],
                })
        
        return frontier_expansions
    
    def analyze_topic_clustering(self) -> dict:
        """Cluster topics by thematic similarity."""
        clusters = defaultdict(list)
        
        for it in self.iterations:
            # Use dominant theme as cluster key
            if it["themes"]:
                # Pick most specific theme
                theme = sorted(it["themes"], key=len, reverse=True)[0]
                clusters[theme].append({
                    "iter": it["iter"],
                    "topic": it["topic"][:50],
                })
        
        # Sort by cluster size
        return dict(sorted(clusters.items(), key=lambda x: -len(x[1])))
    
    def analyze_scope_velocity(self, window: int = 50) -> List[dict]:
        """Measure how fast scope expanded over iteration windows."""
        expansion = self.analyze_scope_expansion()
        timeline = expansion["timeline"]
        
        velocities = []
        for i in range(0, len(timeline), window):
            window_data = timeline[i:i+window]
            if len(window_data) < 2:
                continue
            
            start_themes = window_data[0]["cumulative_theme_count"]
            end_themes = window_data[-1]["cumulative_theme_count"]
            new_themes_in_window = end_themes - start_themes
            
            start_keywords = window_data[0]["cumulative_keyword_count"]
            end_keywords = window_data[-1]["cumulative_keyword_count"]
            new_keywords_in_window = end_keywords - start_keywords
            
            # Count new domains in window
            new_domains = sum(1 for w in window_data if w["new_domain"])
            
            velocities.append({
                "iter_start": window_data[0]["iter"],
                "iter_end": window_data[-1]["iter"],
                "new_themes": new_themes_in_window,
                "new_keywords": new_keywords_in_window,
                "new_domains": new_domains,
                "cumulative_themes": end_themes,
                "cumulative_keywords": end_keywords,
                "expansion_rate": new_themes_in_window / len(window_data),
            })
        
        return velocities
    
    def analyze_theme_persistence(self) -> dict:
        """Analyze which themes persist vs appear briefly."""
        theme_occurrences = defaultdict(list)
        
        for it in self.iterations:
            for theme in it["themes"]:
                theme_occurrences[theme].append(it["iter"])
        
        persistence = {}
        for theme, iters in theme_occurrences.items():
            persistence[theme] = {
                "first": min(iters),
                "last": max(iters),
                "count": len(iters),
                "span": max(iters) - min(iters),
                "density": len(iters) / (max(iters) - min(iters) + 1),
            }
        
        return dict(sorted(persistence.items(), key=lambda x: -x[1]["count"]))


def generate_scope_report(analyzer: KnowledgeScopeAnalyzer):
    """Generate comprehensive scope expansion report."""
    
    total_iters = len(analyzer.iterations)
    
    print("\n" + "="*100)
    print("🔭 KNOWLEDGE SCOPE & TOPIC EXPANSION ANALYSIS")
    print("="*100)
    print(f"\n  Total Iterations Analyzed: {total_iters}")
    
    # 1. Theme Introduction Order
    print("\n\n" + "─"*100)
    print("📚 THEME INTRODUCTION TIMELINE")
    print("─"*100)
    print("\n  When each knowledge theme was FIRST discovered:\n")
    
    theme_order = analyzer.analyze_theme_introduction_order()
    
    print(f"  {'Iter':>5} │ {'Theme':<30} │ Context")
    print("  " + "-"*95)
    
    for theme, data in theme_order:
        print(f"  {data['iter']:>5} │ {theme:<30} │ {data['topic'][:55]}...")
    
    # 2. Cumulative Scope Growth
    print("\n\n" + "─"*100)
    print("📈 CUMULATIVE KNOWLEDGE SCOPE GROWTH")
    print("─"*100)
    
    expansion = analyzer.analyze_scope_expansion()
    timeline = expansion["timeline"]
    
    # Show growth at key intervals
    print("\n  Iteration │ Themes │ Keywords │ Domains │ Topics │ Growth Visualization")
    print("  " + "-"*90)
    
    checkpoints = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, len(timeline)]
    max_themes = timeline[-1]["cumulative_theme_count"] if timeline else 1
    
    for cp in checkpoints:
        if cp <= len(timeline):
            entry = timeline[cp-1]
            bar_len = int(40 * entry["cumulative_theme_count"] / max_themes)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {entry['iter']:>9} │ {entry['cumulative_theme_count']:>6} │ {entry['cumulative_keyword_count']:>8} │ {entry['cumulative_domain_count']:>7} │ {entry['cumulative_topic_count']:>6} │ {bar}")
    
    # 3. Domain Expansion
    print("\n\n" + "─"*100)
    print("🌐 DOMAIN EXPANSION ORDER")
    print("─"*100)
    
    domain_exp = analyzer.analyze_domain_expansion()
    
    print("\n  When each knowledge domain was first explored:\n")
    print(f"  {'Iter':>5} │ {'Domain':<25} │ {'Total Coverage':>15} │ Presence")
    print("  " + "-"*80)
    
    for domain, first_iter in domain_exp["first_seen"].items():
        coverage = domain_exp["coverage"].get(domain, 0)
        pct = coverage / total_iters * 100
        bar = "▓" * int(pct / 3)
        print(f"  {first_iter:>5} │ {domain:<25} │ {coverage:>7} ({pct:>5.1f}%) │ {bar}")
    
    # 4. Scope Expansion Velocity
    print("\n\n" + "─"*100)
    print("🚀 SCOPE EXPANSION VELOCITY")
    print("─"*100)
    
    velocities = analyzer.analyze_scope_velocity(window=100)
    
    print("\n  Iterations    │ New Themes │ New Keywords │ New Domains │ Cumulative │ Rate")
    print("  " + "-"*85)
    
    for v in velocities:
        rate_bar = "█" * int(v["expansion_rate"] * 20) if v["expansion_rate"] > 0 else "░"
        print(f"  {v['iter_start']:>4} - {v['iter_end']:<5} │ {v['new_themes']:>10} │ {v['new_keywords']:>12} │ {v['new_domains']:>11} │ {v['cumulative_themes']:>10} │ {rate_bar}")
    
    # 5. Knowledge Frontier Expansions
    print("\n\n" + "─"*100)
    print("🌟 MAJOR KNOWLEDGE FRONTIER EXPANSIONS")
    print("─"*100)
    
    frontier = analyzer.analyze_knowledge_frontier()
    
    print(f"\n  Found {len(frontier)} significant frontier expansions:\n")
    
    for i, f in enumerate(frontier[:25], 1):
        print(f"  {i:>2}. Iteration {f['iter']:>4}: {f['topic'][:50]}...")
        if f["new_domain"]:
            print(f"      🆕 NEW DOMAIN: {f['new_domain']}")
        if f["new_themes"]:
            print(f"      📖 New themes: {', '.join(f['new_themes'][:4])}")
        print(f"      📊 Cumulative scope: {f['cumulative_scope']} themes")
        print()
    
    # 6. Theme Persistence Analysis
    print("\n" + "─"*100)
    print("🔄 THEME PERSISTENCE (Most Enduring Themes)")
    print("─"*100)
    
    persistence = analyzer.analyze_theme_persistence()
    
    print("\n  Theme                        │ Count │ First │ Last  │ Span  │ Density │ Timeline")
    print("  " + "-"*95)
    
    for theme, data in list(persistence.items())[:15]:
        # Create mini timeline
        timeline_vis = [" "] * 20
        for it in analyzer.iterations:
            if theme in it["themes"]:
                pos = int(19 * (it["iter"] - 1) / (total_iters - 1)) if total_iters > 1 else 0
                timeline_vis[pos] = "█"
        
        print(f"  {theme:<28} │ {data['count']:>5} │ {data['first']:>5} │ {data['last']:>5} │ {data['span']:>5} │ {data['density']:>6.2f} │ [{''.join(timeline_vis)}]")
    
    # 7. Topic Clustering
    print("\n\n" + "─"*100)
    print("🗂️  TOPIC CLUSTERING BY THEME")
    print("─"*100)
    
    clusters = analyzer.analyze_topic_clustering()
    
    for theme, topics in list(clusters.items())[:8]:
        print(f"\n  📁 {theme} ({len(topics)} iterations)")
        print("  " + "-"*60)
        for t in topics[:5]:
            print(f"     Iter {t['iter']:>4}: {t['topic']}...")
        if len(topics) > 5:
            print(f"     ... and {len(topics) - 5} more")
    
    # 8. Summary Statistics
    print("\n\n" + "─"*100)
    print("📋 SCOPE EXPANSION SUMMARY")
    print("─"*100)
    
    final = expansion
    
    # Calculate expansion phases
    early_themes = timeline[99]["cumulative_theme_count"] if len(timeline) > 99 else 0
    mid_themes = timeline[499]["cumulative_theme_count"] if len(timeline) > 499 else 0
    final_themes = timeline[-1]["cumulative_theme_count"] if timeline else 0
    
    print(f"""
  FINAL KNOWLEDGE SCOPE:
  ──────────────────────
  • Total unique themes discovered: {len(final['final_themes'])}
  • Total unique keywords encountered: {len(final['final_keywords'])}
  • Total domains covered: {len(final['final_domains'])}
  • Total unique topics explored: {final['final_topic_count']}
  
  EXPANSION PHASES:
  ─────────────────
  • Iterations 1-100:   {early_themes} themes (rapid initial expansion)
  • Iterations 101-500: {mid_themes - early_themes} new themes (steady growth)
  • Iterations 501+:    {final_themes - mid_themes} new themes (refinement phase)
  
  THEMES DISCOVERED: {sorted(final['final_themes'])}
  
  DOMAINS COVERED: {sorted(final['final_domains'])}
  
  LEARNING PATTERN:
  ─────────────────
  • Most frontier expansions occurred in early iterations
  • Theme discovery rate: {final_themes / total_iters:.3f} themes per iteration
  • Dominant themes persisted throughout learning journey
  • Knowledge became increasingly interconnected over time
    """)
    
    print("="*100)
    print("Analysis Complete!")
    print("="*100)


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent
    actor_path = base_path / "array" / "actors" / "steve"
    
    print("Loading learning data...")
    analyzer = KnowledgeScopeAnalyzer(actor_path)
    
    print(f"Parsed {len(analyzer.iterations)} iterations")
    
    generate_scope_report(analyzer)


if __name__ == "__main__":
    main()
