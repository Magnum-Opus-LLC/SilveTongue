#!/usr/bin/env python3
"""
Knowledge Scope Analyzer
========================
Analyze knowledge scope expansion over iterations, tracking themes,
domains, and frontier expansion patterns.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Set


class KnowledgeScopeAnalyzer:
    """Analyze knowledge scope expansion over iterations."""

    THEME_PATTERNS = {
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

    KEYWORDS = [
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

    def __init__(self, actor_name: str, base_dir: str = "array"):
        self.actor_name = actor_name
        self.base_dir = Path(base_dir)
        self.actor_path = self.base_dir / "actors" / actor_name

        # Load learning book
        self.learning_book = self._load_json(
            self.actor_path / "learning_book.json")

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

                desc = entry.get("description", "")
                topic_match = re.search(r"on:\s*(.+)$", desc)
                topic = topic_match.group(
                    1).strip() if topic_match else "unknown"

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

        for theme, patterns in self.THEME_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                themes.add(theme)

        return themes

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract specific technical keywords."""
        text_lower = text.lower()
        keywords = set()

        for kw in self.KEYWORDS:
            if kw in text_lower:
                keywords.add(kw)

        return keywords

    def _categorize_to_domain(self, text: str) -> str:
        """Categorize text into a broad domain."""
        text_lower = text.lower()

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

    def analyze_theme_introduction_order(self) -> List[tuple]:
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
            if it["themes"]:
                theme = sorted(it["themes"], key=len, reverse=True)[0]
                clusters[theme].append({
                    "iter": it["iter"],
                    "topic": it["topic"][:50],
                })

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

    def get_summary(self) -> dict:
        """Get a summary of scope analysis."""
        expansion = self.analyze_scope_expansion()

        return {
            "total_iterations": len(self.iterations),
            "final_themes": len(expansion["final_themes"]),
            "final_keywords": len(expansion["final_keywords"]),
            "final_domains": len(expansion["final_domains"]),
            "final_topics": expansion["final_topic_count"],
        }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze knowledge scope expansion')
    parser.add_argument('actor', help='Actor name (e.g., steve)')
    args = parser.parse_args()

    print("Loading learning data...")
    analyzer = KnowledgeScopeAnalyzer(args.actor)

    summary = analyzer.get_summary()
    print(f"Parsed {summary['total_iterations']} iterations")
    print(f"  - Themes: {summary['final_themes']}")
    print(f"  - Keywords: {summary['final_keywords']}")
    print(f"  - Domains: {summary['final_domains']}")
    print(f"  - Topics: {summary['final_topics']}")


if __name__ == "__main__":
    main()
