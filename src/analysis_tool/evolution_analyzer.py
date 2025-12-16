#!/usr/bin/env python3
"""
Knowledge Evolution Analyzer
=============================
Basic knowledge evolution analysis including growth metrics,
topic patterns, and hourly activity tracking.
"""

import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Set


class KnowledgeEvolutionAnalyzer:
    """Analyze knowledge evolution and growth patterns."""
    
    TOPIC_PATTERNS = [
        r'transformer[s]?', r'attention', r'neural network[s]?', r'deep learning',
        r'machine learning', r'CNN[s]?', r'RNN[s]?', r'GNN[s]?', r'graph neural',
        r'language model[s]?', r'NLP', r'vision', r'multimodal', r'edge AI',
        r'quantization', r'pruning', r'compression', r'deployment',
        r'drift', r'continual learning', r'replay buffer', r'stability',
        r'plasticity', r'bias', r'fairness', r'robustness', r'adversarial',
        r'tensor[s]?', r'optimization', r'gradient', r'convergence',
        r'thermodynamic', r'energy', r'latency', r'efficiency',
        r'scaling law[s]?', r'pre-training', r'fine-tuning', r'transfer learning',
        r'knowledge graph', r'symbolic', r'reasoning', r'inference',
        r'accelerator[s]?', r'GPU', r'TPU', r'ASIC', r'NPU',
    ]
    
    MINDSET_CATEGORIES = {
        "architectural_insights": ['architecture', 'transformer', 'layer', 'attention'],
        "efficiency_principles": ['efficient', 'energy', 'latency', 'compression'],
        "theoretical_connections": ['theorem', 'bound', 'convergence', 'theory'],
        "practical_applications": ['deploy', 'production', 'edge', 'real-world'],
        "cross_domain_links": ['cross', 'multimodal', 'hybrid', 'unified'],
    }
    
    def __init__(self, actor_name: str, base_dir: str = "array"):
        self.actor_name = actor_name
        self.base_dir = Path(base_dir)
        self.actor_path = self.base_dir / "actors" / actor_name
        self.domain_path = self.actor_path / "expertise" / "domain"
        
        # Load learning book
        self.learning_book = self._load_json(self.actor_path / "learning_book.json")
        
    def _load_json(self, path: Path) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_yaml_file(self, file_path: Path):
        """Load a YAML file."""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _count_yaml_entries(self, file_path: Path) -> int:
        """Count entries in a YAML list file."""
        try:
            data = self._load_yaml_file(file_path)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        return len(value)
            elif isinstance(data, list):
                return len(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        return 0
    
    def _extract_topics(self, text: str) -> Set[str]:
        """Extract topic keywords from text."""
        topics = set()
        text_lower = text.lower()
        
        for pattern in self.TOPIC_PATTERNS:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                if match:
                    # Remove trailing 's' for singular form (not all 's' characters)
                    topic = match.group().strip()
                    if topic.endswith('s') and not topic.endswith('ss'):
                        topic = topic[:-1]
                    topics.add(topic)
        
        return topics
    
    def _extract_topics_from_yaml(self, file_path: Path) -> Counter:
        """Extract and count topics from a YAML file."""
        topics = Counter()
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                extracted = self._extract_topics(content)
                topics.update(extracted)
        except Exception as e:
            print(f"Error extracting topics from {file_path}: {e}")
        return topics
    
    def analyze_learning_history(self) -> dict:
        """Analyze the learning history for patterns and growth."""
        history = self.learning_book.get("learning_history", [])
        
        hourly_activity = defaultdict(list)
        topics_over_time = defaultdict(set)
        event_types = Counter()
        insights_by_hour = defaultdict(list)
        
        for entry in history:
            timestamp_str = entry.get("timestamp", "")
            event_type = entry.get("event_type", "unknown")
            description = entry.get("description", "")
            insights = entry.get("insights_gained", [])
            
            event_types[event_type] += 1
            
            try:
                if timestamp_str:
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    hour_key = dt.strftime("%Y-%m-%d %H:00")
                    hourly_activity[hour_key].append(entry)
                    insights_by_hour[hour_key].extend(insights)
                    
                    topics = self._extract_topics(description)
                    topics_over_time[hour_key].update(topics)
            except:
                pass
        
        return {
            "total_events": len(history),
            "event_types": dict(event_types),
            "hourly_activity": {k: len(v) for k, v in sorted(hourly_activity.items())},
            "topics_over_time": {k: list(v) for k, v in sorted(topics_over_time.items())},
            "insights_by_hour": {k: len(v) for k, v in sorted(insights_by_hour.items())},
        }
    
    def analyze_knowledge_domains(self) -> dict:
        """Analyze the knowledge domain files for depth and breadth."""
        results = {
            "principles": {"count": 0, "topics": Counter()},
            "core_knowledge": {"count": 0, "topics": Counter()},
            "experiences": {"count": 0, "topics": Counter()},
            "challenges": {"count": 0, "topics": Counter()},
        }
        
        # Analyze principles
        principles_path = self.domain_path / "principal" / "principles.yaml"
        if principles_path.exists():
            results["principles"]["count"] = self._count_yaml_entries(principles_path)
            results["principles"]["topics"] = self._extract_topics_from_yaml(principles_path)
        
        # Analyze core knowledge
        core_path = self.domain_path / "knowledge" / "core" / "core_knowledge.yaml"
        if core_path.exists():
            results["core_knowledge"]["count"] = self._count_yaml_entries(core_path)
            results["core_knowledge"]["topics"] = self._extract_topics_from_yaml(core_path)
        
        # Analyze experiences
        exp_path = self.domain_path / "experience" / "experiences.yaml"
        if exp_path.exists():
            results["experiences"]["count"] = self._count_yaml_entries(exp_path)
            results["experiences"]["topics"] = self._extract_topics_from_yaml(exp_path)
        
        # Analyze challenges
        chal_path = self.domain_path / "challenge" / "challenges.yaml"
        if chal_path.exists():
            results["challenges"]["count"] = self._count_yaml_entries(chal_path)
            results["challenges"]["topics"] = self._extract_topics_from_yaml(chal_path)
        
        return results
    
    def analyze_mindset_evolution(self) -> dict:
        """Analyze how mindset evolved during learning."""
        history = self.learning_book.get("learning_history", [])
        
        mindset_changes = []
        for entry in history:
            changes = entry.get("mindset_changes", [])
            if changes:
                mindset_changes.extend(changes)
        
        categories = {k: [] for k in self.MINDSET_CATEGORIES.keys()}
        
        for change in mindset_changes:
            change_lower = change.lower()
            for category, keywords in self.MINDSET_CATEGORIES.items():
                if any(k in change_lower for k in keywords):
                    categories[category].append(change)
        
        return {
            "total_mindset_changes": len(mindset_changes),
            "categories": {k: len(v) for k, v in categories.items()},
            "sample_changes": mindset_changes[:10] if mindset_changes else [],
        }
    
    def get_growth_statistics(self) -> dict:
        """Get overall growth statistics."""
        meta = self.learning_book.get("meta", {})
        version = meta.get("version", 0)
        
        domains = self.analyze_knowledge_domains()
        
        total_knowledge = sum(d['count'] for d in domains.values())
        
        all_topics = Counter()
        for domain, data in domains.items():
            all_topics.update(data['topics'])
        
        return {
            "total_iterations": version,
            "iterations_per_hour": version / 24 if version > 0 else 0,
            "total_knowledge_items": total_knowledge,
            "unique_topics": len(all_topics),
            "top_topics": all_topics.most_common(20),
            "knowledge_by_domain": {k: v['count'] for k, v in domains.items()},
        }
    
    def generate_report(self) -> str:
        """Generate a text-based analysis report."""
        lines = []
        
        stats = self.get_growth_statistics()
        domains = self.analyze_knowledge_domains()
        mindset = self.analyze_mindset_evolution()
        
        lines.append("=" * 80)
        lines.append(f"KNOWLEDGE EVOLUTION ANALYSIS - {self.actor_name.upper()}")
        lines.append("=" * 80)
        
        lines.append("\nOVERALL GROWTH STATISTICS")
        lines.append("-" * 40)
        lines.append(f"  Total Learning Iterations: {stats['total_iterations']:,}")
        lines.append(f"  Average Iterations/Hour:   {stats['iterations_per_hour']:.1f}")
        lines.append(f"  Total Knowledge Items:     {stats['total_knowledge_items']:,}")
        lines.append(f"  Unique Topics Covered:     {stats['unique_topics']}")
        
        lines.append("\nKNOWLEDGE BY DOMAIN")
        lines.append("-" * 40)
        for domain, count in stats['knowledge_by_domain'].items():
            bar = "#" * min(50, count // 20)
            lines.append(f"  {domain:<18} {bar} ({count})")
        
        lines.append("\nTOP TOPICS")
        lines.append("-" * 40)
        for topic, count in stats['top_topics'][:15]:
            bar = "#" * min(30, count)
            lines.append(f"  {topic:<20} {bar} ({count})")
        
        lines.append("\nMINDSET EVOLUTION")
        lines.append("-" * 40)
        lines.append(f"  Total Mindset Changes: {mindset['total_mindset_changes']}")
        for category, count in mindset['categories'].items():
            lines.append(f"  {category.replace('_', ' ').title()}: {count}")
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze knowledge evolution')
    parser.add_argument('actor', help='Actor name (e.g., steve)')
    args = parser.parse_args()
    
    print("Loading Steve's learning data...")
    analyzer = KnowledgeEvolutionAnalyzer(args.actor)
    
    report = analyzer.generate_report()
    print(report)


if __name__ == "__main__":
    main()
