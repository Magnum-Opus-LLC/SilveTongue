#!/usr/bin/env python3
"""
Iteration Analyzer
==================
Analyze learning trends by iteration number, tracking topic transitions,
learning velocity, and mindset evolution.
"""

import json
import yaml
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict


class IterationAnalyzer:
    """Analyze learning trends by iteration number."""
    
    def __init__(self, actor_name: str, base_dir: str = "array"):
        self.actor_name = actor_name
        self.base_dir = Path(base_dir)
        self.actor_path = self.base_dir / "actors" / actor_name
        self.domain_path = self.actor_path / "expertise" / "domain"
        
        # Load all data
        self.learning_book = self._load_json(self.actor_path / "learning_book.json")
        self.principles = self._load_yaml_list(self.domain_path / "principal" / "principles.yaml", "principles")
        self.core_knowledge = self._load_yaml_list(self.domain_path / "knowledge" / "core" / "core_knowledge.yaml", "core_knowledge")
        self.experiences = self._load_yaml_list(self.domain_path / "experience" / "experiences.yaml", "experiences")
        self.challenges = self._load_yaml_list(self.domain_path / "challenge" / "challenges.yaml", "challenges")
        
        # Parse learning history by iteration
        self.iterations = self._parse_iterations()
        
    def _load_json(self, path: Path) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_yaml_list(self, path: Path, key: str) -> list:
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and key in data:
                    return data[key]
                elif isinstance(data, list):
                    return data
        except Exception as e:
            print(f"Error loading {path}: {e}")
        return []
    
    def _parse_iterations(self) -> List[dict]:
        """Parse learning history into iteration-indexed data."""
        iterations = []
        history = self.learning_book.get("learning_history", [])
        
        global_iter = 0
        for entry in history:
            if entry.get("event_type") == "cognitive_loop_update":
                global_iter += 1
                
                desc = entry.get("description", "")
                topic_match = re.search(r"on:\s*(.+)$", desc)
                topic = topic_match.group(1) if topic_match else "unknown"
                
                local_iter_match = re.search(r"iteration\s+(\d+)", desc)
                local_iter = int(local_iter_match.group(1)) if local_iter_match else 0
                
                iterations.append({
                    "global_iter": global_iter,
                    "local_iter": local_iter,
                    "timestamp": entry.get("timestamp", ""),
                    "topic": topic.strip(),
                    "insights": entry.get("insights_gained", []),
                    "mindset_changes": entry.get("mindset_changes", []),
                    "papers_studied": self._extract_papers_count(entry.get("insights_gained", [])),
                    "items_persisted": self._extract_items_persisted(entry.get("insights_gained", [])),
                })
        
        return iterations
    
    def _extract_papers_count(self, insights: list) -> int:
        """Extract papers studied from insights."""
        for insight in insights:
            match = re.search(r"Studied\s+(\d+)\s+papers", str(insight))
            if match:
                return int(match.group(1))
        return 0
    
    def _extract_items_persisted(self, insights: list) -> int:
        """Extract items persisted from insights."""
        for insight in insights:
            match = re.search(r"Persisted\s+(\d+)\s+items", str(insight))
            if match:
                return int(match.group(1))
        return 0
    
    def _categorize_domain(self, topic: str) -> str:
        """Categorize topic into a domain."""
        topic_lower = topic.lower()
        
        if any(w in topic_lower for w in ['edge', 'deploy', 'accelerator', 'npu', '5g', 'iot']):
            return "Edge Deployment"
        elif any(w in topic_lower for w in ['continual', 'drift', 'forget', 'replay']):
            return "Continual Learning"
        elif any(w in topic_lower for w in ['robust', 'adversarial', 'attack', 'reliability']):
            return "Robustness"
        elif any(w in topic_lower for w in ['quantiz', 'prune', 'compress', 'efficient']):
            return "Efficiency"
        elif any(w in topic_lower for w in ['transformer', 'language model', 'attention', 'llm']):
            return "Language Models"
        elif any(w in topic_lower for w in ['multimodal', 'vision', 'cross-modal']):
            return "Multimodal"
        elif any(w in topic_lower for w in ['bias', 'fair', 'misinformation']):
            return "Ethics & Bias"
        elif any(w in topic_lower for w in ['architecture', 'neural', 'network', 'model']):
            return "ML Architecture"
        else:
            return "General"
    
    def _categorize_mindset(self, text: str) -> str:
        """Categorize mindset change."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['efficient', 'latency', 'energy', 'compress', 'prune', 'quantiz']):
            return "Efficiency Focus"
        elif any(w in text_lower for w in ['robust', 'adversarial', 'reliability', 'fault']):
            return "Robustness Focus"
        elif any(w in text_lower for w in ['continual', 'drift', 'forget', 'replay', 'buffer']):
            return "Continual Learning"
        elif any(w in text_lower for w in ['bias', 'fair', 'ethic', 'align']):
            return "Ethics & Fairness"
        elif any(w in text_lower for w in ['modular', 'hybrid', 'architecture', 'design']):
            return "Architecture Design"
        elif any(w in text_lower for w in ['deploy', 'edge', 'production', 'constraint']):
            return "Deployment Awareness"
        elif any(w in text_lower for w in ['theoretical', 'formal', 'bound', 'proof']):
            return "Theoretical Rigor"
        else:
            return "General"
    
    def analyze_topic_timeline(self) -> dict:
        """Analyze when topics were introduced and how long they were studied."""
        topic_data = defaultdict(lambda: {
            "first_iter": float('inf'),
            "last_iter": 0,
            "total_iterations": 0,
            "papers_studied": 0,
            "items_persisted": 0,
            "mindset_changes": 0,
        })
        
        for it in self.iterations:
            topic = it["topic"]
            topic_data[topic]["first_iter"] = min(topic_data[topic]["first_iter"], it["global_iter"])
            topic_data[topic]["last_iter"] = max(topic_data[topic]["last_iter"], it["global_iter"])
            topic_data[topic]["total_iterations"] += 1
            topic_data[topic]["papers_studied"] += it["papers_studied"]
            topic_data[topic]["items_persisted"] += it["items_persisted"]
            topic_data[topic]["mindset_changes"] += len(it["mindset_changes"])
        
        return dict(sorted(topic_data.items(), key=lambda x: x[1]["first_iter"]))
    
    def analyze_learning_velocity(self, window_size: int = 50) -> list:
        """Analyze learning velocity over iteration windows."""
        velocities = []
        
        for i in range(0, len(self.iterations), window_size):
            window = self.iterations[i:i+window_size]
            if not window:
                continue
            
            total_papers = sum(it["papers_studied"] for it in window)
            total_items = sum(it["items_persisted"] for it in window)
            total_mindset = sum(len(it["mindset_changes"]) for it in window)
            unique_topics = len(set(it["topic"] for it in window))
            
            velocities.append({
                "iter_start": window[0]["global_iter"],
                "iter_end": window[-1]["global_iter"],
                "papers_per_iter": total_papers / len(window),
                "items_per_iter": total_items / len(window),
                "mindset_changes_per_iter": total_mindset / len(window),
                "topic_diversity": unique_topics,
                "actual_iterations": len(window),
            })
        
        return velocities
    
    def analyze_topic_transitions(self) -> list:
        """Analyze how topics transitioned over iterations."""
        transitions = []
        prev_topic = None
        
        for it in self.iterations:
            if prev_topic and prev_topic != it["topic"]:
                transitions.append({
                    "iteration": it["global_iter"],
                    "from_topic": prev_topic,
                    "to_topic": it["topic"],
                })
            prev_topic = it["topic"]
        
        return transitions
    
    def analyze_mindset_evolution(self) -> dict:
        """Track how mindset evolved over iterations."""
        mindset_by_iter = []
        
        for it in self.iterations:
            for change in it["mindset_changes"]:
                mindset_by_iter.append({
                    "iteration": it["global_iter"],
                    "topic": it["topic"],
                    "change": change[:150] + "..." if len(change) > 150 else change,
                    "category": self._categorize_mindset(change),
                })
        
        category_timeline = defaultdict(list)
        for m in mindset_by_iter:
            category_timeline[m["category"]].append(m["iteration"])
        
        return {
            "all_changes": mindset_by_iter,
            "by_category": {k: {"count": len(v), "first": min(v), "last": max(v), "iterations": v} 
                          for k, v in category_timeline.items()},
        }
    
    def analyze_domain_depth_over_time(self) -> dict:
        """Track how knowledge depth evolved by iteration buckets."""
        bucket_size = 100
        buckets = defaultdict(lambda: defaultdict(int))
        
        for it in self.iterations:
            bucket = (it["global_iter"] - 1) // bucket_size * bucket_size + 1
            domain = self._categorize_domain(it["topic"])
            buckets[bucket][domain] += it["items_persisted"]
        
        return dict(sorted(buckets.items()))
    
    def find_breakthrough_iterations(self) -> list:
        """Find iterations with significant mindset shifts or high productivity."""
        breakthroughs = []
        
        avg_items = sum(it["items_persisted"] for it in self.iterations) / len(self.iterations) if self.iterations else 0
        avg_mindset = sum(len(it["mindset_changes"]) for it in self.iterations) / len(self.iterations) if self.iterations else 0
        
        for it in self.iterations:
            if it["items_persisted"] > avg_items * 2:
                breakthroughs.append({
                    "iteration": it["global_iter"],
                    "topic": it["topic"],
                    "type": "High Productivity",
                    "metric": f"{it['items_persisted']} items (avg: {avg_items:.1f})",
                })
            
            if len(it["mindset_changes"]) >= 3:
                breakthroughs.append({
                    "iteration": it["global_iter"],
                    "topic": it["topic"],
                    "type": "Major Mindset Shift",
                    "metric": f"{len(it['mindset_changes'])} changes",
                    "changes": it["mindset_changes"][:2],
                })
        
        return sorted(breakthroughs, key=lambda x: x["iteration"])
    
    def analyze_topic_saturation(self) -> dict:
        """Detect when topics reached saturation (diminishing returns)."""
        topic_productivity = defaultdict(list)
        
        for it in self.iterations:
            topic_productivity[it["topic"]].append({
                "iter": it["global_iter"],
                "items": it["items_persisted"],
                "mindset": len(it["mindset_changes"]),
            })
        
        saturation = {}
        for topic, data in topic_productivity.items():
            if len(data) >= 5:
                mid = len(data) // 2
                first_half_avg = sum(d["items"] for d in data[:mid]) / mid
                second_half_avg = sum(d["items"] for d in data[mid:]) / (len(data) - mid)
                
                if first_half_avg > 0 and second_half_avg / first_half_avg < 0.7:
                    saturation[topic] = {
                        "total_iterations": len(data),
                        "saturation_point": data[mid]["iter"],
                        "early_productivity": first_half_avg,
                        "late_productivity": second_half_avg,
                        "decline_ratio": second_half_avg / first_half_avg,
                    }
        
        return saturation
    
    def get_summary(self) -> dict:
        """Get a summary of iterations."""
        total_papers = sum(it["papers_studied"] for it in self.iterations)
        total_items = sum(it["items_persisted"] for it in self.iterations)
        total_mindset = sum(len(it["mindset_changes"]) for it in self.iterations)
        unique_topics = len(set(it["topic"] for it in self.iterations))
        
        return {
            "total_iterations": len(self.iterations),
            "total_papers": total_papers,
            "total_items": total_items,
            "total_mindset_changes": total_mindset,
            "unique_topics": unique_topics,
            "papers_per_iter": total_papers / len(self.iterations) if self.iterations else 0,
            "items_per_iter": total_items / len(self.iterations) if self.iterations else 0,
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze iteration trends')
    parser.add_argument('actor', help='Actor name (e.g., steve)')
    args = parser.parse_args()
    
    print("Loading learning data...")
    analyzer = IterationAnalyzer(args.actor)
    
    summary = analyzer.get_summary()
    print(f"Parsed {summary['total_iterations']} cognitive loop iterations")
    print(f"  - Papers: {summary['total_papers']}")
    print(f"  - Items persisted: {summary['total_items']}")
    print(f"  - Mindset changes: {summary['total_mindset_changes']}")
    print(f"  - Unique topics: {summary['unique_topics']}")


if __name__ == "__main__":
    main()
