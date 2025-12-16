#!/usr/bin/env python3
"""
Iteration-Based Trend Analysis for Steve's Learning Journey

Analyzes knowledge evolution based on cognitive loop iteration numbers,
tracking domain emergence, topic shifts, and learning velocity.
"""

import json
import yaml
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, List, Tuple


class IterationAnalyzer:
    """Analyze learning trends by iteration number."""
    
    def __init__(self, actor_path: Path):
        self.actor_path = actor_path
        self.domain_path = actor_path / "expertise" / "domain"
        
        # Load all data
        self.learning_book = self._load_json(actor_path / "learning_book.json")
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
                
                # Extract topic from description
                desc = entry.get("description", "")
                topic_match = re.search(r"on:\s*(.+)$", desc)
                topic = topic_match.group(1) if topic_match else "unknown"
                
                # Extract local iteration number
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
        
        # Group by category
        category_timeline = defaultdict(list)
        for m in mindset_by_iter:
            category_timeline[m["category"]].append(m["iteration"])
        
        return {
            "all_changes": mindset_by_iter,
            "by_category": {k: {"count": len(v), "first": min(v), "last": max(v), "iterations": v} 
                          for k, v in category_timeline.items()},
        }
    
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
    
    def analyze_domain_depth_over_time(self) -> dict:
        """Track how knowledge depth evolved by iteration buckets."""
        # Group iterations into buckets
        bucket_size = 100
        buckets = defaultdict(lambda: defaultdict(int))
        
        for it in self.iterations:
            bucket = (it["global_iter"] - 1) // bucket_size * bucket_size + 1
            
            # Categorize topic into domain
            domain = self._categorize_domain(it["topic"])
            buckets[bucket][domain] += it["items_persisted"]
        
        return dict(sorted(buckets.items()))
    
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
    
    def find_breakthrough_iterations(self) -> list:
        """Find iterations with significant mindset shifts or high productivity."""
        breakthroughs = []
        
        # Calculate baseline metrics
        avg_items = sum(it["items_persisted"] for it in self.iterations) / len(self.iterations)
        avg_mindset = sum(len(it["mindset_changes"]) for it in self.iterations) / len(self.iterations)
        
        for it in self.iterations:
            # High productivity iteration
            if it["items_persisted"] > avg_items * 2:
                breakthroughs.append({
                    "iteration": it["global_iter"],
                    "topic": it["topic"],
                    "type": "High Productivity",
                    "metric": f"{it['items_persisted']} items (avg: {avg_items:.1f})",
                })
            
            # Multiple mindset changes
            if len(it["mindset_changes"]) >= 3:
                breakthroughs.append({
                    "iteration": it["global_iter"],
                    "topic": it["topic"],
                    "type": "Major Mindset Shift",
                    "metric": f"{len(it['mindset_changes'])} changes",
                    "changes": it["mindset_changes"][:2],  # First 2
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
                # Compare first half vs second half productivity
                mid = len(data) // 2
                first_half_avg = sum(d["items"] for d in data[:mid]) / mid
                second_half_avg = sum(d["items"] for d in data[mid:]) / (len(data) - mid)
                
                # Detect diminishing returns
                if first_half_avg > 0 and second_half_avg / first_half_avg < 0.7:
                    saturation[topic] = {
                        "total_iterations": len(data),
                        "saturation_point": data[mid]["iter"],
                        "early_productivity": first_half_avg,
                        "late_productivity": second_half_avg,
                        "decline_ratio": second_half_avg / first_half_avg,
                    }
        
        return saturation


def generate_trend_report(analyzer: IterationAnalyzer):
    """Generate comprehensive trend report."""
    
    total_iterations = len(analyzer.iterations)
    
    print("\n" + "="*95)
    print("📈 ITERATION-BASED TREND ANALYSIS - STEVE'S LEARNING JOURNEY")
    print("="*95)
    print(f"\n  Total Cognitive Loop Iterations: {total_iterations}")
    print(f"  Learning Book Version: {analyzer.learning_book['meta']['version']}")
    
    # 1. Topic Introduction Timeline
    print("\n\n" + "─"*95)
    print("📅 TOPIC INTRODUCTION TIMELINE")
    print("─"*95)
    
    topic_timeline = analyzer.analyze_topic_timeline()
    
    print("\n  Topic                                                    │ First │ Last  │ Iters │ Papers │ Items │ Mindset")
    print("  " + "-"*110)
    
    for topic, data in topic_timeline.items():
        short_topic = topic[:55] + "..." if len(topic) > 55 else topic
        print(f"  {short_topic:<58} │ {data['first_iter']:>5} │ {data['last_iter']:>5} │ {data['total_iterations']:>5} │ {data['papers_studied']:>6} │ {data['items_persisted']:>5} │ {data['mindset_changes']:>5}")
    
    # 2. Learning Velocity Trend
    print("\n\n" + "─"*95)
    print("🚀 LEARNING VELOCITY OVER TIME")
    print("─"*95)
    
    velocities = analyzer.analyze_learning_velocity(window_size=100)
    
    print("\n  Iterations     │ Papers/Iter │ Items/Iter │ Mindset/Iter │ Topics │ Velocity Chart")
    print("  " + "-"*90)
    
    max_items = max(v["items_per_iter"] for v in velocities) if velocities else 1
    
    for v in velocities:
        bar_len = int(20 * v["items_per_iter"] / max_items)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {v['iter_start']:>4} - {v['iter_end']:<4}  │ {v['papers_per_iter']:>11.2f} │ {v['items_per_iter']:>10.2f} │ {v['mindset_changes_per_iter']:>12.2f} │ {v['topic_diversity']:>6} │ {bar}")
    
    # 3. Topic Transitions
    print("\n\n" + "─"*95)
    print("🔄 TOPIC TRANSITIONS (Major Shifts)")
    print("─"*95)
    
    transitions = analyzer.analyze_topic_transitions()
    
    print(f"\n  Total topic transitions: {len(transitions)}")
    print("\n  Iter │ From → To")
    print("  " + "-"*85)
    
    for t in transitions[:30]:  # Show first 30
        from_short = t["from_topic"][:35] + "..." if len(t["from_topic"]) > 35 else t["from_topic"]
        to_short = t["to_topic"][:35] + "..." if len(t["to_topic"]) > 35 else t["to_topic"]
        print(f"  {t['iteration']:>4} │ {from_short:<38} → {to_short}")
    
    if len(transitions) > 30:
        print(f"  ... and {len(transitions) - 30} more transitions")
    
    # 4. Domain Knowledge Growth Over Iterations
    print("\n\n" + "─"*95)
    print("📊 DOMAIN KNOWLEDGE GROWTH BY ITERATION BUCKETS")
    print("─"*95)
    
    domain_growth = analyzer.analyze_domain_depth_over_time()
    
    # Get all domains
    all_domains = set()
    for bucket_data in domain_growth.values():
        all_domains.update(bucket_data.keys())
    all_domains = sorted(all_domains)
    
    # Print header
    header = "  Iters    │ " + " │ ".join(f"{d[:12]:<12}" for d in all_domains[:6])
    print(f"\n{header}")
    print("  " + "-"*100)
    
    for bucket, data in domain_growth.items():
        row = f"  {bucket:>4}-{bucket+99:<4} │ "
        for domain in all_domains[:6]:
            count = data.get(domain, 0)
            row += f"{count:>12} │ "
        print(row)
    
    # 5. Mindset Evolution Categories
    print("\n\n" + "─"*95)
    print("🧠 MINDSET EVOLUTION BY CATEGORY")
    print("─"*95)
    
    mindset_data = analyzer.analyze_mindset_evolution()
    by_category = mindset_data["by_category"]
    
    print("\n  Category              │ Count │ First Iter │ Last Iter │ Spread │ Timeline")
    print("  " + "-"*90)
    
    for cat, data in sorted(by_category.items(), key=lambda x: -x[1]["count"]):
        spread = data["last"] - data["first"]
        # Create mini timeline visualization
        if analyzer.iterations:
            total_iters = analyzer.iterations[-1]["global_iter"]
            timeline = [" "] * 30
            for it in data["iterations"]:
                pos = int(29 * it / total_iters)
                timeline[pos] = "█"
            timeline_str = "".join(timeline)
        else:
            timeline_str = ""
        
        print(f"  {cat:<21} │ {data['count']:>5} │ {data['first']:>10} │ {data['last']:>9} │ {spread:>6} │ [{timeline_str}]")
    
    # 6. Breakthrough Iterations
    print("\n\n" + "─"*95)
    print("⭐ BREAKTHROUGH ITERATIONS")
    print("─"*95)
    
    breakthroughs = analyzer.find_breakthrough_iterations()
    
    print(f"\n  Found {len(breakthroughs)} breakthrough moments:")
    print()
    
    for b in breakthroughs[:20]:
        short_topic = b["topic"][:40] + "..." if len(b["topic"]) > 40 else b["topic"]
        print(f"  Iter {b['iteration']:>4} │ {b['type']:<20} │ {short_topic}")
        print(f"           │ Metric: {b['metric']}")
        if "changes" in b:
            for change in b["changes"]:
                print(f"           │ → {change[:70]}...")
        print()
    
    # 7. Topic Saturation Analysis
    print("\n" + "─"*95)
    print("📉 TOPIC SATURATION DETECTION (Diminishing Returns)")
    print("─"*95)
    
    saturation = analyzer.analyze_topic_saturation()
    
    if saturation:
        print("\n  Topic                                          │ Iters │ Saturation │ Early Prod │ Late Prod │ Decline")
        print("  " + "-"*105)
        
        for topic, data in sorted(saturation.items(), key=lambda x: x[1]["decline_ratio"]):
            short_topic = topic[:45] + "..." if len(topic) > 45 else topic
            print(f"  {short_topic:<48} │ {data['total_iterations']:>5} │ iter {data['saturation_point']:>5} │ {data['early_productivity']:>10.1f} │ {data['late_productivity']:>9.1f} │ {data['decline_ratio']*100:>5.0f}%")
    else:
        print("\n  No significant topic saturation detected yet.")
    
    # 8. Summary Statistics
    print("\n\n" + "─"*95)
    print("📋 TREND SUMMARY")
    print("─"*95)
    
    total_papers = sum(it["papers_studied"] for it in analyzer.iterations)
    total_items = sum(it["items_persisted"] for it in analyzer.iterations)
    total_mindset = sum(len(it["mindset_changes"]) for it in analyzer.iterations)
    unique_topics = len(set(it["topic"] for it in analyzer.iterations))
    
    # Find peak productivity period
    velocities = analyzer.analyze_learning_velocity(window_size=100)
    if velocities:
        peak = max(velocities, key=lambda x: x["items_per_iter"])
        peak_period = f"iterations {peak['iter_start']}-{peak['iter_end']}"
    else:
        peak_period = "N/A"
    
    print(f"""
  OVERALL METRICS:
  ────────────────
  • Total iterations: {total_iterations}
  • Total papers studied: {total_papers}
  • Total knowledge items persisted: {total_items}
  • Total mindset changes: {total_mindset}
  • Unique topics explored: {unique_topics}
  
  AVERAGES:
  ─────────
  • Papers per iteration: {total_papers/total_iterations:.2f}
  • Items per iteration: {total_items/total_iterations:.2f}
  • Mindset changes per iteration: {total_mindset/total_iterations:.2f}
  
  LEARNING PATTERNS:
  ──────────────────
  • Peak productivity period: {peak_period}
  • Topic transitions: {len(transitions)}
  • Topics showing saturation: {len(saturation)}
  • Breakthrough iterations: {len(breakthroughs)}
  
  DOMINANT MINDSET SHIFTS:
  ────────────────────────""")
    
    for cat, data in sorted(by_category.items(), key=lambda x: -x[1]["count"])[:5]:
        print(f"  • {cat}: {data['count']} changes")
    
    print("\n" + "="*95)
    print("Analysis Complete!")
    print("="*95)


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent
    actor_path = base_path / "array" / "actors" / "steve"
    
    print("Loading learning data...")
    analyzer = IterationAnalyzer(actor_path)
    
    print(f"Parsed {len(analyzer.iterations)} cognitive loop iterations")
    
    generate_trend_report(analyzer)


if __name__ == "__main__":
    main()
