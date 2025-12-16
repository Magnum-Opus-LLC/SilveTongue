#!/usr/bin/env python3
"""
Knowledge Analyzer
==================
Deep analyzer for knowledge evolution patterns including principles,
experiences, challenges, and their interconnections.
"""

import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Optional, List, Dict


class KnowledgeAnalyzer:
    """Deep analyzer for knowledge evolution patterns."""
    
    DOMAIN_KEYWORDS = {
        "ML Architecture": ['transformer', 'attention', 'cnn', 'rnn', 'gnn', 'neural network', 'architecture', 'layer'],
        "Edge AI & Efficiency": ['edge', 'latency', 'energy', 'quantization', 'pruning', 'compression', 'accelerator', 'npu', 'tpu'],
        "Continual Learning": ['continual', 'replay buffer', 'forgetting', 'drift', 'stability', 'plasticity', 'online learning'],
        "Bias & Fairness": ['bias', 'fairness', 'misinformation', 'ethical', 'societal'],
        "Multimodal & Cross-Domain": ['multimodal', 'vision-language', 'cross-modal', 'unified', 'graph neural'],
        "Tensor & Mathematics": ['tensor', 'contraction', 'decomposition', 'manifold', 'differential geometry'],
        "Reliability & Robustness": ['reliability', 'robustness', 'adversarial', 'degradation', 'fault', 'thermal'],
        "Scaling & Trade-offs": ['scaling law', 'trade-off', 'bias-variance', 'diminishing return'],
        "Neural-Symbolic": ['symbolic', 'reasoning', 'logic', 'knowledge graph', 'rule'],
        "Optimization": ['gradient', 'convergence', 'optimizer', 'loss', 'regularization'],
        "Thermodynamics": ['thermodynamic', 'landauer', 'energy conservation', 'entropy'],
    }
    
    KEY_TERMS = [
        'transformer', 'attention', 'quantization', 'pruning', 'compression',
        'edge', 'latency', 'energy', 'drift', 'continual', 'forgetting',
        'replay', 'stability', 'plasticity', 'bias', 'fairness', 'robustness',
        'adversarial', 'multimodal', 'scaling', 'convergence', 'optimization',
        'tensor', 'symbolic', 'reasoning', 'graph', 'neural', 'gradient',
        'regularization', 'normalization', 'sparsity', 'buffer', 'threshold',
        'calibration', 'uncertainty', 'reliability', 'deployment', 'inference',
    ]
    
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
    
    def extract_timestamp(self, entry) -> Optional[datetime]:
        """Extract timestamp from an entry."""
        if isinstance(entry, dict):
            text = str(entry.get('description', '')) + str(entry.get('topic', '')) + str(entry.get('notes', ''))
        else:
            text = str(entry)
        
        patterns = [
            r'\[(\d{4}-\d{2}-\d{2})\]',
            r'\((\d{4}-\d{2}-\d{2})\)',
            r'(\d{4}[-‑]\d{2}[-‑]\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    date_str = match.group(1).replace('‑', '-')
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    pass
        return None
    
    def classify_entry_source(self, entry) -> str:
        """Classify if entry is seed data or learned during study."""
        if isinstance(entry, dict):
            text = str(entry.get('description', '')) + str(entry.get('topic', '')) + str(entry.get('notes', ''))
        else:
            text = str(entry)
        
        if 'cognitive_loop' in text.lower() or '[Source:' in text:
            return "learned"
        return "seed"
    
    def categorize_domain(self, text: str) -> List[str]:
        """Categorize text into knowledge domains."""
        text_lower = text.lower()
        domains = []
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                domains.append(domain)
        
        return domains if domains else ["General"]
    
    def extract_key_phrases(self, text: str) -> set:
        """Extract key phrases/themes from text."""
        text_lower = text.lower()
        found = set()
        
        for term in self.KEY_TERMS:
            if term in text_lower:
                found.add(term)
        
        return found
    
    def analyze_principle_timeline(self) -> dict:
        """Analyze when principles were introduced and their domains."""
        timeline = defaultdict(list)
        domain_first_appearance = {}
        
        for principle in self.principles:
            text = str(principle)
            timestamp = self.extract_timestamp(principle)
            domains = self.categorize_domain(text)
            
            date_key = timestamp.strftime("%Y-%m-%d") if timestamp else "Initial"
            
            for domain in domains:
                timeline[date_key].append({
                    "domain": domain,
                    "principle": text[:150] + "..." if len(text) > 150 else text,
                })
                
                if domain not in domain_first_appearance:
                    domain_first_appearance[domain] = date_key
        
        return {
            "timeline": dict(sorted(timeline.items())),
            "domain_first_appearance": domain_first_appearance,
            "total_principles": len(self.principles),
        }
    
    def analyze_challenge_evolution(self) -> dict:
        """Analyze how challenges evolved and which got addressed."""
        challenge_data = []
        challenge_topics = defaultdict(list)
        
        for challenge in self.challenges:
            if isinstance(challenge, dict):
                topic = challenge.get('topic', 'Unknown')
                description = challenge.get('description', '')
                status = challenge.get('status', 'open')
                notes = challenge.get('notes', '')
                timestamp = self.extract_timestamp(challenge)
                
                challenge_data.append({
                    "topic": topic,
                    "description": description,
                    "status": status,
                    "notes": notes,
                    "timestamp": timestamp,
                    "domains": self.categorize_domain(topic + " " + description),
                })
            else:
                text = str(challenge)
                timestamp = self.extract_timestamp(text)
                domains = self.categorize_domain(text)
                
                if " - " in text:
                    parts = text.split(" - ", 1)
                    assumption = parts[0]
                    counter = parts[1] if len(parts) > 1 else ""
                else:
                    assumption = text
                    counter = ""
                
                challenge_data.append({
                    "topic": "General",
                    "description": assumption[:200],
                    "counter_evidence": counter[:200] if counter else None,
                    "timestamp": timestamp,
                    "domains": domains,
                })
                
                for domain in domains:
                    challenge_topics[domain].append(text[:100])
        
        return {
            "total_challenges": len(challenge_data),
            "challenges_by_domain": {k: len(v) for k, v in challenge_topics.items()},
            "sample_challenges": challenge_data[:50],
        }
    
    def find_challenge_resolutions(self) -> list:
        """Find challenges that appear to have been addressed by later learning."""
        resolutions = []
        
        challenge_themes = []
        for challenge in self.challenges[:500]:
            if isinstance(challenge, dict):
                text = challenge.get('topic', '') + " " + challenge.get('description', '')
            else:
                text = str(challenge)
            
            timestamp = self.extract_timestamp(text)
            themes = self.extract_key_phrases(text)
            challenge_themes.append({
                "text": text[:200],
                "timestamp": timestamp,
                "themes": themes,
            })
        
        for ch in challenge_themes[:100]:
            if not ch["themes"]:
                continue
                
            for exp in self.experiences:
                exp_text = str(exp)
                exp_timestamp = self.extract_timestamp(exp_text)
                
                if exp_timestamp and ch["timestamp"]:
                    if exp_timestamp > ch["timestamp"]:
                        exp_themes = self.extract_key_phrases(exp_text)
                        overlap = ch["themes"] & exp_themes
                        
                        if len(overlap) >= 2:
                            resolutions.append({
                                "challenge": ch["text"][:150],
                                "challenge_date": ch["timestamp"].strftime("%Y-%m-%d") if ch["timestamp"] else "Unknown",
                                "resolution": exp_text[:150],
                                "resolution_date": exp_timestamp.strftime("%Y-%m-%d"),
                                "shared_themes": list(overlap)[:5],
                            })
                            break
        
        return resolutions[:50]
    
    def analyze_knowledge_progression(self) -> dict:
        """Analyze the progression of knowledge building."""
        progression = defaultdict(lambda: {"principles": 0, "experiences": 0, "challenges": 0, "knowledge": 0})
        source_counts = {
            "seed": {"principles": 0, "experiences": 0, "challenges": 0, "knowledge": 0},
            "learned": {"principles": 0, "experiences": 0, "challenges": 0, "knowledge": 0}
        }
        
        for p in self.principles:
            ts = self.extract_timestamp(p)
            source = self.classify_entry_source(p)
            source_counts[source]["principles"] += 1
            if ts:
                date = ts.strftime("%Y-%m-%d")
            elif source == "learned":
                date = "Learned (undated)"
            else:
                date = "Seed Data"
            progression[date]["principles"] += 1
        
        for e in self.experiences:
            ts = self.extract_timestamp(e)
            source = self.classify_entry_source(e)
            source_counts[source]["experiences"] += 1
            if ts:
                date = ts.strftime("%Y-%m-%d")
            elif source == "learned":
                date = "Learned (undated)"
            else:
                date = "Seed Data"
            progression[date]["experiences"] += 1
        
        for c in self.challenges:
            ts = self.extract_timestamp(c)
            source = self.classify_entry_source(c)
            source_counts[source]["challenges"] += 1
            if ts:
                date = ts.strftime("%Y-%m-%d")
            elif source == "learned":
                date = "Learned (undated)"
            else:
                date = "Seed Data"
            progression[date]["challenges"] += 1
        
        for k in self.core_knowledge:
            ts = self.extract_timestamp(k)
            source = self.classify_entry_source(k)
            source_counts[source]["knowledge"] += 1
            if ts:
                date = ts.strftime("%Y-%m-%d")
            elif source == "learned":
                date = "Learned (undated)"
            else:
                date = "Seed Data"
            progression[date]["knowledge"] += 1
        
        def sort_key(item):
            if item[0] == "Seed Data":
                return "0"
            elif item[0] == "Learned (undated)":
                return "9999"
            else:
                return item[0]
        
        return {
            "by_date": dict(sorted(progression.items(), key=sort_key)),
            "by_source": source_counts
        }
    
    def analyze_domain_depth(self) -> dict:
        """Analyze depth of knowledge in each domain."""
        domain_depth = defaultdict(lambda: {
            "principles": [],
            "experiences": [],
            "challenges": [],
            "knowledge": [],
        })
        
        for p in self.principles:
            text = str(p)
            for domain in self.categorize_domain(text):
                domain_depth[domain]["principles"].append(text[:100])
        
        for e in self.experiences:
            text = str(e)
            for domain in self.categorize_domain(text):
                domain_depth[domain]["experiences"].append(text[:100])
        
        for c in self.challenges:
            text = str(c) if not isinstance(c, dict) else c.get('description', '')
            for domain in self.categorize_domain(text):
                domain_depth[domain]["challenges"].append(text[:100])
        
        for k in self.core_knowledge:
            text = str(k)
            for domain in self.categorize_domain(text):
                domain_depth[domain]["knowledge"].append(text[:100])
        
        depth_scores = {}
        for domain, data in domain_depth.items():
            depth_scores[domain] = {
                "principles_count": len(data["principles"]),
                "experiences_count": len(data["experiences"]),
                "challenges_count": len(data["challenges"]),
                "knowledge_count": len(data["knowledge"]),
                "total_items": sum(len(v) for v in data.values()),
                "depth_score": len(data["principles"]) * 2 + len(data["experiences"]) + len(data["knowledge"]) * 1.5,
            }
        
        return dict(sorted(depth_scores.items(), key=lambda x: -x[1]["depth_score"]))
    
    def find_emerging_themes(self) -> dict:
        """Identify themes that emerged and grew over time."""
        theme_timeline = defaultdict(lambda: defaultdict(int))
        
        all_entries = []
        for p in self.principles:
            ts = self.extract_timestamp(p)
            if ts:
                all_entries.append((ts, str(p)))
        
        for e in self.experiences:
            ts = self.extract_timestamp(e)
            if ts:
                all_entries.append((ts, str(e)))
        
        all_entries.sort(key=lambda x: x[0])
        
        for ts, text in all_entries:
            date = ts.strftime("%Y-%m-%d")
            themes = self.extract_key_phrases(text)
            for theme in themes:
                theme_timeline[theme][date] += 1
        
        growing_themes = {}
        for theme, dates in theme_timeline.items():
            sorted_dates = sorted(dates.items())
            if len(sorted_dates) >= 2:
                total = sum(d[1] for d in sorted_dates)
                
                growing_themes[theme] = {
                    "first_appearance": sorted_dates[0][0],
                    "total_mentions": total,
                    "growth_pattern": [(d[0], d[1]) for d in sorted_dates],
                }
        
        return dict(sorted(growing_themes.items(), key=lambda x: -x[1]["total_mentions"]))
    
    def get_summary(self) -> dict:
        """Get a summary of all knowledge."""
        return {
            "principles_count": len(self.principles),
            "experiences_count": len(self.experiences),
            "challenges_count": len(self.challenges),
            "core_knowledge_count": len(self.core_knowledge),
            "total_items": len(self.principles) + len(self.experiences) + len(self.challenges) + len(self.core_knowledge),
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze knowledge evolution')
    parser.add_argument('actor', help='Actor name (e.g., steve)')
    args = parser.parse_args()
    
    print("Loading knowledge bases...")
    analyzer = KnowledgeAnalyzer(args.actor)
    
    summary = analyzer.get_summary()
    print(f"Loaded:")
    print(f"  - {summary['principles_count']} principles")
    print(f"  - {summary['experiences_count']} experiences")
    print(f"  - {summary['challenges_count']} challenges")
    print(f"  - {summary['core_knowledge_count']} core knowledge items")


if __name__ == "__main__":
    main()
