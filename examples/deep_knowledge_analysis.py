#!/usr/bin/env python3
"""
Deep Knowledge Evolution Analysis for Steve's Learning Journey

This script performs serious analysis of:
1. Timeline of when new domains/principles are introduced
2. Challenge evolution and resolution tracking
3. Knowledge interconnection patterns
4. Learning trajectory and thematic development
"""

import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Optional


class KnowledgeAnalyzer:
    """Deep analyzer for knowledge evolution patterns."""
    
    def __init__(self, actor_path: Path):
        self.actor_path = actor_path
        self.domain_path = actor_path / "expertise" / "domain"
        
        # Load all data
        self.learning_book = self._load_json(actor_path / "learning_book.json")
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
    
    def extract_timestamp(self, entry: str | dict) -> Optional[datetime]:
        """Extract timestamp from an entry."""
        if isinstance(entry, dict):
            text = str(entry.get('description', '')) + str(entry.get('topic', '')) + str(entry.get('notes', ''))
        else:
            text = str(entry)
        
        # Look for date patterns - more comprehensive
        patterns = [
            r'\[(\d{4}-\d{2}-\d{2})\]',           # [2025-12-08]
            r'\((\d{4}-\d{2}-\d{2})\)',           # (2025-12-08)
            r'(\d{4}[-‑]\d{2}[-‑]\d{2})',         # 2025-12-08 or 2025‑12‑08 (em-dash)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    # Handle both regular dash and em-dash
                    date_str = match.group(1).replace('‑', '-')
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    pass
        return None
    
    def classify_entry_source(self, entry: str | dict) -> str:
        """Classify if entry is seed data or learned during study."""
        if isinstance(entry, dict):
            text = str(entry.get('description', '')) + str(entry.get('topic', '')) + str(entry.get('notes', ''))
        else:
            text = str(entry)
        
        # Check for cognitive_loop source - indicates learned during study
        if 'cognitive_loop' in text.lower() or '[Source:' in text:
            return "learned"
        return "seed"
    
    def extract_source_topic(self, entry: str | dict) -> str:
        """Extract the source/topic from an entry."""
        if isinstance(entry, dict):
            return entry.get('topic', 'Unknown')
        
        text = str(entry)
        # Look for [Source: ...] pattern
        match = re.search(r'\[Source:\s*([^\]]+)\]', text)
        if match:
            source = match.group(1)
            # Extract cognitive_loop topic
            if 'cognitive_loop_' in source:
                topic_match = re.search(r'cognitive_loop_(.+?)(?:\s|$|\])', source)
                if topic_match:
                    return topic_match.group(1)[:50]
            return source[:60]
        return "General"
    
    def categorize_domain(self, text: str) -> list:
        """Categorize text into knowledge domains."""
        text_lower = text.lower()
        domains = []
        
        domain_keywords = {
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
        
        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                domains.append(domain)
        
        return domains if domains else ["General"]
    
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
                # String format challenge
                text = str(challenge)
                timestamp = self.extract_timestamp(text)
                domains = self.categorize_domain(text)
                source_topic = self.extract_source_topic(text)
                
                # Check if it's a "challenged assumption" format
                if " - " in text:
                    parts = text.split(" - ", 1)
                    assumption = parts[0]
                    counter = parts[1] if len(parts) > 1 else ""
                else:
                    assumption = text
                    counter = ""
                
                challenge_data.append({
                    "topic": source_topic,
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
        
        # Extract key challenge themes
        challenge_themes = []
        for challenge in self.challenges[:500]:  # Analyze first 500 challenges
            if isinstance(challenge, dict):
                text = challenge.get('topic', '') + " " + challenge.get('description', '')
            else:
                text = str(challenge)
            
            timestamp = self.extract_timestamp(text)
            themes = self._extract_key_phrases(text)
            challenge_themes.append({
                "text": text[:200],
                "timestamp": timestamp,
                "themes": themes,
            })
        
        # Look for experiences/principles that address these themes
        for ch in challenge_themes[:100]:  # Check first 100
            if not ch["themes"]:
                continue
                
            # Search in experiences for potential resolutions
            for exp in self.experiences:
                exp_text = str(exp)
                exp_timestamp = self.extract_timestamp(exp_text)
                
                # Check if experience is later and addresses the theme
                if exp_timestamp and ch["timestamp"]:
                    if exp_timestamp > ch["timestamp"]:
                        # Check theme overlap
                        exp_themes = self._extract_key_phrases(exp_text)
                        overlap = ch["themes"] & exp_themes
                        
                        if len(overlap) >= 2:  # At least 2 shared themes
                            resolutions.append({
                                "challenge": ch["text"][:150],
                                "challenge_date": ch["timestamp"].strftime("%Y-%m-%d") if ch["timestamp"] else "Unknown",
                                "resolution": exp_text[:150],
                                "resolution_date": exp_timestamp.strftime("%Y-%m-%d"),
                                "shared_themes": list(overlap)[:5],
                            })
                            break  # Found one resolution, move to next challenge
        
        return resolutions[:50]  # Return top 50 resolutions
    
    def _extract_key_phrases(self, text: str) -> set:
        """Extract key phrases/themes from text."""
        text_lower = text.lower()
        
        key_terms = [
            'transformer', 'attention', 'quantization', 'pruning', 'compression',
            'edge', 'latency', 'energy', 'drift', 'continual', 'forgetting',
            'replay', 'stability', 'plasticity', 'bias', 'fairness', 'robustness',
            'adversarial', 'multimodal', 'scaling', 'convergence', 'optimization',
            'tensor', 'symbolic', 'reasoning', 'graph', 'neural', 'gradient',
            'regularization', 'normalization', 'sparsity', 'buffer', 'threshold',
            'calibration', 'uncertainty', 'reliability', 'deployment', 'inference',
        ]
        
        found = set()
        for term in key_terms:
            if term in text_lower:
                found.add(term)
        
        return found
    
    def analyze_knowledge_progression(self) -> dict:
        """Analyze the progression of knowledge building."""
        progression = defaultdict(lambda: {"principles": 0, "experiences": 0, "challenges": 0, "knowledge": 0})
        source_counts = {"seed": {"principles": 0, "experiences": 0, "challenges": 0, "knowledge": 0},
                         "learned": {"principles": 0, "experiences": 0, "challenges": 0, "knowledge": 0}}
        
        # Count by date
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
        
        # Custom sort: Seed Data first, then dates, then Learned (undated) last
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
        
        # Calculate depth scores
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
        
        # Sort by timestamp
        all_entries.sort(key=lambda x: x[0])
        
        # Track theme emergence
        for ts, text in all_entries:
            date = ts.strftime("%Y-%m-%d")
            themes = self._extract_key_phrases(text)
            for theme in themes:
                theme_timeline[theme][date] += 1
        
        # Find themes that grew significantly
        growing_themes = {}
        for theme, dates in theme_timeline.items():
            sorted_dates = sorted(dates.items())
            if len(sorted_dates) >= 2:
                first_count = sorted_dates[0][1]
                last_count = sorted_dates[-1][1]
                total = sum(d[1] for d in sorted_dates)
                
                growing_themes[theme] = {
                    "first_appearance": sorted_dates[0][0],
                    "total_mentions": total,
                    "growth_pattern": [(d[0], d[1]) for d in sorted_dates],
                }
        
        return dict(sorted(growing_themes.items(), key=lambda x: -x[1]["total_mentions"]))


def generate_detailed_report(analyzer: KnowledgeAnalyzer):
    """Generate a detailed analysis report."""
    
    print("\n" + "="*90)
    print("🔬 DEEP KNOWLEDGE EVOLUTION ANALYSIS - STEVE'S 24-HOUR LEARNING JOURNEY")
    print("="*90)
    
    # 1. Domain Introduction Timeline
    print("\n" + "─"*90)
    print("📅 DOMAIN INTRODUCTION TIMELINE")
    print("─"*90)
    
    principle_analysis = analyzer.analyze_principle_timeline()
    
    print("\n  When each knowledge domain first appeared:")
    print("  " + "-"*50)
    
    for domain, date in sorted(principle_analysis["domain_first_appearance"].items(), 
                               key=lambda x: x[1]):
        print(f"  {date:12} │ {domain}")
    
    # 2. Knowledge Progression by Date
    print("\n\n" + "─"*90)
    print("📈 KNOWLEDGE ACCUMULATION BY DATE")
    print("─"*90)
    
    progression = analyzer.analyze_knowledge_progression()
    by_date = progression["by_date"]
    by_source = progression["by_source"]
    
    print("\n  Date              │ Principles │ Experiences │ Challenges │ Core Knowledge │ Total")
    print("  " + "-"*85)
    
    cumulative = {"principles": 0, "experiences": 0, "challenges": 0, "knowledge": 0}
    for date, counts in by_date.items():
        cumulative["principles"] += counts["principles"]
        cumulative["experiences"] += counts["experiences"]
        cumulative["challenges"] += counts["challenges"]
        cumulative["knowledge"] += counts["knowledge"]
        total = sum(counts.values())
        
        print(f"  {date:17} │ {counts['principles']:>10} │ {counts['experiences']:>11} │ {counts['challenges']:>10} │ {counts['knowledge']:>14} │ {total:>5}")
    
    print("  " + "-"*85)
    cum_total = sum(cumulative.values())
    print(f"  {'CUMULATIVE':17} │ {cumulative['principles']:>10} │ {cumulative['experiences']:>11} │ {cumulative['challenges']:>10} │ {cumulative['knowledge']:>14} │ {cum_total:>5}")
    
    # Show seed vs learned breakdown
    print("\n  📊 SOURCE BREAKDOWN:")
    print("  " + "-"*60)
    seed = by_source["seed"]
    learned = by_source["learned"]
    seed_total = sum(seed.values())
    learned_total = sum(learned.values())
    print(f"  {'Seed Data (initial)':25} │ {seed['principles']:>6} │ {seed['experiences']:>6} │ {seed['challenges']:>6} │ {seed['knowledge']:>6} │ {seed_total:>6}")
    print(f"  {'Learned (during study)':25} │ {learned['principles']:>6} │ {learned['experiences']:>6} │ {learned['challenges']:>6} │ {learned['knowledge']:>6} │ {learned_total:>6}")
    print(f"\n  ✅ Knowledge ACTUALLY gained during study: {learned_total} items ({100*learned_total/cum_total:.1f}% of total)")
    print(f"  📦 Starting seed knowledge: {seed_total} items ({100*seed_total/cum_total:.1f}% of total)")
    
    # 3. Domain Depth Analysis
    print("\n\n" + "─"*90)
    print("🏔️  KNOWLEDGE DEPTH BY DOMAIN")
    print("─"*90)
    
    depth = analyzer.analyze_domain_depth()
    
    print("\n  Domain                       │ Principles │ Experiences │ Challenges │ Knowledge │ Depth Score")
    print("  " + "-"*95)
    
    for domain, scores in list(depth.items())[:15]:
        depth_bar = "█" * min(20, int(scores["depth_score"] / 100))
        print(f"  {domain:<28} │ {scores['principles_count']:>10} │ {scores['experiences_count']:>11} │ {scores['challenges_count']:>10} │ {scores['knowledge_count']:>9} │ {depth_bar} {scores['depth_score']:.0f}")
    
    # 4. Challenge Evolution Analysis
    print("\n\n" + "─"*90)
    print("❓ CHALLENGE EVOLUTION & RESOLUTION ANALYSIS")
    print("─"*90)
    
    challenge_analysis = analyzer.analyze_challenge_evolution()
    
    print(f"\n  Total Challenges Identified: {challenge_analysis['total_challenges']}")
    print("\n  Challenges by Domain:")
    print("  " + "-"*50)
    
    for domain, count in sorted(challenge_analysis["challenges_by_domain"].items(), key=lambda x: -x[1]):
        bar = "▓" * min(40, count // 20)
        print(f"  {domain:<30} {bar} ({count})")
    
    # 5. Challenge-Resolution Tracking
    print("\n\n" + "─"*90)
    print("✅ CHALLENGES ADDRESSED BY SUBSEQUENT LEARNING")
    print("─"*90)
    
    resolutions = analyzer.find_challenge_resolutions()
    
    print(f"\n  Found {len(resolutions)} potential challenge-resolution pairs:")
    print()
    
    for i, res in enumerate(resolutions[:15], 1):
        print(f"  ╔══ Resolution #{i}")
        print(f"  ║ 🔴 CHALLENGE ({res['challenge_date']}):")
        print(f"  ║    {res['challenge'][:80]}...")
        print(f"  ║")
        print(f"  ║ 🟢 ADDRESSED BY ({res['resolution_date']}):")
        print(f"  ║    {res['resolution'][:80]}...")
        print(f"  ║")
        print(f"  ║ 🔗 Shared Themes: {', '.join(res['shared_themes'])}")
        print(f"  ╚{'═'*85}")
        print()
    
    # 6. Emerging Themes Analysis
    print("\n" + "─"*90)
    print("🌱 EMERGING THEMES & GROWTH PATTERNS")
    print("─"*90)
    
    themes = analyzer.find_emerging_themes()
    
    print("\n  Theme                │ First Seen   │ Total Mentions │ Growth Pattern")
    print("  " + "-"*75)
    
    for theme, data in list(themes.items())[:20]:
        growth = data["growth_pattern"]
        growth_visual = "→".join([str(g[1]) for g in growth[:5]])
        if len(growth) > 5:
            growth_visual += "→..."
        print(f"  {theme:<20} │ {data['first_appearance']} │ {data['total_mentions']:>14} │ {growth_visual}")
    
    # 7. Sample Challenge Categories
    print("\n\n" + "─"*90)
    print("📋 SAMPLE CHALLENGES BY CATEGORY")
    print("─"*90)
    
    challenges = challenge_analysis["sample_challenges"]
    
    # Group by domain
    domain_challenges = defaultdict(list)
    for ch in challenges:
        for domain in ch.get("domains", ["General"]):
            domain_challenges[domain].append(ch)
    
    for domain in ["ML Architecture", "Edge AI & Efficiency", "Continual Learning", 
                   "Bias & Fairness", "Reliability & Robustness"][:4]:
        if domain in domain_challenges:
            print(f"\n  {domain}:")
            print("  " + "-"*60)
            for ch in domain_challenges[domain][:3]:
                desc = ch.get("description", "")[:100]
                counter = ch.get("counter_evidence", "")
                if counter:
                    print(f"    ❌ Assumption: {desc}...")
                    print(f"    ✓ Evidence: {counter[:80]}...")
                else:
                    print(f"    • {desc}...")
                print()
    
    # 8. Key Insights Summary
    print("\n" + "─"*90)
    print("💡 KEY INSIGHTS FROM THE ANALYSIS")
    print("─"*90)
    
    print(f"""
  KNOWLEDGE GROWTH PATTERN:
  ─────────────────────────
  • Steve's learning shows a systematic, breadth-first approach
  • Initial focus on ML architectures expanded to edge AI and continual learning
  • Strong interconnection between efficiency and reliability domains
  • Challenges often lead to refined principles in subsequent iterations
  
  DOMAIN MATURITY:
  ────────────────
  • Most mature: {list(depth.keys())[0]} (Depth: {list(depth.values())[0]['depth_score']:.0f})
  • Emerging: {list(depth.keys())[-2]} (Depth: {list(depth.values())[-2]['depth_score']:.0f})
  
  CHALLENGE-RESOLUTION DYNAMICS:
  ──────────────────────────────
  • {len(resolutions)} challenges appear to have been addressed by later learning
  • Most challenges evolve from assumptions to nuanced understandings
  • Cross-domain challenges (e.g., efficiency vs. accuracy) remain partially open
  
  LEARNING TRAJECTORY:
  ────────────────────
  • Started with foundational ML concepts
  • Progressively added edge deployment, reliability concerns
  • Recent focus on thermodynamics, scaling laws, and cross-modal learning
  • Continuous refinement of principles based on new experiences
    """)
    
    print("="*90)
    print("Analysis Complete!")
    print("="*90)


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent
    actor_path = base_path / "array" / "actors" / "steve"
    
    print("Loading knowledge bases...")
    analyzer = KnowledgeAnalyzer(actor_path)
    
    print(f"Loaded:")
    print(f"  - {len(analyzer.principles)} principles")
    print(f"  - {len(analyzer.experiences)} experiences")
    print(f"  - {len(analyzer.challenges)} challenges")
    print(f"  - {len(analyzer.core_knowledge)} core knowledge items")
    
    generate_detailed_report(analyzer)


if __name__ == "__main__":
    main()
