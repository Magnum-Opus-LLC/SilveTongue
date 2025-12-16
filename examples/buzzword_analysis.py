#!/usr/bin/env python3
"""
Paper & Topic Buzzword Analysis

Analyzes the actual topics/papers studied to understand:
- Why knowledge scope expanded rapidly then plateaued
- What buzzwords/terms concentrate in the research
- Topic generation patterns and feedback loops
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Set


class TopicBuzzwordAnalyzer:
    """Analyze topics and papers for buzzword concentration."""
    
    def __init__(self, actor_path: Path, data_path: Path):
        self.actor_path = actor_path
        self.data_path = data_path
        
        # Load learning book
        self.learning_book = self._load_json(actor_path / "learning_book.json")
        
        # Load cognitive loop files
        self.cognitive_loops = self._load_cognitive_loops()
        
        # Parse all topics
        self.topics = self._extract_all_topics()
        
    def _load_json(self, path: Path) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_cognitive_loops(self) -> List[dict]:
        """Load all cognitive loop files."""
        loops_dir = self.data_path / "cognitive_loops"
        loops = []
        
        if loops_dir.exists():
            for f in loops_dir.glob("loop_*.json"):
                try:
                    with open(f, 'r') as file:
                        loops.append(json.load(file))
                except:
                    pass
        
        return loops
    
    def _extract_all_topics(self) -> List[dict]:
        """Extract all topics from learning history."""
        topics = []
        history = self.learning_book.get("learning_history", [])
        
        global_iter = 0
        for entry in history:
            if entry.get("event_type") == "cognitive_loop_update":
                global_iter += 1
                desc = entry.get("description", "")
                topic_match = re.search(r"on:\s*(.+)$", desc)
                topic = topic_match.group(1).strip() if topic_match else "unknown"
                
                topics.append({
                    "iter": global_iter,
                    "topic": topic,
                    "insights": entry.get("insights_gained", []),
                    "mindset_changes": entry.get("mindset_changes", []),
                })
        
        return topics
    
    def extract_buzzwords(self, text: str) -> Counter:
        """Extract buzzwords from text."""
        text_lower = text.lower()
        
        # Comprehensive buzzword list
        buzzwords = [
            # Architecture
            'transformer', 'attention', 'cnn', 'rnn', 'gnn', 'lstm', 'bert', 'gpt',
            'architecture', 'neural', 'network', 'model', 'layer', 'module',
            'encoder', 'decoder', 'embedding', 'backbone', 'head',
            
            # Efficiency
            'quantization', 'pruning', 'distillation', 'compression', 'sparse', 'sparsity',
            'efficient', 'efficiency', 'latency', 'throughput', 'inference',
            'low-power', 'edge', 'mobile', 'embedded', 'accelerator',
            
            # Training
            'training', 'pre-training', 'fine-tuning', 'transfer', 'curriculum',
            'optimizer', 'gradient', 'convergence', 'learning rate', 'batch',
            'regularization', 'dropout', 'normalization',
            
            # Learning paradigms
            'continual', 'lifelong', 'incremental', 'online', 'meta-learning',
            'forgetting', 'replay', 'buffer', 'stability', 'plasticity',
            'drift', 'distribution shift', 'covariate',
            
            # Robustness
            'robust', 'robustness', 'adversarial', 'attack', 'perturbation',
            'reliable', 'reliability', 'fault', 'degradation', 'failure',
            
            # Ethics
            'bias', 'fairness', 'ethical', 'misinformation', 'alignment',
            'safety', 'interpretable', 'interpretability', 'explainable',
            
            # Multimodal
            'multimodal', 'cross-modal', 'vision-language', 'audio', 'fusion',
            'alignment', 'unified', 'joint',
            
            # Graph
            'graph', 'node', 'edge', 'message passing', 'aggregation',
            
            # Math/Theory
            'tensor', 'contraction', 'decomposition', 'manifold',
            'scaling', 'trade-off', 'optimal', 'threshold',
            'bound', 'theoretical', 'empirical',
            
            # Hardware
            'gpu', 'tpu', 'npu', 'hardware', 'accelerator', 'asic', 'fpga',
            '5g', '6g', 'iot', 'wireless', 'network',
            
            # Metrics
            'benchmark', 'metric', 'evaluation', 'accuracy', 'performance',
            'calibration', 'uncertainty', 'confidence',
            
            # Specific techniques
            'attention mechanism', 'self-attention', 'cross-attention',
            'mixture of experts', 'moe', 'knowledge distillation',
            'neural architecture search', 'nas', 'automl',
            'replay buffer', 'experience replay',
            
            # Domains
            'nlp', 'cv', 'computer vision', 'natural language',
            'recommendation', 'autonomous', 'robotics',
            
            # General ML
            'deep learning', 'machine learning', 'ai', 'model',
            'dataset', 'data', 'sample', 'batch',
        ]
        
        counts = Counter()
        for bw in buzzwords:
            count = text_lower.count(bw)
            if count > 0:
                counts[bw] = count
        
        return counts
    
    def analyze_topic_buzzwords(self) -> dict:
        """Analyze buzzword distribution across topics."""
        topic_buzzwords = []
        all_buzzwords = Counter()
        
        for t in self.topics:
            # Combine topic and insights
            full_text = t["topic"] + " " + " ".join(t["insights"])
            buzzwords = self.extract_buzzwords(full_text)
            all_buzzwords.update(buzzwords)
            
            topic_buzzwords.append({
                "iter": t["iter"],
                "topic": t["topic"][:80],
                "buzzwords": buzzwords,
                "top_buzzwords": buzzwords.most_common(5),
            })
        
        return {
            "by_topic": topic_buzzwords,
            "overall": all_buzzwords,
        }
    
    def analyze_buzzword_evolution(self, window: int = 100) -> List[dict]:
        """Track how buzzwords evolved over iteration windows."""
        evolution = []
        
        for i in range(0, len(self.topics), window):
            window_topics = self.topics[i:i+window]
            window_text = " ".join(t["topic"] + " " + " ".join(t["insights"]) for t in window_topics)
            buzzwords = self.extract_buzzwords(window_text)
            
            evolution.append({
                "iter_start": window_topics[0]["iter"] if window_topics else 0,
                "iter_end": window_topics[-1]["iter"] if window_topics else 0,
                "top_buzzwords": buzzwords.most_common(15),
                "unique_buzzwords": len(buzzwords),
                "total_mentions": sum(buzzwords.values()),
            })
        
        return evolution
    
    def analyze_topic_similarity(self) -> dict:
        """Analyze how similar consecutive topics are."""
        similarities = []
        
        for i in range(1, len(self.topics)):
            prev_topic = self.topics[i-1]["topic"].lower()
            curr_topic = self.topics[i]["topic"].lower()
            
            # Simple word overlap similarity
            prev_words = set(prev_topic.split())
            curr_words = set(curr_topic.split())
            
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words)
                union = len(prev_words | curr_words)
                jaccard = overlap / union if union > 0 else 0
            else:
                jaccard = 0
            
            similarities.append({
                "iter": self.topics[i]["iter"],
                "similarity": jaccard,
                "prev_topic": prev_topic[:50],
                "curr_topic": curr_topic[:50],
            })
        
        return {
            "similarities": similarities,
            "avg_similarity": sum(s["similarity"] for s in similarities) / len(similarities) if similarities else 0,
        }
    
    def analyze_topic_patterns(self) -> dict:
        """Identify patterns in topic generation."""
        # Extract common phrases
        phrase_counts = Counter()
        word_counts = Counter()
        
        for t in self.topics:
            topic = t["topic"].lower()
            
            # Count common starting phrases
            for phrase_len in [2, 3, 4]:
                words = topic.split()[:phrase_len]
                if len(words) == phrase_len:
                    phrase = " ".join(words)
                    phrase_counts[phrase] += 1
            
            # Count individual significant words
            for word in topic.split():
                if len(word) > 4:  # Skip short words
                    word_counts[word] += 1
        
        return {
            "common_phrases": phrase_counts.most_common(30),
            "common_words": word_counts.most_common(40),
        }
    
    def analyze_papers_from_loops(self) -> dict:
        """Analyze actual papers discovered in cognitive loops."""
        all_papers = []
        paper_titles = []
        sources = Counter()
        
        for loop in self.cognitive_loops:
            state = loop.get("state", {})
            papers = state.get("discovered_papers", [])
            topic = state.get("topic", "unknown")
            
            for paper in papers:
                title = paper.get("title", "")
                source = paper.get("source", "unknown")
                citations = paper.get("citation_count", 0)
                
                all_papers.append({
                    "title": title,
                    "source": source,
                    "citations": citations,
                    "topic": topic[:50],
                })
                paper_titles.append(title.lower())
                sources[source] += 1
        
        # Analyze paper title buzzwords
        paper_text = " ".join(paper_titles)
        paper_buzzwords = self.extract_buzzwords(paper_text)
        
        return {
            "total_papers": len(all_papers),
            "unique_sources": dict(sources.most_common(20)),
            "paper_buzzwords": paper_buzzwords.most_common(30),
            "sample_papers": all_papers[:20],
        }
    
    def find_feedback_loops(self) -> dict:
        """Identify potential feedback loops in topic generation."""
        # Track topic → mindset → new topic patterns
        feedback_patterns = []
        
        for i in range(len(self.topics) - 1):
            curr = self.topics[i]
            next_t = self.topics[i + 1]
            
            # Check if mindset changes influenced next topic
            for change in curr.get("mindset_changes", []):
                change_lower = change.lower()
                next_topic_lower = next_t["topic"].lower()
                
                # Look for keyword overlap
                change_words = set(w for w in change_lower.split() if len(w) > 5)
                topic_words = set(w for w in next_topic_lower.split() if len(w) > 5)
                
                overlap = change_words & topic_words
                if len(overlap) >= 2:
                    feedback_patterns.append({
                        "iter": curr["iter"],
                        "mindset_change": change[:100],
                        "next_topic": next_t["topic"][:80],
                        "overlap_words": list(overlap)[:5],
                    })
        
        return {
            "patterns": feedback_patterns[:30],
            "total_feedback_loops": len(feedback_patterns),
        }


def generate_buzzword_report(analyzer: TopicBuzzwordAnalyzer):
    """Generate comprehensive buzzword analysis report."""
    
    print("\n" + "="*100)
    print("🔍 PAPER & TOPIC BUZZWORD CONCENTRATION ANALYSIS")
    print("="*100)
    print(f"\n  Total Topics Analyzed: {len(analyzer.topics)}")
    print(f"  Cognitive Loop Files: {len(analyzer.cognitive_loops)}")
    
    # 1. Overall Buzzword Distribution
    print("\n\n" + "─"*100)
    print("📊 TOP BUZZWORDS ACROSS ALL TOPICS")
    print("─"*100)
    
    buzzword_analysis = analyzer.analyze_topic_buzzwords()
    overall = buzzword_analysis["overall"]
    
    print("\n  Rank │ Buzzword                 │ Count │ Frequency Bar")
    print("  " + "-"*75)
    
    max_count = overall.most_common(1)[0][1] if overall else 1
    for rank, (word, count) in enumerate(overall.most_common(35), 1):
        bar_len = int(40 * count / max_count)
        bar = "█" * bar_len
        print(f"  {rank:>4} │ {word:<24} │ {count:>5} │ {bar}")
    
    # 2. Buzzword Evolution Over Time
    print("\n\n" + "─"*100)
    print("📈 BUZZWORD EVOLUTION BY ITERATION WINDOW")
    print("─"*100)
    
    evolution = analyzer.analyze_buzzword_evolution(window=100)
    
    for e in evolution:
        print(f"\n  Iterations {e['iter_start']:>4} - {e['iter_end']:<4} ({e['unique_buzzwords']} unique, {e['total_mentions']} total):")
        top_5 = e["top_buzzwords"][:10]
        buzzwords_str = ", ".join(f"{w}({c})" for w, c in top_5)
        print(f"    Top: {buzzwords_str}")
    
    # 3. Topic Similarity Analysis
    print("\n\n" + "─"*100)
    print("🔗 TOPIC SIMILARITY ANALYSIS (Why scope plateaus?)")
    print("─"*100)
    
    similarity = analyzer.analyze_topic_similarity()
    
    print(f"\n  Average topic-to-topic similarity: {similarity['avg_similarity']:.3f}")
    
    # Find highly similar consecutive topics
    high_sim = [s for s in similarity["similarities"] if s["similarity"] > 0.3]
    print(f"  Topics with >30% similarity to previous: {len(high_sim)} ({len(high_sim)/len(analyzer.topics)*100:.1f}%)")
    
    # Show distribution
    sim_buckets = defaultdict(int)
    for s in similarity["similarities"]:
        bucket = int(s["similarity"] * 10) / 10
        sim_buckets[bucket] += 1
    
    print("\n  Similarity Distribution:")
    for bucket in sorted(sim_buckets.keys()):
        bar = "█" * (sim_buckets[bucket] // 5)
        print(f"    {bucket:.1f}-{bucket+0.1:.1f}: {bar} ({sim_buckets[bucket]})")
    
    # 4. Topic Patterns
    print("\n\n" + "─"*100)
    print("🔄 TOPIC GENERATION PATTERNS")
    print("─"*100)
    
    patterns = analyzer.analyze_topic_patterns()
    
    print("\n  Most Common Topic Starting Phrases:")
    print("  " + "-"*60)
    for phrase, count in patterns["common_phrases"][:20]:
        if count > 5:
            print(f"    \"{phrase}...\" : {count} times")
    
    print("\n  Most Frequent Words in Topics:")
    print("  " + "-"*60)
    word_line = ""
    for word, count in patterns["common_words"][:30]:
        if count > 10:
            word_line += f"{word}({count}) "
            if len(word_line) > 80:
                print(f"    {word_line}")
                word_line = ""
    if word_line:
        print(f"    {word_line}")
    
    # 5. Paper Analysis (from cognitive loops)
    print("\n\n" + "─"*100)
    print("📚 ACTUAL PAPERS ANALYZED")
    print("─"*100)
    
    papers = analyzer.analyze_papers_from_loops()
    
    print(f"\n  Total Papers Discovered: {papers['total_papers']}")
    print("\n  Paper Sources:")
    for source, count in list(papers["unique_sources"].items())[:10]:
        print(f"    {source}: {count}")
    
    print("\n  Buzzwords in Paper Titles:")
    print("  " + "-"*60)
    for word, count in papers["paper_buzzwords"][:20]:
        print(f"    {word}: {count}")
    
    print("\n  Sample Papers:")
    for p in papers["sample_papers"][:10]:
        print(f"    • {p['title'][:70]}...")
        print(f"      Citations: {p['citations']}, Source: {p['source']}")
    
    # 6. Feedback Loop Analysis
    print("\n\n" + "─"*100)
    print("🔁 FEEDBACK LOOP DETECTION")
    print("─"*100)
    
    feedback = analyzer.find_feedback_loops()
    
    print(f"\n  Detected {feedback['total_feedback_loops']} potential feedback loops")
    print("  (where mindset changes influenced next topic selection)")
    
    print("\n  Examples:")
    for f in feedback["patterns"][:8]:
        print(f"\n  Iter {f['iter']}:")
        print(f"    Mindset: \"{f['mindset_change'][:70]}...\"")
        print(f"    → Next Topic: \"{f['next_topic'][:60]}...\"")
        print(f"    Overlap: {f['overlap_words']}")
    
    # 7. Key Insights
    print("\n\n" + "─"*100)
    print("💡 WHY KNOWLEDGE SCOPE PLATEAUS - KEY INSIGHTS")
    print("─"*100)
    
    # Calculate concentration metrics
    top_10_buzzwords = sum(c for _, c in overall.most_common(10))
    total_buzzwords = sum(overall.values())
    concentration = top_10_buzzwords / total_buzzwords if total_buzzwords > 0 else 0
    
    phrase_repetition = sum(c for _, c in patterns["common_phrases"] if c > 10)
    
    print(f"""
  1. BUZZWORD CONCENTRATION:
     • Top 10 buzzwords account for {concentration*100:.1f}% of all mentions
     • Most frequent: {', '.join(w for w, _ in overall.most_common(5))}
     • This suggests topics cluster around a narrow set of themes
  
  2. TOPIC SIMILARITY:
     • Average consecutive topic similarity: {similarity['avg_similarity']:.1%}
     • {len(high_sim)} topics ({len(high_sim)/len(analyzer.topics)*100:.1f}%) are very similar to their predecessor
     • Topics tend to refine the same concepts rather than explore new ones
  
  3. PHRASE REPETITION:
     • Common phrases like "optimal", "trade-off", "long-term" repeat {phrase_repetition}+ times
     • Suggests the system is asking variations of the same questions
  
  4. FEEDBACK LOOP EFFECT:
     • {feedback['total_feedback_loops']} detected feedback loops
     • Mindset changes → similar next topics → reinforces same themes
     • This creates a "gravity well" around established topics
  
  5. ROOT CAUSE HYPOTHESIS:
     ─────────────────────────
     The "T-shaped" learning pattern likely emerges because:
     
     a) TOPIC GENERATION uses terms from previous mindset changes,
        creating self-reinforcing loops around established themes
     
     b) BUZZWORD CLUSTERING - certain terms (efficiency, edge, robust,
        trade-off) appear so frequently that new topics inevitably
        touch on existing themes
     
     c) SEMANTIC OVERLAP - "Optimal X", "Trade-off between Y and Z",
        "Long-term effects of W" are variations, not new directions
     
     d) RESEARCH FIELD STRUCTURE - AI/ML research itself clusters
        around hot topics (transformers, edge AI, continual learning)
    """)
    
    print("="*100)
    print("Analysis Complete!")
    print("="*100)


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent
    actor_path = base_path / "array" / "actors" / "steve"
    data_path = base_path / "array" / "data" / "steve"
    
    print("Loading topic and paper data...")
    analyzer = TopicBuzzwordAnalyzer(actor_path, data_path)
    
    generate_buzzword_report(analyzer)


if __name__ == "__main__":
    main()
