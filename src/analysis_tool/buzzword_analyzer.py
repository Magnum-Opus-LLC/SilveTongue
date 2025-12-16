#!/usr/bin/env python3
"""
Topic Buzzword Analyzer
=======================
Analyzes topics and papers for buzzword concentration and patterns.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Set


class TopicBuzzwordAnalyzer:
    """Analyze topics and papers for buzzword concentration."""
    
    # Comprehensive buzzword list
    BUZZWORDS = [
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
    
    def __init__(self, actor_name: str, base_dir: str = "array"):
        self.actor_name = actor_name
        self.base_dir = Path(base_dir)
        self.actor_path = self.base_dir / "actors" / actor_name
        self.data_path = self.base_dir / "data" / actor_name
        
        # Load learning book
        self.learning_book = self._load_json(self.actor_path / "learning_book.json")
        
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
        counts = Counter()
        
        for bw in self.BUZZWORDS:
            count = text_lower.count(bw)
            if count > 0:
                counts[bw] = count
        
        return counts
    
    def analyze_topic_buzzwords(self) -> dict:
        """Analyze buzzword distribution across topics."""
        topic_buzzwords = []
        all_buzzwords = Counter()
        
        for t in self.topics:
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
        phrase_counts = Counter()
        word_counts = Counter()
        
        for t in self.topics:
            topic = t["topic"].lower()
            
            for phrase_len in [2, 3, 4]:
                words = topic.split()[:phrase_len]
                if len(words) == phrase_len:
                    phrase = " ".join(words)
                    phrase_counts[phrase] += 1
            
            for word in topic.split():
                if len(word) > 4:
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
        feedback_patterns = []
        
        for i in range(len(self.topics) - 1):
            curr = self.topics[i]
            next_t = self.topics[i + 1]
            
            for change in curr.get("mindset_changes", []):
                change_lower = change.lower()
                next_topic_lower = next_t["topic"].lower()
                
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
    
    def generate_report(self) -> str:
        """Generate comprehensive buzzword analysis report."""
        lines = []
        
        lines.append("=" * 100)
        lines.append("PAPER & TOPIC BUZZWORD CONCENTRATION ANALYSIS")
        lines.append("=" * 100)
        lines.append(f"\n  Total Topics Analyzed: {len(self.topics)}")
        lines.append(f"  Cognitive Loop Files: {len(self.cognitive_loops)}")
        
        # Overall Buzzword Distribution
        buzzword_analysis = self.analyze_topic_buzzwords()
        overall = buzzword_analysis["overall"]
        
        lines.append("\n\n" + "-" * 100)
        lines.append("TOP BUZZWORDS ACROSS ALL TOPICS")
        lines.append("-" * 100)
        
        max_count = overall.most_common(1)[0][1] if overall else 1
        for rank, (word, count) in enumerate(overall.most_common(35), 1):
            bar_len = int(40 * count / max_count)
            bar = "#" * bar_len
            lines.append(f"  {rank:>4} | {word:<24} | {count:>5} | {bar}")
        
        # Topic Similarity
        similarity = self.analyze_topic_similarity()
        lines.append("\n\n" + "-" * 100)
        lines.append("TOPIC SIMILARITY ANALYSIS")
        lines.append("-" * 100)
        lines.append(f"\n  Average topic-to-topic similarity: {similarity['avg_similarity']:.3f}")
        
        high_sim = [s for s in similarity["similarities"] if s["similarity"] > 0.3]
        lines.append(f"  Topics with >30% similarity to previous: {len(high_sim)} ({len(high_sim)/len(self.topics)*100:.1f}%)")
        
        # Feedback Loops
        feedback = self.find_feedback_loops()
        lines.append("\n\n" + "-" * 100)
        lines.append("FEEDBACK LOOP DETECTION")
        lines.append("-" * 100)
        lines.append(f"\n  Detected {feedback['total_feedback_loops']} potential feedback loops")
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze buzzwords in learning data')
    parser.add_argument('actor', help='Actor name (e.g., steve)')
    args = parser.parse_args()
    
    print("Loading topic and paper data...")
    analyzer = TopicBuzzwordAnalyzer(args.actor)
    
    report = analyzer.generate_report()
    print(report)


if __name__ == "__main__":
    main()
