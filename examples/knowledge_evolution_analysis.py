#!/usr/bin/env python3
"""
Knowledge Evolution Analysis Tool for Steve's 24-Hour Learning Session

This script analyzes and visualizes how Steve's knowledge grew over 24 hours
of continuous "endless study" mode, examining:
- Knowledge width (topic diversity)
- Knowledge depth (principles, experiences, challenges per topic)
- Temporal evolution of learning
- Mindset changes and insights
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import re


def load_learning_book(actor_path: Path) -> dict:
    """Load the learning book JSON."""
    learning_book_path = actor_path / "learning_book.json"
    with open(learning_book_path, 'r') as f:
        return json.load(f)


def load_yaml_file(file_path: Path) -> dict | list:
    """Load a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def count_yaml_entries(file_path: Path) -> int:
    """Count entries in a YAML list file."""
    try:
        data = load_yaml_file(file_path)
        if isinstance(data, dict):
            # Get the first list value
            for key, value in data.items():
                if isinstance(value, list):
                    return len(value)
        elif isinstance(data, list):
            return len(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return 0


def analyze_learning_history(learning_book: dict) -> dict:
    """Analyze the learning history for patterns and growth."""
    history = learning_book.get("learning_history", [])
    
    # Parse timestamps and group by hour
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
        
        # Extract hour from timestamp
        try:
            if timestamp_str:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                hour_key = dt.strftime("%Y-%m-%d %H:00")
                hourly_activity[hour_key].append(entry)
                insights_by_hour[hour_key].extend(insights)
                
                # Extract topics from description
                topics = extract_topics(description)
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


def extract_topics(text: str) -> set:
    """Extract topic keywords from text."""
    # Common ML/AI topics to look for
    topic_patterns = [
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
    
    topics = set()
    text_lower = text.lower()
    for pattern in topic_patterns:
        if re.search(pattern, text_lower):
            # Normalize the topic name
            match = re.search(pattern, text_lower)
            if match:
                topics.add(match.group().replace('s', '').strip())
    
    return topics


def analyze_knowledge_domains(actor_path: Path) -> dict:
    """Analyze the knowledge domain files for depth and breadth."""
    domain_path = actor_path / "expertise" / "domain"
    
    results = {
        "principles": {"count": 0, "topics": Counter()},
        "core_knowledge": {"count": 0, "topics": Counter()},
        "experiences": {"count": 0, "topics": Counter()},
        "challenges": {"count": 0, "topics": Counter()},
    }
    
    # Analyze principles
    principles_path = domain_path / "principal" / "principles.yaml"
    if principles_path.exists():
        results["principles"]["count"] = count_yaml_entries(principles_path)
        results["principles"]["topics"] = extract_topics_from_yaml(principles_path)
    
    # Analyze core knowledge
    core_path = domain_path / "knowledge" / "core" / "core_knowledge.yaml"
    if core_path.exists():
        results["core_knowledge"]["count"] = count_yaml_entries(core_path)
        results["core_knowledge"]["topics"] = extract_topics_from_yaml(core_path)
    
    # Analyze experiences
    exp_path = domain_path / "experience" / "experiences.yaml"
    if exp_path.exists():
        results["experiences"]["count"] = count_yaml_entries(exp_path)
        results["experiences"]["topics"] = extract_topics_from_yaml(exp_path)
    
    # Analyze challenges
    chal_path = domain_path / "challenge" / "challenges.yaml"
    if chal_path.exists():
        results["challenges"]["count"] = count_yaml_entries(chal_path)
        results["challenges"]["topics"] = extract_topics_from_yaml(chal_path)
    
    return results


def extract_topics_from_yaml(file_path: Path) -> Counter:
    """Extract and count topics from a YAML file."""
    topics = Counter()
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            extracted = extract_topics(content)
            topics.update(extracted)
    except Exception as e:
        print(f"Error extracting topics from {file_path}: {e}")
    return topics


def analyze_mindset_evolution(learning_book: dict) -> dict:
    """Analyze how Steve's mindset evolved during learning."""
    history = learning_book.get("learning_history", [])
    
    mindset_changes = []
    for entry in history:
        changes = entry.get("mindset_changes", [])
        if changes:
            mindset_changes.extend(changes)
    
    # Categorize mindset changes
    categories = {
        "architectural_insights": [],
        "efficiency_principles": [],
        "theoretical_connections": [],
        "practical_applications": [],
        "cross_domain_links": [],
    }
    
    for change in mindset_changes:
        change_lower = change.lower()
        if any(k in change_lower for k in ['architecture', 'transformer', 'layer', 'attention']):
            categories["architectural_insights"].append(change)
        elif any(k in change_lower for k in ['efficient', 'energy', 'latency', 'compression']):
            categories["efficiency_principles"].append(change)
        elif any(k in change_lower for k in ['theorem', 'bound', 'convergence', 'theory']):
            categories["theoretical_connections"].append(change)
        elif any(k in change_lower for k in ['deploy', 'production', 'edge', 'real-world']):
            categories["practical_applications"].append(change)
        elif any(k in change_lower for k in ['cross', 'multimodal', 'hybrid', 'unified']):
            categories["cross_domain_links"].append(change)
    
    return {
        "total_mindset_changes": len(mindset_changes),
        "categories": {k: len(v) for k, v in categories.items()},
        "sample_changes": mindset_changes[:10] if mindset_changes else [],
    }


def generate_text_visualization(stats: dict):
    """Generate a text-based visualization of the knowledge evolution."""
    
    print("\n" + "="*80)
    print("📊 STEVE'S 24-HOUR KNOWLEDGE EVOLUTION ANALYSIS")
    print("="*80)
    
    # Overall Statistics
    print("\n📈 OVERALL GROWTH STATISTICS")
    print("-"*40)
    print(f"  Total Learning Iterations: {stats['total_iterations']:,}")
    print(f"  Average Iterations/Hour:   {stats['iterations_per_hour']:.1f}")
    print(f"  Learning Duration:         ~24 hours")
    
    # Knowledge Base Growth
    print("\n📚 KNOWLEDGE BASE GROWTH")
    print("-"*40)
    
    domains = stats['knowledge_domains']
    initial = {"principles": 25, "core_knowledge": 25, "experiences": 16, "challenges": 4}
    
    for domain, data in domains.items():
        count = data['count']
        init = initial.get(domain, 0)
        growth = count - init
        growth_pct = (growth / init * 100) if init > 0 else 0
        
        bar_len = min(50, count // 20)
        bar = "█" * bar_len
        
        print(f"\n  {domain.upper().replace('_', ' ')}")
        print(f"    Initial:  {init:>5}")
        print(f"    Final:    {count:>5}")
        print(f"    Growth:   +{growth:>4} ({growth_pct:,.0f}%)")
        print(f"    {bar}")
    
    # Topic Coverage (Knowledge Width)
    print("\n\n🌐 KNOWLEDGE WIDTH - TOP TOPICS COVERED")
    print("-"*40)
    
    all_topics = Counter()
    for domain, data in domains.items():
        all_topics.update(data['topics'])
    
    for topic, count in all_topics.most_common(20):
        bar_len = min(30, count)
        bar = "▓" * bar_len
        print(f"  {topic:<20} {bar} ({count})")
    
    # Hourly Activity Pattern
    if stats.get('hourly_activity'):
        print("\n\n⏰ LEARNING ACTIVITY OVER TIME")
        print("-"*40)
        
        hourly = stats['hourly_activity']
        max_activity = max(hourly.values()) if hourly else 1
        
        for hour, count in list(hourly.items())[-24:]:  # Last 24 hours
            bar_len = int(count / max_activity * 40)
            bar = "●" * bar_len
            print(f"  {hour[-5:]} {bar} ({count})")
    
    # Mindset Evolution
    if stats.get('mindset_evolution'):
        print("\n\n🧠 MINDSET EVOLUTION")
        print("-"*40)
        
        mindset = stats['mindset_evolution']
        print(f"  Total Mindset Changes: {mindset['total_mindset_changes']}")
        
        print("\n  Category Breakdown:")
        for cat, count in mindset['categories'].items():
            cat_name = cat.replace('_', ' ').title()
            bar_len = min(30, count // 5)
            bar = "◆" * bar_len if bar_len > 0 else "◇"
            print(f"    {cat_name:<25} {bar} ({count})")
    
    # Learning Event Types
    if stats.get('event_types'):
        print("\n\n📋 LEARNING EVENT TYPES")
        print("-"*40)
        
        for event_type, count in sorted(stats['event_types'].items(), key=lambda x: -x[1]):
            bar_len = min(30, count // 50)
            bar = "■" * bar_len if bar_len > 0 else "□"
            print(f"  {event_type:<30} {bar} ({count})")
    
    # Key Insights Summary
    print("\n\n💡 KEY GROWTH INSIGHTS")
    print("-"*40)
    
    total_knowledge = sum(d['count'] for d in domains.values())
    total_initial = sum(initial.values())
    overall_growth = total_knowledge - total_initial
    
    print(f"""
  ✓ Steve processed {stats['total_iterations']:,} cognitive loop iterations
  ✓ Knowledge base grew from {total_initial} to {total_knowledge:,} items ({overall_growth:,} new)
  ✓ Covered {len(all_topics)} distinct topic areas
  ✓ Average of {stats['iterations_per_hour']:.1f} learning cycles per hour
  ✓ Domains covered: ML architectures, edge AI, continual learning,
    bias/fairness, multimodal fusion, reliability, and more
    """)
    
    print("="*80)
    print("Analysis Complete!")
    print("="*80)


def main():
    """Main analysis function."""
    # Set paths
    base_path = Path(__file__).parent.parent
    actor_path = base_path / "array" / "actors" / "steve"
    
    print("Loading Steve's learning data...")
    
    # Load learning book
    learning_book = load_learning_book(actor_path)
    
    # Get basic stats
    meta = learning_book.get("meta", {})
    version = meta.get("version", 0)
    
    # Analyze learning history
    print("Analyzing learning history...")
    history_analysis = analyze_learning_history(learning_book)
    
    # Analyze knowledge domains
    print("Analyzing knowledge domains...")
    domains = analyze_knowledge_domains(actor_path)
    
    # Analyze mindset evolution
    print("Analyzing mindset evolution...")
    mindset = analyze_mindset_evolution(learning_book)
    
    # Compile statistics
    stats = {
        "total_iterations": version,
        "iterations_per_hour": version / 24,
        "knowledge_domains": domains,
        "hourly_activity": history_analysis.get("hourly_activity", {}),
        "event_types": history_analysis.get("event_types", {}),
        "mindset_evolution": mindset,
    }
    
    # Generate visualization
    generate_text_visualization(stats)
    
    return stats


if __name__ == "__main__":
    stats = main()
