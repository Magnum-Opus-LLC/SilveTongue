"""
Scholar Search Example

Demonstrates how to use Lingkui's Google Scholar search capabilities.

Usage:
    python examples/scholar_search_example.py "your search query"
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scholar_search import ScholarSearcher, quick_search, quick_survey
from src.utils.research import researcher


def demo_basic_search(query: str):
    """Basic search demonstration."""
    print(f"\n{'='*60}")
    print(f"🔍 Basic Scholar Search: {query}")
    print('='*60)
    
    searcher = ScholarSearcher()
    papers = searcher.search(query, depth=2)
    
    print(f"\nFound {len(papers)} papers\n")
    
    for i, paper in enumerate(papers[:10], 1):
        print(f"{i}. [{paper.citation_count} citations] {paper.title}")
        print(f"   {paper.publication_info}")
        print(f"   {paper.link[:80]}..." if len(paper.link) > 80 else f"   {paper.link}")
        print()


def demo_advanced_search(query: str):
    """Advanced search with filters."""
    print(f"\n{'='*60}")
    print(f"🔬 Advanced Search (2020-2024): {query}")
    print('='*60)
    
    searcher = ScholarSearcher()
    papers = searcher.search_advanced(
        query=query,
        year_from=2020,
        year_to=2024,
        depth=2
    )
    
    print(f"\nFound {len(papers)} papers from 2020-2024\n")
    
    # Get highly cited
    top_papers = searcher.get_highly_cited(papers, top_n=5)
    
    print("Top 5 Most Cited:\n")
    for i, paper in enumerate(top_papers, 1):
        print(f"{i}. [{paper.citation_count}] {paper.title}")
        print(f"   Year: {paper.year or 'Unknown'}, Authors: {paper.authors[:50]}...")
        print()


def demo_survey(query: str):
    """Survey with word analysis and citation histogram."""
    print(f"\n{'='*60}")
    print(f"📊 Research Survey: {query}")
    print('='*60)
    
    survey = quick_survey(query)
    
    print(f"\nSearched at: {survey.get('searched_at', 'Unknown')}")
    
    # Top papers
    print("\n📚 Top Papers:")
    for i, paper in enumerate(survey.get('top_papers', [])[:5], 1):
        print(f"   {i}. [{paper.citation_count}] {paper.title[:60]}...")
    
    # Word analysis
    print("\n📝 Key Terms in Titles:")
    for word, percentage in survey.get('word_analysis', [])[:10]:
        bar = '█' * int(percentage / 2)
        print(f"   {word:20} {bar} {percentage:.1f}%")
    
    # Citation histogram
    print("\n📈 Citation Distribution:")
    for bin_range, percentage in list(survey.get('citation_histogram', {}).items())[:5]:
        bar = '█' * int(percentage / 3)
        print(f"   {bin_range:15} {bar} {percentage:.1f}%")


def demo_researcher_integration(query: str):
    """Using the integrated ResearchManager."""
    print(f"\n{'='*60}")
    print(f"🧠 Lingkui Research Manager: {query}")
    print('='*60)
    
    # Combined web + scholar search
    print("\n--- Combined Research (Web + Scholar) ---\n")
    findings = researcher.conduct_research(query, include_scholar=True)
    print(findings)
    
    # Pure academic research
    print("\n--- Academic Research Only ---\n")
    academic = researcher.conduct_academic_research(query, depth=1)
    print(academic)


def demo_title_analysis():
    """Analyze word frequency in titles."""
    print(f"\n{'='*60}")
    print("📊 Title Word Analysis Demo")
    print('='*60)
    
    searcher = ScholarSearcher()
    papers = searcher.search("machine learning healthcare", depth=2)
    
    analysis = searcher.analyze_titles(papers, top_n=15)
    
    print("\nTop 15 words in paper titles:\n")
    for word, percentage in analysis:
        bar = '█' * int(percentage)
        print(f"   {word:20} {bar} {percentage:.1f}%")


def demo_reference_formatting():
    """Format papers as references."""
    print(f"\n{'='*60}")
    print("📖 Reference Formatting Demo")
    print('='*60)
    
    searcher = ScholarSearcher()
    papers = searcher.search("transformer attention mechanism", depth=1)
    
    top_5 = searcher.get_highly_cited(papers, top_n=5)
    
    print("\nFormatted References:\n")
    print(searcher.format_as_references(top_5))


def main():
    if len(sys.argv) < 2:
        query = "artificial intelligence chemical engineering"
        print(f"No query provided, using default: '{query}'")
    else:
        query = " ".join(sys.argv[1:])
    
    print(f"\n🔬 Scholar Search Demo for: '{query}'")
    print("="*60)
    
    # Run demos
    try:
        demo_basic_search(query)
    except Exception as e:
        print(f"Basic search failed: {e}")
    
    try:
        demo_survey(query)
    except Exception as e:
        print(f"Survey failed: {e}")
    
    try:
        demo_researcher_integration(query)
    except Exception as e:
        print(f"Researcher integration failed: {e}")
    
    print("\n✨ Demo Complete!")


if __name__ == '__main__':
    main()
