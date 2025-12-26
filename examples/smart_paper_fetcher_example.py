"""
Example: Smart Paper Fetcher
Demonstrates intelligent paper search and PDF downloading with LLM-guided extraction.
"""

from src.scholar_search.smart_paper_fetcher import SmartPaperFetcher
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_basic_search():
    """Basic search and download example."""
    print("="*80)
    print("EXAMPLE 1: Basic Search and Download")
    print("="*80 + "\n")

    fetcher = SmartPaperFetcher(use_browser=True)

    # Search for papers about transformers in NLP
    result = fetcher.search_and_download(
        query="attention mechanism transformers natural language processing",
        max_papers=3,
        pages=1,
        year_from=2017,
        use_oxylabs=False  # Set to True if you have Oxylabs credentials
    )

    print("\nDownload Results:")
    for paper in result['download_results']:
        status = "✓" if paper['pdf_downloaded'] else "✗"
        print(f"{status} {paper['title']}")
        if paper['pdf_path']:
            print(f"  → {paper['pdf_path']}")

    return result


def example_targeted_search():
    """Targeted search for specific research area."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Targeted Research Search")
    print("="*80 + "\n")

    fetcher = SmartPaperFetcher()

    # Search for recent papers on specific topic
    result = fetcher.search_and_download(
        query="large language model reasoning chain of thought",
        max_papers=5,
        pages=2,
        year_from=2022,
        use_oxylabs=False
    )

    # Show storage stats
    stats = fetcher.get_storage_stats()
    print(f"\nStorage Statistics:")
    print(f"  Total PDFs: {stats['total_pdfs']}")
    print(f"  Total Size: {stats['total_size_mb']:.2f} MB")
    print(f"  Location: {stats['storage_path']}")

    return result


def example_list_cached():
    """List all cached papers."""
    print("\n" + "="*80)
    print("EXAMPLE 3: List Cached Papers")
    print("="*80 + "\n")

    fetcher = SmartPaperFetcher()
    papers = fetcher.list_cached_papers()

    if not papers:
        print("No cached papers yet. Run example_basic_search() first.")
        return

    print(f"Found {len(papers)} cached papers:\n")
    for paper in papers:
        print(f"ID: {paper['id']}")
        print(f"  Title: {paper['title']}")
        print(f"  File: {paper['filename']}")
        print(f"  Size: {paper['file_size'] / 1024:.1f} KB")
        print(f"  Downloaded: {paper['downloaded_at'][:10]}")
        print()


def example_manual_extraction():
    """Manually extract PDFs from specific search results."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Manual PDF Extraction")
    print("="*80 + "\n")

    from src.scholar_search.pdf_extractor import PDFExtractor

    extractor = PDFExtractor()

    # Manually provide search results (e.g., from a previous search)
    search_results = [
        {
            'title': 'Attention Is All You Need',
            'link': 'https://arxiv.org/abs/1706.03762',
            'citation': '90000',
            'publication_info': 'A Vaswani, N Shazeer, et al. - NeurIPS 2017'
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'link': 'https://arxiv.org/abs/1810.04805',
            'citation': '70000',
            'publication_info': 'J Devlin, MW Chang, et al. - NAACL 2019'
        }
    ]

    # Use LLM to extract PDF URLs and download
    results = extractor.process_paper_batch(search_results, max_papers=2)

    for result in results:
        print(f"{'✓' if result['pdf_downloaded'] else '✗'} {result['title']}")
        if result['pdf_path']:
            print(f"  → {result['pdf_path']}")


def example_cli_usage():
    """Show CLI usage examples."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Command Line Interface")
    print("="*80 + "\n")

    print("You can also use the smart paper fetcher from the command line:\n")

    print("# Basic search")
    print('python -m src.scholar_search.smart_paper_fetcher "transformers NLP" --max-papers 5\n')

    print("# Search with year filter")
    print('python -m src.scholar_search.smart_paper_fetcher "machine learning" --year-from 2020\n')

    print("# Search more pages")
    print('python -m src.scholar_search.smart_paper_fetcher "deep learning" --pages 3\n')

    print("# List cached papers")
    print('python -m src.scholar_search.smart_paper_fetcher "" --list-cache\n')

    print("# Show storage stats")
    print('python -m src.scholar_search.smart_paper_fetcher "" --stats\n')

    print("# Disable browser automation")
    print('python -m src.scholar_search.smart_paper_fetcher "AI" --no-browser\n')


def main():
    """Run all examples."""
    print("\n" + "🔬 SMART PAPER FETCHER EXAMPLES 🔬\n")
    print("This demonstrates LLM-powered PDF extraction and downloading.")
    print("The system will:")
    print("  1. Search Google Scholar for papers")
    print("  2. Use LLM to identify PDF URLs from search results")
    print("  3. Download PDFs using multiple strategies (direct, page parsing, browser)")
    print("  4. Store PDFs in array/data/pdf_cache with metadata bookkeeping")
    print("\n" + "="*80 + "\n")

    try:
        # Run basic example
        result1 = example_basic_search()

        # Show CLI usage
        example_cli_usage()

        # List cached papers
        example_list_cached()

        print("\n" + "="*80)
        print("All examples completed!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
