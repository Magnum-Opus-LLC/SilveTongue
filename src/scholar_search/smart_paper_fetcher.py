"""
Integrated paper search and download workflow.
Combines scholar search with intelligent PDF extraction and downloading.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.scholar_search.scholar_caller import search_google_scholar
from src.scholar_search.pdf_extractor import PDFExtractor
from src.scholar_search.browser_downloader import download_with_browser_sync, PLAYWRIGHT_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class SmartPaperFetcher:
    """
    End-to-end paper search and download system.
    """

    def __init__(self, array_dir: Optional[str] = None, use_browser: bool = True):
        """
        Initialize smart paper fetcher.

        Args:
            array_dir: Array directory for storage
            use_browser: Whether to use browser automation for difficult downloads
        """
        self.pdf_extractor = PDFExtractor(array_dir)
        self.use_browser = use_browser and PLAYWRIGHT_AVAILABLE

        if use_browser and not PLAYWRIGHT_AVAILABLE:
            logger.warning(
                "Browser automation requested but Playwright not installed")
            logger.info(
                "Install with: pip install playwright && playwright install")

    def search_and_download(
        self,
        query: str,
        max_papers: int = 5,
        pages: int = 2,
        year_from: Optional[int] = None,
        use_oxylabs: bool = True,
        save_results: bool = True
    ) -> Dict:
        """
        Search for papers and download PDFs.

        Args:
            query: Search query
            max_papers: Maximum papers to download
            pages: Number of search result pages to fetch
            year_from: Filter papers from this year onwards
            use_oxylabs: Use Oxylabs proxy for search
            save_results: Save search results JSON

        Returns:
            Dictionary with search results, download results, and statistics
        """
        logger.info(f"{'='*70}")
        logger.info(f"SMART PAPER FETCHER")
        logger.info(f"Query: {query}")
        logger.info(f"Max papers: {max_papers}, Pages: {pages}")
        logger.info(f"{'='*70}\n")

        # Step 1: Search Google Scholar
        logger.info("Step 1: Searching Google Scholar...")
        search_results = search_google_scholar(
            query=query,
            pages=pages,
            year_from=year_from,
            use_oxylabs=use_oxylabs
        )

        if not search_results:
            logger.error("No search results found")
            return {
                'success': False,
                'error': 'No search results',
                'search_results': [],
                'downloads': []
            }

        logger.info(f"  Found {len(search_results)} papers\n")

        # Save search results if requested
        if save_results:
            results_file = self.pdf_extractor.pdf_storage / \
                f'search_results_{query[:30]}.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(search_results, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved search results to {results_file.name}\n")

        # Step 2: Extract and download PDFs
        logger.info("Step 2: Extracting and downloading PDFs...")
        download_results = self.pdf_extractor.process_paper_batch(
            search_results,
            max_papers=max_papers
        )

        # Step 3: Retry failed downloads with browser if enabled
        if self.use_browser:
            failed_downloads = [
                r for r in download_results if not r['pdf_downloaded']]

            if failed_downloads:
                logger.info(
                    f"\nStep 3: Retrying {len(failed_downloads)} failed downloads with browser...")

                for result in failed_downloads:
                    paper_num = result['paper_number']
                    title = result['title']

                    # Find original search result
                    original = next(
                        (r for i, r in enumerate(
                            search_results[:max_papers], 1) if i == paper_num),
                        None
                    )

                    if original:
                        url = original.get('link', '')
                        if url:
                            logger.info(
                                f"\n  Trying browser download for: {title}")

                            # Generate output path
                            safe_title = title[:50].replace(
                                '/', '_').replace(' ', '_')
                            output_path = self.pdf_extractor.pdf_storage / \
                                f"browser_{paper_num}_{safe_title}.pdf"

                            success = download_with_browser_sync(
                                url, output_path, strategy='auto')

                            if success:
                                result['pdf_downloaded'] = True
                                result['pdf_path'] = str(output_path)
                                result['download_method'] = 'browser'
                                logger.info(f"  ✓ Browser download successful")
                            else:
                                logger.info(f"  ✗ Browser download failed")

        # Generate summary
        total_papers = len(search_results)
        papers_processed = len(download_results)
        papers_downloaded = sum(
            1 for r in download_results if r['pdf_downloaded'])

        storage_stats = self.pdf_extractor.get_storage_stats()

        summary = {
            'success': True,
            'query': query,
            'search_results_count': total_papers,
            'papers_processed': papers_processed,
            'papers_downloaded': papers_downloaded,
            'success_rate': f"{papers_downloaded}/{papers_processed}" if papers_processed > 0 else "0/0",
            'search_results': search_results[:max_papers],
            'download_results': download_results,
            'storage': storage_stats
        }

        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY")
        logger.info(f"  Search results: {total_papers}")
        logger.info(f"  Papers processed: {papers_processed}")
        logger.info(
            f"  PDFs downloaded: {papers_downloaded}/{papers_processed}")
        logger.info(
            f"  Success rate: {(papers_downloaded/papers_processed*100) if papers_processed > 0 else 0:.1f}%")
        logger.info(
            f"  Storage: {storage_stats['total_pdfs']} PDFs ({storage_stats['total_size_mb']:.1f} MB)")
        logger.info(f"  Location: {storage_stats['storage_path']}")
        logger.info(f"{'='*70}\n")

        return summary

    def download_from_results_file(self, results_file: Path, max_papers: int = 5) -> Dict:
        """
        Download PDFs from a previously saved search results JSON file.

        Args:
            results_file: Path to search results JSON
            max_papers: Maximum papers to download

        Returns:
            Download results dictionary
        """
        logger.info(f"Loading search results from {results_file}")

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                search_results = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results file: {e}")
            return {'success': False, 'error': str(e)}

        logger.info(f"Loaded {len(search_results)} results")

        download_results = self.pdf_extractor.process_paper_batch(
            search_results,
            max_papers=max_papers
        )

        return {
            'success': True,
            'download_results': download_results
        }

    def list_cached_papers(self) -> List[Dict]:
        """List all cached papers."""
        return self.pdf_extractor.list_cached_pdfs()

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        return self.pdf_extractor.get_storage_stats()


def main():
    """CLI interface for smart paper fetcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Smart Paper Fetcher - Search and download academic papers')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--max-papers', type=int, default=5,
                        help='Maximum papers to download (default: 5)')
    parser.add_argument('--pages', type=int, default=2,
                        help='Search result pages to fetch (default: 2)')
    parser.add_argument('--year-from', type=int,
                        help='Filter papers from this year onwards')
    parser.add_argument('--no-oxylabs', action='store_true',
                        help='Disable Oxylabs proxy')
    parser.add_argument('--no-browser', action='store_true',
                        help='Disable browser automation')
    parser.add_argument('--list-cache', action='store_true',
                        help='List cached papers')
    parser.add_argument('--stats', action='store_true',
                        help='Show storage statistics')

    args = parser.parse_args()

    fetcher = SmartPaperFetcher(use_browser=not args.no_browser)

    if args.list_cache:
        papers = fetcher.list_cached_papers()
        print(json.dumps(papers, indent=2))
        return

    if args.stats:
        stats = fetcher.get_storage_stats()
        print(json.dumps(stats, indent=2))
        return

    # Search and download
    result = fetcher.search_and_download(
        query=args.query,
        max_papers=args.max_papers,
        pages=args.pages,
        year_from=args.year_from,
        use_oxylabs=not args.no_oxylabs
    )

    # Save summary
    summary_file = fetcher.pdf_extractor.pdf_storage / 'last_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
