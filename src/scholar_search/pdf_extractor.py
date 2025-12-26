"""
LLM-powered PDF URL extraction and download system.
Uses intelligent parsing and browser automation to fetch academic papers.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.utils.chat_utils import get_chat_completion, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Intelligent PDF extractor that uses LLM to identify PDF URLs from search results
    and download them using various strategies.
    """

    def __init__(self, array_dir: Optional[str] = None):
        """
        Initialize PDF extractor.

        Args:
            array_dir: Base array directory. If None, loads from config.
        """
        if array_dir is None:
            config = load_config()
            array_dir = config.get('array_dir', 'array')

        self.array_dir = Path(array_dir)
        self.pdf_storage = self.array_dir / 'data' / 'pdf_cache'
        self.metadata_file = self.pdf_storage / 'pdf_metadata.json'

        # Create storage directory
        self.pdf_storage.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        self.metadata = self._load_metadata()

        logger.info(f"PDF Extractor initialized. Storage: {self.pdf_storage}")

    def _load_metadata(self) -> Dict:
        """Load PDF metadata bookkeeping file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                return {"pdfs": {}, "last_updated": None}
        return {"pdfs": {}, "last_updated": None}

    def _save_metadata(self):
        """Save PDF metadata bookkeeping file."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.debug("Metadata saved")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def extract_pdf_urls_with_llm(self, search_results: List[Dict], max_papers: int = 5) -> List[Dict]:
        """
        Use LLM to intelligently extract PDF URLs from search results.

        Args:
            search_results: List of search result dicts with title, link, citation, etc.
            max_papers: Maximum number of papers to process

        Returns:
            List of dicts with enhanced metadata including potential PDF URLs
        """
        logger.info(
            f"Using LLM to extract PDF URLs from {len(search_results)} results...")

        # Prepare search results for LLM
        results_text = []
        for i, result in enumerate(search_results[:max_papers], 1):
            results_text.append(f"""
Paper {i}:
Title: {result.get('title', 'N/A')}
Link: {result.get('link', 'N/A')}
Citations: {result.get('citation', '0')}
Publication Info: {result.get('publication_info', 'N/A')}
""")

        prompt = f"""You are an expert at finding PDF download links for academic papers.

Analyze these Google Scholar search results and identify the most likely PDF URLs:

{chr(10).join(results_text)}

For each paper, determine:
1. **Direct PDF URL**: If the main link is a direct PDF, note it
2. **Paper page URL**: If the link goes to a paper landing page (arXiv, ResearchGate, ACM, IEEE, etc.)
3. **PDF extraction strategy**: How to get the PDF from that page
   - ArXiv: Look for /pdf/ version of /abs/ URL
   - ResearchGate: Has "Download full-text PDF" button
   - IEEE: Check for PDF download link on paper page
   - ACM: Look for PDF download button
   - Institutional repos: Check for PDF links
4. **Alternative sources**: Where else might this paper be available (e.g., author's website, preprint servers)

Return a JSON array with this structure:
[
  {{
    "paper_number": 1,
    "title": "Paper title",
    "primary_url": "main link from search",
    "url_type": "direct_pdf|paper_page|unknown",
    "platform": "arxiv|researchgate|ieee|acm|institutional|other",
    "pdf_strategies": [
      {{"method": "direct", "url": "https://...", "confidence": "high|medium|low"}},
      {{"method": "page_parse", "landing_page": "https://...", "selector": "css selector or xpath", "confidence": "medium"}},
      {{"method": "alternative", "url": "https://...", "source": "description", "confidence": "low"}}
    ],
    "notes": "Any relevant observations"
  }}
]

Only include papers where you can identify at least one plausible PDF URL strategy."""

        try:
            response = get_chat_completion([
                {"role": "system", "content": "You are an expert at finding and extracting PDFs from academic paper sources. Return ONLY valid JSON, no explanations."},
                {"role": "user", "content": prompt}
            ], temperature=0.2)

            # Clean up response - remove markdown code blocks if present
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.startswith('```'):
                response_clean = response_clean[3:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()

            # Parse JSON response
            pdf_info = json.loads(response_clean)

            logger.info(
                f"LLM identified PDF strategies for {len(pdf_info)} papers")
            return pdf_info

        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}...")

            # Fallback: create strategies manually for common platforms
            logger.info("Using fallback strategy generation")
            return self._generate_fallback_strategies(search_results, max_papers)

        except Exception as e:
            logger.error(f"LLM PDF extraction failed: {e}")
            return self._generate_fallback_strategies(search_results, max_papers)

    def _generate_fallback_strategies(self, search_results: List[Dict], max_papers: int) -> List[Dict]:
        """
        Generate PDF extraction strategies using rule-based logic (fallback when LLM fails).

        Args:
            search_results: List of search results
            max_papers: Max papers to process

        Returns:
            List of strategy dicts
        """
        logger.info("Generating fallback strategies using rule-based logic")
        strategies = []

        for i, result in enumerate(search_results[:max_papers], 1):
            title = result.get('title', 'Unknown')
            url = result.get('link', '')

            if not url:
                continue

            # Detect platform
            platform = 'other'
            pdf_strategies = []

            if 'arxiv.org' in url:
                platform = 'arxiv'
                if '/abs/' in url:
                    pdf_url = url.replace('/abs/', '/pdf/') + '.pdf'
                    pdf_strategies.append({
                        'method': 'direct',
                        'url': pdf_url,
                        'confidence': 'high'
                    })
            elif 'researchgate.net' in url:
                platform = 'researchgate'
                pdf_strategies.append({
                    'method': 'page_parse',
                    'landing_page': url,
                    'confidence': 'medium'
                })
            elif 'ieee.org' in url:
                platform = 'ieee'
                pdf_strategies.append({
                    'method': 'page_parse',
                    'landing_page': url,
                    'confidence': 'medium'
                })
            elif 'acm.org' in url:
                platform = 'acm'
                pdf_strategies.append({
                    'method': 'page_parse',
                    'landing_page': url,
                    'confidence': 'medium'
                })
            elif url.endswith('.pdf'):
                platform = 'direct'
                pdf_strategies.append({
                    'method': 'direct',
                    'url': url,
                    'confidence': 'high'
                })
            else:
                # Generic landing page
                pdf_strategies.append({
                    'method': 'page_parse',
                    'landing_page': url,
                    'confidence': 'low'
                })

            if pdf_strategies:
                strategies.append({
                    'paper_number': i,
                    'title': title,
                    'primary_url': url,
                    'platform': platform,
                    'pdf_strategies': pdf_strategies,
                    'notes': 'Generated by fallback logic'
                })

        logger.info(f"Generated {len(strategies)} fallback strategies")
        return strategies

    def fetch_page_content(self, url: str, timeout: int = 15) -> Optional[str]:
        """Fetch page HTML content."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        try:
            response = requests.get(
                url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def extract_pdf_from_page(self, url: str, platform: str) -> Optional[str]:
        """
        Extract PDF URL from a paper landing page using platform-specific logic.

        Args:
            url: Landing page URL
            platform: Platform identifier (arxiv, researchgate, etc.)

        Returns:
            PDF URL if found, None otherwise
        """
        logger.info(f"Extracting PDF from {platform} page: {url}")

        # ArXiv: Convert /abs/ to /pdf/
        if 'arxiv.org' in url:
            if '/abs/' in url:
                pdf_url = url.replace('/abs/', '/pdf/') + '.pdf'
                logger.info(f"  ArXiv PDF URL: {pdf_url}")
                return pdf_url
            return None

        # For other platforms, fetch and parse the page
        html = self.fetch_page_content(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        # ResearchGate
        if 'researchgate.net' in url:
            # Look for PDF download links
            pdf_links = soup.find_all('a', href=re.compile(r'\.pdf'))
            for link in pdf_links:
                href = link.get('href', '')
                if 'download' in href.lower() or 'fulltext' in href.lower():
                    full_url = urljoin(url, href)
                    logger.info(f"  ResearchGate PDF: {full_url}")
                    return full_url

        # IEEE
        elif 'ieee.org' in url:
            # Look for PDF download meta tags or links
            pdf_meta = soup.find('meta', {'name': 'citation_pdf_url'})
            if pdf_meta:
                pdf_url = pdf_meta.get('content')
                logger.info(f"  IEEE PDF: {pdf_url}")
                return pdf_url

        # ACM
        elif 'acm.org' in url:
            # Look for PDF links
            pdf_link = soup.find('a', {'title': re.compile(r'PDF', re.I)})
            if pdf_link:
                href = pdf_link.get('href')
                full_url = urljoin(url, href)
                logger.info(f"  ACM PDF: {full_url}")
                return full_url

        # Generic: Look for citation_pdf_url meta tag (common across many platforms)
        pdf_meta = soup.find('meta', {'name': 'citation_pdf_url'})
        if pdf_meta:
            pdf_url = pdf_meta.get('content')
            logger.info(f"  Found PDF via meta tag: {pdf_url}")
            return pdf_url

        # Generic: Look for links containing ".pdf"
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.I))
        if pdf_links:
            href = pdf_links[0].get('href')
            full_url = urljoin(url, href)
            logger.info(f"  Found generic PDF link: {full_url}")
            return full_url

        logger.warning(f"  Could not extract PDF from {platform} page")
        return None

    def download_pdf(self, pdf_url: str, paper_title: str, paper_id: Optional[str] = None) -> Optional[Path]:
        """
        Download PDF and save to storage with metadata.

        Args:
            pdf_url: Direct URL to PDF file
            paper_title: Title of the paper (for filename)
            paper_id: Optional unique identifier

        Returns:
            Path to downloaded file, or None if failed
        """
        # Generate safe filename
        safe_title = re.sub(r'[^\w\s-]', '', paper_title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        safe_title = safe_title[:100]  # Limit length

        if paper_id:
            filename = f"{paper_id}_{safe_title}.pdf"
        else:
            # Use hash of URL as ID
            import hashlib
            url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
            filename = f"{url_hash}_{safe_title}.pdf"

        filepath = self.pdf_storage / filename

        # Check if already downloaded
        if filepath.exists():
            logger.info(f"PDF already exists: {filename}")
            return filepath

        # Download PDF
        logger.info(f"Downloading PDF: {filename}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/pdf'
        }

        try:
            response = requests.get(
                pdf_url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()

            # Verify it's actually a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and not pdf_url.endswith('.pdf'):
                logger.warning(
                    f"Response may not be PDF. Content-Type: {content_type}")

            # Save file
            with open(filepath, 'wb') as f:
                f.write(response.content)

            file_size = filepath.stat().st_size
            logger.info(f"✓ Downloaded {file_size:,} bytes to {filename}")

            # Update metadata
            pdf_id = paper_id or url_hash
            self.metadata['pdfs'][pdf_id] = {
                'filename': filename,
                'title': paper_title,
                'source_url': pdf_url,
                'downloaded_at': datetime.now().isoformat(),
                'file_size': file_size,
                'filepath': str(filepath)
            }
            self._save_metadata()

            return filepath

        except Exception as e:
            logger.error(f"Failed to download PDF from {pdf_url}: {e}")
            return None

    def process_paper_batch(self, search_results: List[Dict], max_papers: int = 5) -> List[Dict]:
        """
        Process a batch of search results: extract PDFs and download them.

        Args:
            search_results: Search results from scholar_caller
            max_papers: Maximum papers to process

        Returns:
            List of results with download status
        """
        logger.info(
            f"Processing batch of {len(search_results)} papers (max: {max_papers})...")

        # Step 1: Use LLM to identify PDF URLs
        pdf_strategies = self.extract_pdf_urls_with_llm(
            search_results, max_papers)

        if not pdf_strategies:
            logger.warning("No PDF strategies identified by LLM")
            return []

        # Step 2: Try to download each PDF
        results = []
        for strategy in pdf_strategies:
            paper_title = strategy.get('title', 'Unknown')
            logger.info(f"\nProcessing: {paper_title}")

            pdf_downloaded = False
            downloaded_path = None

            # Try each strategy in order of confidence
            strategies = strategy.get('pdf_strategies', [])
            strategies.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}.get(
                x.get('confidence', 'low'), 0), reverse=True)

            for pdf_strategy in strategies:
                method = pdf_strategy.get('method')

                if method == 'direct':
                    # Direct PDF URL
                    pdf_url = pdf_strategy.get('url')
                    if pdf_url:
                        downloaded_path = self.download_pdf(
                            pdf_url, paper_title, str(strategy.get('paper_number')))
                        if downloaded_path:
                            pdf_downloaded = True
                            break

                elif method == 'page_parse':
                    # Need to extract PDF from landing page
                    landing_page = pdf_strategy.get('landing_page')
                    platform = strategy.get('platform', 'unknown')
                    if landing_page:
                        pdf_url = self.extract_pdf_from_page(
                            landing_page, platform)
                        if pdf_url:
                            downloaded_path = self.download_pdf(
                                pdf_url, paper_title, str(strategy.get('paper_number')))
                            if downloaded_path:
                                pdf_downloaded = True
                                break

                elif method == 'alternative':
                    # Try alternative source
                    pdf_url = pdf_strategy.get('url')
                    if pdf_url:
                        downloaded_path = self.download_pdf(
                            pdf_url, paper_title, str(strategy.get('paper_number')))
                        if downloaded_path:
                            pdf_downloaded = True
                            break

            # Record result
            result = {
                'title': paper_title,
                'paper_number': strategy.get('paper_number'),
                'platform': strategy.get('platform'),
                'pdf_downloaded': pdf_downloaded,
                'pdf_path': str(downloaded_path) if downloaded_path else None,
                'strategies_tried': len(strategies)
            }
            results.append(result)

            logger.info(
                f"  {'✓ Downloaded' if pdf_downloaded else '✗ Failed'}")

            # Be polite with rate limiting
            time.sleep(2)

        # Summary
        success_count = sum(1 for r in results if r['pdf_downloaded'])
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Batch complete: {success_count}/{len(results)} PDFs downloaded")
        logger.info(f"Storage location: {self.pdf_storage}")
        logger.info(f"{'='*60}")

        return results

    def get_cached_pdf(self, paper_id: str) -> Optional[Path]:
        """Get path to cached PDF by paper ID."""
        if paper_id in self.metadata['pdfs']:
            filepath = Path(self.metadata['pdfs'][paper_id]['filepath'])
            if filepath.exists():
                return filepath
        return None

    def list_cached_pdfs(self) -> List[Dict]:
        """List all cached PDFs with metadata."""
        return [
            {
                'id': pdf_id,
                **info
            }
            for pdf_id, info in self.metadata['pdfs'].items()
        ]

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        pdf_files = list(self.pdf_storage.glob('*.pdf'))
        total_size = sum(f.stat().st_size for f in pdf_files)

        return {
            'total_pdfs': len(pdf_files),
            'total_size_mb': total_size / (1024 * 1024),
            'storage_path': str(self.pdf_storage),
            'metadata_entries': len(self.metadata['pdfs'])
        }


if __name__ == '__main__':
    # Test the extractor
    extractor = PDFExtractor()

    # Example search results (these would come from scholar_caller)
    test_results = [
        {
            'title': 'Attention Is All You Need',
            'link': 'https://arxiv.org/abs/1706.03762',
            'citation': '50000',
            'publication_info': 'A Vaswani, N Shazeer - NeurIPS 2017'
        }
    ]

    results = extractor.process_paper_batch(test_results, max_papers=1)
    print(json.dumps(results, indent=2))
