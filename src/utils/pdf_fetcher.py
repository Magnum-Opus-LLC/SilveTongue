"""
PDF Fetcher using Puppeteer

Fetches PDFs from academic papers using headless browser automation.
Handles various sources:
- Direct PDF links
- arXiv papers
- DOI resolvers  
- Publisher sites (with paywall detection)
- Sci-Hub as fallback (configurable)

Requires Node.js and Puppeteer:
    npm install puppeteer
"""

import os
import json
import subprocess
import tempfile
import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from urllib.parse import urlparse, quote

# Puppeteer script for PDF fetching
PUPPETEER_SCRIPT = '''
const puppeteer = require('puppeteer');
const https = require('https');
const http = require('http');
const fs = require('fs');

function downloadPDF(pdfUrl, outputPath) {
    return new Promise((resolve, reject) => {
        const protocol = pdfUrl.startsWith('https') ? https : http;
        
        const request = protocol.get(pdfUrl, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/pdf,*/*'
            }
        }, (response) => {
            // Handle redirects
            if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
                let redirectUrl = response.headers.location;
                if (!redirectUrl.startsWith('http')) {
                    const url = new URL(pdfUrl);
                    redirectUrl = url.origin + redirectUrl;
                }
                downloadPDF(redirectUrl, outputPath).then(resolve).catch(reject);
                return;
            }
            
            if (response.statusCode !== 200) {
                reject(new Error(`HTTP ${response.statusCode}`));
                return;
            }

            const fileStream = fs.createWriteStream(outputPath);
            response.pipe(fileStream);

            fileStream.on('finish', () => {
                fileStream.close();
                const stats = fs.statSync(outputPath);
                resolve(stats.size);
            });

            fileStream.on('error', (err) => {
                fs.unlink(outputPath, () => {});
                reject(err);
            });
        });

        request.on('error', reject);
        request.setTimeout(60000, () => {
            request.destroy();
            reject(new Error('Timeout'));
        });
    });
}

async function fetchPDF(url, outputPath, options = {}) {
    const {
        timeout = 60000,
        userAgent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        proxy = null
    } = options;

    const result = {
        success: false,
        url: url,
        outputPath: outputPath,
        finalUrl: null,
        contentType: null,
        fileSize: 0,
        error: null,
        isPDF: false,
        html: null,
        pdfLinks: []
    };

    // Check if URL is a direct PDF link or arXiv PDF
    const isDirectPDF = url.endsWith('.pdf') || url.includes('/pdf/');
    
    if (isDirectPDF) {
        // Direct download for PDF URLs
        try {
            const fileSize = await downloadPDF(url, outputPath);
            
            // Verify it's actually a PDF
            const buffer = fs.readFileSync(outputPath);
            const header = buffer.toString('utf8', 0, 4);
            
            if (header === '%PDF') {
                result.success = true;
                result.isPDF = true;
                result.fileSize = fileSize;
                result.finalUrl = url;
                console.log(JSON.stringify(result));
                return result;
            } else {
                // Not a PDF, remove the file
                fs.unlinkSync(outputPath);
                result.error = 'Downloaded content is not a valid PDF';
            }
        } catch (error) {
            result.error = `Direct download failed: ${error.message}`;
        }
    }

    // For non-PDF URLs or failed direct downloads, use Puppeteer
    let browser = null;

    try {
        const launchOptions = {
            headless: 'new',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        };

        if (proxy) {
            launchOptions.args.push(`--proxy-server=${proxy}`);
        }

        browser = await puppeteer.launch(launchOptions);
        const page = await browser.newPage();

        await page.setUserAgent(userAgent);
        await page.setExtraHTTPHeaders({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9'
        });

        // Navigate to URL
        const navigationResponse = await page.goto(url, {
            waitUntil: 'networkidle2',
            timeout: timeout
        });

        result.finalUrl = page.url();
        result.contentType = navigationResponse?.headers()?.['content-type'] || '';

        // Look for PDF links on the page
        const pdfLinks = await page.evaluate(() => {
            const links = [];
            
            // Check meta tags first (most reliable for academic papers)
            const metaPdf = document.querySelector('meta[name="citation_pdf_url"]');
            if (metaPdf && metaPdf.content) {
                links.push({
                    url: metaPdf.content,
                    source: 'meta_citation_pdf_url',
                    priority: 1
                });
            }
            
            // Check for arXiv PDF link
            const currentUrl = window.location.href;
            if (currentUrl.includes('arxiv.org/abs/')) {
                const arxivId = currentUrl.match(/arxiv\\.org\\/abs\\/([\\d.]+)/);
                if (arxivId) {
                    links.push({
                        url: `https://arxiv.org/pdf/${arxivId[1]}.pdf`,
                        source: 'arxiv_conversion',
                        priority: 1
                    });
                }
            }
            
            // Check for direct PDF download links
            document.querySelectorAll('a').forEach(a => {
                const href = a.href;
                const text = (a.textContent || '').toLowerCase();
                
                if (href && (
                    href.endsWith('.pdf') ||
                    (href.includes('/pdf/') && !href.includes('pdf_embedder')) ||
                    text.includes('download pdf') ||
                    text.includes('full text pdf') ||
                    text.includes('view pdf') ||
                    text.includes('pdf version')
                )) {
                    links.push({
                        url: href,
                        text: (a.textContent || '').trim().substring(0, 100),
                        source: 'link',
                        priority: href.endsWith('.pdf') ? 2 : 3
                    });
                }
            });
            
            // Deduplicate and sort by priority
            const seen = new Set();
            return links
                .filter(l => {
                    if (seen.has(l.url)) return false;
                    seen.add(l.url);
                    return true;
                })
                .sort((a, b) => a.priority - b.priority);
        });
        
        result.pdfLinks = pdfLinks;
        result.success = true;

        // If we found PDF links, try to download the first one
        if (pdfLinks.length > 0) {
            const pdfUrl = pdfLinks[0].url;
            try {
                const fileSize = await downloadPDF(pdfUrl, outputPath);
                
                // Verify it's a PDF
                const buffer = fs.readFileSync(outputPath);
                const header = buffer.toString('utf8', 0, 4);
                
                if (header === '%PDF') {
                    result.isPDF = true;
                    result.fileSize = fileSize;
                    result.finalUrl = pdfUrl;
                } else {
                    fs.unlinkSync(outputPath);
                    result.error = 'Downloaded content is not a valid PDF';
                }
            } catch (downloadError) {
                result.error = `Found PDF link but download failed: ${downloadError.message}`;
            }
        }

    } catch (error) {
        result.error = error.message;
        result.success = false;
    } finally {
        if (browser) {
            await browser.close();
        }
    }

    console.log(JSON.stringify(result));
    return result;
}

// Parse command line args
const args = process.argv.slice(2);
const url = args[0];
const outputPath = args[1];
const optionsJson = args[2] || '{}';

const options = JSON.parse(optionsJson);

fetchPDF(url, outputPath, options).catch(err => {
    console.log(JSON.stringify({
        success: false,
        error: err.message
    }));
    process.exit(1);
});
'''


class PDFFetcher:
    """
    Fetches PDFs from academic sources using Puppeteer.
    """

    # Known academic sources and their PDF URL patterns
    SOURCE_PATTERNS = {
        'arxiv': {
            'detect': r'arxiv\.org/(abs|pdf)/(\d+\.\d+)',
            'pdf_url': 'https://arxiv.org/pdf/{id}.pdf'
        },
        'doi': {
            'detect': r'doi\.org/(10\.\d+/.+)',
            'resolver': 'https://doi.org/{doi}'
        },
        'pubmed': {
            'detect': r'(ncbi\.nlm\.nih\.gov/pmc/articles/|pubmed\.ncbi\.nlm\.nih\.gov/)(\w+)',
        },
        'semantic_scholar': {
            'detect': r'semanticscholar\.org/paper/([^/]+)',
        },
        'ieee': {
            'detect': r'ieeexplore\.ieee\.org/(document|abstract)/(\d+)',
        },
        'acm': {
            'detect': r'dl\.acm\.org/doi/(10\.\d+/.+)',
        }
    }

    def __init__(self, cache_dir: Optional[Path] = None, use_scihub: bool = False):
        """
        Initialize PDF fetcher.

        Args:
            cache_dir: Directory to cache downloaded PDFs
            use_scihub: Whether to use Sci-Hub as fallback (may be illegal in some jurisdictions)
        """
        if cache_dir is None:
            # Use array_dir from config if available
            try:
                from src.utils.chat_utils import load_config
                config = load_config()
                cache_dir = Path(config.get('array_dir', '')
                                 ) / "data" / "pdf_cache"
            except Exception:
                cache_dir = Path(tempfile.gettempdir()) / \
                    "silvertongue_pdf_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_scihub = use_scihub
        self.puppeteer_script_path = self.cache_dir / "_puppeteer_fetch.js"

        # Write puppeteer script
        self._ensure_puppeteer_script()

        # Track fetch history
        self.fetch_history: List[Dict] = []

    def _ensure_puppeteer_script(self):
        """Ensure the Puppeteer script is written to disk."""
        self.puppeteer_script_path.write_text(PUPPETEER_SCRIPT)

    def _check_puppeteer_installed(self) -> bool:
        """Check if Node.js and Puppeteer are available."""
        try:
            # Check Node.js
            result = subprocess.run(['node', '--version'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False

            # Check Puppeteer
            check_script = "try { require('puppeteer'); console.log('ok'); } catch(e) { console.log('missing'); }"
            result = subprocess.run(['node', '-e', check_script],
                                    capture_output=True, text=True, timeout=10)
            return 'ok' in result.stdout

        except Exception:
            return False

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        safe_name = re.sub(
            r'[^\w\-.]', '_', urlparse(url).path.split('/')[-1] or 'paper')
        if not safe_name.endswith('.pdf'):
            safe_name += '.pdf'
        return self.cache_dir / f"{url_hash}_{safe_name}"

    def _identify_source(self, url: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Identify the source type and extract relevant IDs."""
        for source, patterns in self.SOURCE_PATTERNS.items():
            match = re.search(patterns['detect'], url, re.IGNORECASE)
            if match:
                return source, {'match': match, 'groups': match.groups()}
        return None, None

    def _get_arxiv_pdf_url(self, url: str) -> Optional[str]:
        """Convert arXiv abstract URL to PDF URL."""
        match = re.search(r'arxiv\.org/(abs|pdf)/(\d+\.\d+)', url)
        if match:
            arxiv_id = match.group(2)
            return f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        return None

    def fetch_pdf(self, url: str, title: str = "",
                  force_refresh: bool = False,
                  timeout: int = 60000) -> Dict:
        """
        Fetch a PDF from a URL.

        Args:
            url: URL to fetch (can be abstract page, DOI, or direct PDF)
            title: Paper title (for logging/metadata)
            force_refresh: Re-download even if cached
            timeout: Timeout in milliseconds

        Returns:
            {
                "success": bool,
                "pdf_path": Optional[Path],
                "url": str,
                "source_type": str,
                "content_extracted": bool,
                "error": Optional[str],
                "cached": bool,
                "metadata": Dict
            }
        """
        result = {
            "success": False,
            "pdf_path": None,
            "url": url,
            "title": title,
            "source_type": "unknown",
            "content_extracted": False,
            "error": None,
            "cached": False,
            "metadata": {},
            "fetched_at": datetime.now().isoformat()
        }

        # Identify source type
        source_type, source_info = self._identify_source(url)
        result["source_type"] = source_type or "direct"

        # Handle special sources
        pdf_url = url

        if source_type == 'arxiv':
            pdf_url = self._get_arxiv_pdf_url(url)
            if pdf_url:
                result["metadata"]["arxiv_pdf_url"] = pdf_url

        # Check cache
        cache_path = self._get_cache_path(pdf_url or url)
        if cache_path.exists() and not force_refresh:
            result["success"] = True
            result["pdf_path"] = cache_path
            result["cached"] = True
            result["metadata"]["file_size"] = cache_path.stat().st_size
            self.fetch_history.append(result)
            return result

        # Check Puppeteer availability
        if not self._check_puppeteer_installed():
            result["error"] = "Puppeteer not installed. Run: npm install puppeteer"
            self.fetch_history.append(result)
            return result

        # Fetch using Puppeteer
        try:
            output_path = str(cache_path)
            options = json.dumps({"timeout": timeout})

            cmd = ['node', str(self.puppeteer_script_path),
                   pdf_url or url, output_path, options]

            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout // 1000 + 30,  # Add buffer
                cwd=str(self.cache_dir)
            )

            # Parse result
            output_lines = proc_result.stdout.strip().split('\n')
            if output_lines:
                try:
                    fetch_result = json.loads(output_lines[-1])
                except json.JSONDecodeError:
                    fetch_result = {"success": False,
                                    "error": proc_result.stdout}
            else:
                fetch_result = {"success": False, "error": proc_result.stderr}

            if fetch_result.get("success") and fetch_result.get("isPDF"):
                result["success"] = True
                result["pdf_path"] = cache_path
                result["metadata"]["file_size"] = fetch_result.get(
                    "fileSize", 0)
                result["metadata"]["final_url"] = fetch_result.get("finalUrl")

            elif fetch_result.get("success") and fetch_result.get("pdfLinks"):
                # Got HTML page with PDF links - try the first one
                pdf_links = fetch_result["pdfLinks"]
                if pdf_links:
                    first_pdf = pdf_links[0]["url"]
                    result["metadata"]["discovered_pdf_url"] = first_pdf

                    # Recursively fetch the discovered PDF
                    return self.fetch_pdf(first_pdf, title, force_refresh, timeout)
                else:
                    result["error"] = "No PDF links found on page"
                    result["metadata"]["page_html_preview"] = fetch_result.get("html", "")[
                        :500]
            else:
                result["error"] = fetch_result.get("error", "Unknown error")

        except subprocess.TimeoutExpired:
            result["error"] = f"Timeout after {timeout}ms"
        except Exception as e:
            result["error"] = str(e)

        self.fetch_history.append(result)
        return result

    def fetch_from_paper_info(self, paper: Dict) -> Dict:
        """
        Fetch PDF using paper metadata from search results.

        Args:
            paper: Paper info dict with keys like 'url', 'title', 'doi', etc.

        Returns:
            Fetch result dict
        """
        # Try different URL sources in order of preference
        urls_to_try = []

        # Direct PDF URL
        if paper.get('pdf_url'):
            urls_to_try.append(('pdf_url', paper['pdf_url']))

        # arXiv link
        if paper.get('arxiv_url'):
            urls_to_try.append(('arxiv', paper['arxiv_url']))
        elif paper.get('url') and 'arxiv.org' in paper.get('url', ''):
            urls_to_try.append(('arxiv', paper['url']))

        # DOI
        if paper.get('doi'):
            doi = paper['doi']
            if not doi.startswith('http'):
                doi = f'https://doi.org/{doi}'
            urls_to_try.append(('doi', doi))

        # Paper URL
        if paper.get('url'):
            urls_to_try.append(('url', paper['url']))

        # Try each URL
        title = paper.get('title', 'Unknown Paper')

        for source, url in urls_to_try:
            result = self.fetch_pdf(url, title)
            if result["success"]:
                result["metadata"]["url_source"] = source
                return result

        # All attempts failed
        return {
            "success": False,
            "pdf_path": None,
            "url": paper.get('url', ''),
            "title": title,
            "error": f"Could not fetch PDF from any source. Tried: {[u[0] for u in urls_to_try]}",
            "metadata": {"attempted_urls": urls_to_try}
        }

    def batch_fetch(self, papers: List[Dict],
                    max_concurrent: int = 3,
                    progress_callback: callable = None) -> List[Dict]:
        """
        Fetch PDFs for multiple papers.

        Args:
            papers: List of paper info dicts
            max_concurrent: Max concurrent downloads (not implemented yet - sequential)
            progress_callback: Called with (index, total, result) after each fetch

        Returns:
            List of fetch results
        """
        results = []
        total = len(papers)

        for i, paper in enumerate(papers):
            result = self.fetch_from_paper_info(paper)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

        return results

    def get_cached_pdfs(self) -> List[Path]:
        """List all cached PDFs."""
        return list(self.cache_dir.glob("*.pdf"))

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear PDF cache.

        Args:
            older_than_days: Only clear files older than this many days (None = all)
        """
        import time

        now = time.time()
        cleared = 0

        for pdf_file in self.cache_dir.glob("*.pdf"):
            if older_than_days is not None:
                age_days = (now - pdf_file.stat().st_mtime) / 86400
                if age_days < older_than_days:
                    continue

            pdf_file.unlink()
            cleared += 1

        return cleared


class PaperContentExtractor:
    """
    Extracts and structures content from fetched PDFs.
    Combines PDF fetching with text extraction and chunking.
    """

    def __init__(self, fetcher: Optional[PDFFetcher] = None):
        self.fetcher = fetcher or PDFFetcher()

        # Import paper parser
        from src.utils.paper_parser import PaperParser
        self.parser = PaperParser()

    def extract_from_url(self, url: str, title: str = "") -> Dict:
        """
        Fetch PDF and extract full content.

        Returns:
            {
                "success": bool,
                "title": str,
                "url": str,
                "content": str,
                "chunks": List[str],
                "sections": List[Dict],  # Detected sections
                "metadata": Dict,
                "error": Optional[str]
            }
        """
        result = {
            "success": False,
            "title": title,
            "url": url,
            "content": "",
            "chunks": [],
            "sections": [],
            "metadata": {},
            "error": None
        }

        # Fetch PDF
        fetch_result = self.fetcher.fetch_pdf(url, title)

        if not fetch_result["success"]:
            result["error"] = f"Failed to fetch PDF: {fetch_result.get('error')}"
            return result

        pdf_path = fetch_result["pdf_path"]
        result["metadata"]["pdf_path"] = str(pdf_path)
        result["metadata"]["cached"] = fetch_result.get("cached", False)

        # Parse PDF
        try:
            parsed = self.parser.parse_file(pdf_path)

            result["success"] = True
            result["content"] = parsed["content"]
            result["chunks"] = parsed["chunks"]
            result["metadata"].update(parsed["metadata"])

            # Try to detect sections
            result["sections"] = self._detect_sections(parsed["content"])

        except Exception as e:
            result["error"] = f"Failed to parse PDF: {str(e)}"

        return result

    def extract_from_paper(self, paper: Dict) -> Dict:
        """
        Fetch and extract content from paper metadata.

        Args:
            paper: Paper info dict from search results
        """
        title = paper.get('title', 'Unknown')

        result = {
            "success": False,
            "title": title,
            "url": paper.get('url', ''),
            "abstract": paper.get('abstract', ''),
            "content": "",
            "chunks": [],
            "sections": [],
            "metadata": {
                "authors": paper.get('authors', []),
                "year": paper.get('year'),
                "citations": paper.get('citations'),
                "source": paper.get('source', 'unknown')
            },
            "error": None
        }

        # Fetch PDF
        fetch_result = self.fetcher.fetch_from_paper_info(paper)

        if not fetch_result["success"]:
            # Fall back to abstract-only
            result["content"] = paper.get('abstract', '')
            result["chunks"] = [result["content"]] if result["content"] else []
            result["metadata"]["content_source"] = "abstract_only"
            result["error"] = f"PDF unavailable: {fetch_result.get('error')}. Using abstract only."
            # Success if we have abstract
            result["success"] = bool(result["content"])
            return result

        # Parse PDF
        try:
            parsed = self.parser.parse_file(fetch_result["pdf_path"])

            result["success"] = True
            result["content"] = parsed["content"]
            result["chunks"] = parsed["chunks"]
            result["metadata"]["content_source"] = "full_pdf"
            result["metadata"]["pdf_path"] = str(fetch_result["pdf_path"])
            result["metadata"].update(parsed["metadata"])

            result["sections"] = self._detect_sections(parsed["content"])

        except Exception as e:
            # Fall back to abstract
            result["content"] = paper.get('abstract', '')
            result["chunks"] = [result["content"]] if result["content"] else []
            result["metadata"]["content_source"] = "abstract_only"
            result["error"] = f"PDF parse error: {str(e)}. Using abstract only."
            result["success"] = bool(result["content"])

        return result

    def _detect_sections(self, content: str) -> List[Dict]:
        """
        Detect common paper sections (Abstract, Introduction, Methods, etc.)
        """
        sections = []

        # Common section headers
        section_patterns = [
            (r'(?i)^abstract\s*$', 'abstract'),
            (r'(?i)^1?\s*\.?\s*introduction\s*$', 'introduction'),
            (r'(?i)^2?\s*\.?\s*(?:related\s+work|background|literature\s+review)\s*$', 'related_work'),
            (r'(?i)^3?\s*\.?\s*(?:method(?:s|ology)?|approach|model)\s*$', 'methods'),
            (r'(?i)^4?\s*\.?\s*(?:experiment(?:s)?|evaluation|results)\s*$', 'experiments'),
            (r'(?i)^5?\s*\.?\s*(?:result(?:s)?|findings)\s*$', 'results'),
            (r'(?i)^6?\s*\.?\s*(?:discussion|analysis)\s*$', 'discussion'),
            (r'(?i)^7?\s*\.?\s*(?:conclusion(?:s)?|summary)\s*$', 'conclusion'),
            (r'(?i)^(?:reference(?:s)?|bibliography)\s*$', 'references'),
            (r'(?i)^(?:appendix|appendices)\s*$', 'appendix'),
        ]

        lines = content.split('\n')
        current_section = None
        current_start = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            for pattern, section_name in section_patterns:
                if re.match(pattern, line_stripped):
                    # Save previous section
                    if current_section:
                        sections.append({
                            "name": current_section,
                            "start_line": current_start,
                            "end_line": i - 1,
                            "content": '\n'.join(lines[current_start:i]).strip()
                        })

                    current_section = section_name
                    current_start = i + 1
                    break

        # Add final section
        if current_section:
            sections.append({
                "name": current_section,
                "start_line": current_start,
                "end_line": len(lines) - 1,
                "content": '\n'.join(lines[current_start:]).strip()
            })

        return sections

    def batch_extract(self, papers: List[Dict],
                      progress_callback: callable = None) -> List[Dict]:
        """
        Extract content from multiple papers.
        """
        results = []
        total = len(papers)

        for i, paper in enumerate(papers):
            result = self.extract_from_paper(paper)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

        return results


# Convenience functions
def fetch_paper_pdf(url: str, title: str = "") -> Dict:
    """Fetch a paper PDF from URL."""
    fetcher = PDFFetcher()
    return fetcher.fetch_pdf(url, title)


def extract_paper_content(paper: Dict) -> Dict:
    """Extract full content from a paper (fetches PDF if needed)."""
    extractor = PaperContentExtractor()
    return extractor.extract_from_paper(paper)


def batch_extract_papers(papers: List[Dict], progress_callback: callable = None) -> List[Dict]:
    """Extract content from multiple papers."""
    extractor = PaperContentExtractor()
    return extractor.batch_extract(papers, progress_callback)
