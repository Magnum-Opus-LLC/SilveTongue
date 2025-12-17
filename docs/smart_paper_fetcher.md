# Smart Paper Fetcher

Intelligent paper search and PDF downloading system powered by LLM-guided extraction.

## Overview

The Smart Paper Fetcher solves the common problem of failed automated PDF downloads from academic search results by:

1. **LLM-Guided URL Extraction**: Uses language models to intelligently identify PDF URLs from search results
2. **Multi-Strategy Downloading**: Tries multiple methods (direct download, page parsing, browser automation)
3. **Platform-Specific Logic**: Handles ArXiv, ResearchGate, IEEE, ACM, and institutional repositories
4. **Centralized Storage**: Stores PDFs in `array/data/pdf_cache` with metadata bookkeeping
5. **Reusability**: Maintains metadata for sharing PDFs across actors and future uses

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SmartPaperFetcher                         │
│  (Main orchestrator - coordinates search and download)      │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌────────────────┐ ┌────────────┐ ┌──────────────────┐
│ ScholarCaller  │ │PDFExtractor│ │BrowserDownloader │
│ (Google Search)│ │(LLM Extract)│ │  (Playwright)    │
└────────────────┘ └────────────┘ └──────────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  array/data/        │
              │    pdf_cache/       │
              │      ├── *.pdf      │
              │      └── metadata   │
              └─────────────────────┘
```

## Components

### 1. SmartPaperFetcher (`smart_paper_fetcher.py`)

Main interface that orchestrates the entire workflow.

```python
from src.scholar_search import SmartPaperFetcher

fetcher = SmartPaperFetcher(use_browser=True)

result = fetcher.search_and_download(
    query="transformers natural language processing",
    max_papers=5,
    pages=2,
    year_from=2017
)
```

**Methods:**
- `search_and_download()`: Complete search and download workflow
- `download_from_results_file()`: Download from saved search results
- `list_cached_papers()`: List all cached PDFs
- `get_storage_stats()`: Storage statistics

### 2. PDFExtractor (`pdf_extractor.py`)

LLM-powered PDF URL extraction and downloading.

**Key Features:**
- Uses LLM to analyze search results and identify PDF URLs
- Platform-specific extraction logic (ArXiv, ResearchGate, IEEE, ACM)
- Metadata bookkeeping in `pdf_metadata.json`
- Deduplication (skips already-downloaded PDFs)

**Platform Support:**
- **ArXiv**: Converts `/abs/` URLs to `/pdf/`
- **ResearchGate**: Finds "Download full-text PDF" buttons
- **IEEE**: Extracts citation_pdf_url meta tags
- **ACM**: Locates PDF download links
- **Institutional Repos**: Generic PDF link detection

### 3. BrowserDownloader (`browser_downloader.py`)

Browser automation for complex downloads that require JavaScript.

**Requires:** `playwright` (optional dependency)
```bash
pip install playwright
playwright install
```

**Strategies:**
- `direct`: URL is direct PDF
- `button`: Click download button
- `print`: Use browser print-to-PDF
- `auto`: Try strategies in order

**Usage:**
```python
from src.scholar_search.browser_downloader import download_with_browser_sync

success = download_with_browser_sync(
    url="https://example.com/paper",
    output_path=Path("paper.pdf"),
    strategy="auto"
)
```

## Storage Structure

```
array/
└── data/
    └── pdf_cache/
        ├── pdf_metadata.json          # Bookkeeping database
        ├── 1_attention_is_all.pdf     # Downloaded papers
        ├── 2_bert_pretraining.pdf
        ├── search_results_*.json      # Search result archives
        └── last_summary.json          # Latest run summary
```

### Metadata Format

`pdf_metadata.json`:
```json
{
  "pdfs": {
    "paper_1": {
      "filename": "1_attention_is_all.pdf",
      "title": "Attention Is All You Need",
      "source_url": "https://arxiv.org/pdf/1706.03762.pdf",
      "downloaded_at": "2025-12-16T10:30:00",
      "file_size": 2048576,
      "filepath": "/path/to/array/data/pdf_cache/1_attention_is_all.pdf"
    }
  },
  "last_updated": "2025-12-16T10:30:00"
}
```

## Usage Examples

### Example 1: Basic Search and Download

```python
from src.scholar_search import SmartPaperFetcher

fetcher = SmartPaperFetcher()

result = fetcher.search_and_download(
    query="machine learning interpretability",
    max_papers=5,
    pages=2,
    year_from=2020
)

# Result contains:
# - search_results: Google Scholar results
# - download_results: Download status for each paper
# - storage: Storage statistics
```

### Example 2: From Command Line

```bash
# Basic search
python -m src.scholar_search.smart_paper_fetcher \
    "deep learning optimization" \
    --max-papers 5 \
    --pages 2

# With year filter
python -m src.scholar_search.smart_paper_fetcher \
    "neural architecture search" \
    --year-from 2021 \
    --max-papers 10

# List cached papers
python -m src.scholar_search.smart_paper_fetcher "" --list-cache

# Storage statistics
python -m src.scholar_search.smart_paper_fetcher "" --stats
```

### Example 3: Manual PDF Extraction

```python
from src.scholar_search import PDFExtractor

extractor = PDFExtractor()

# Provide search results manually
search_results = [
    {
        'title': 'Your Paper Title',
        'link': 'https://arxiv.org/abs/1234.5678',
        'citation': '100',
        'publication_info': 'Author et al. - Conference 2023'
    }
]

# Extract and download
results = extractor.process_paper_batch(search_results, max_papers=1)
```

### Example 4: Browser Automation Only

```python
from pathlib import Path
from src.scholar_search.browser_downloader import download_with_browser_sync

# Download a difficult PDF that requires JavaScript
success = download_with_browser_sync(
    url="https://dl.acm.org/doi/10.1145/example",
    output_path=Path("paper.pdf"),
    strategy="auto"  # or 'button', 'print', 'direct'
)
```

## LLM Prompt Engineering

The system uses carefully crafted prompts to guide PDF extraction:

```python
prompt = f"""You are an expert at finding PDF download links for academic papers.

Analyze these Google Scholar search results and identify the most likely PDF URLs:

{search_results}

For each paper, determine:
1. **Direct PDF URL**: If the main link is a direct PDF
2. **Paper page URL**: If the link goes to a landing page
3. **PDF extraction strategy**: How to get the PDF
4. **Alternative sources**: Where else might this paper be available

Return JSON with extraction strategies ranked by confidence (high/medium/low).
"""
```

This allows the LLM to:
- Recognize platform-specific patterns
- Suggest multiple fallback strategies
- Prioritize high-confidence methods
- Identify alternative sources

## Configuration

### Environment Variables

```bash
# Optional: Oxylabs proxy for Google Scholar searches
export PROXY_USER_NAME="your_oxylabs_username"
export PROXY_PASS="your_oxylabs_password"
```

### Config.json

```json
{
  "array_dir": "/path/to/array"
}
```

## Error Handling

The system includes robust error handling:

1. **Search Failures**: Retries with exponential backoff
2. **Rate Limiting**: Uses Oxylabs proxy rotation
3. **Download Failures**: Tries multiple strategies before giving up
4. **Invalid PDFs**: Verifies Content-Type headers
5. **Metadata Corruption**: Rebuilds from filesystem if needed

## Performance Considerations

- **Rate Limiting**: 2-4 second delays between requests
- **Parallel Downloads**: Sequential to avoid rate limits
- **Caching**: Skips already-downloaded PDFs
- **Storage**: PDFs stored in shared location for reuse

## Troubleshooting

### PDFs Not Downloading

1. Check if Oxylabs credentials are set (for search)
2. Install Playwright for browser automation:
   ```bash
   pip install playwright
   playwright install
   ```
3. Try with browser automation: `use_browser=True`
4. Check storage permissions in `array/data/pdf_cache`

### Rate Limiting

- Reduce `pages` parameter
- Increase delays in `scholar_caller.py`
- Use Oxylabs proxy credentials

### LLM Not Finding URLs

- Check LLM API configuration in `config.json`
- Verify search results contain valid links
- Try with more specific search queries

## Future Enhancements

Planned improvements:

1. **OCR Integration**: Extract text from scanned PDFs
2. **Citation Network**: Build citation graphs from downloaded papers
3. **Semantic Search**: LLM-based paper relevance filtering
4. **Batch Processing**: Queue system for large-scale downloads
5. **Paper Parser Integration**: Automatic metadata extraction from PDFs
6. **Dashboard Integration**: UI for browsing cached papers

## Integration with Silvertongue

The Smart Paper Fetcher integrates with the cognitive AI framework:

```python
from src.utils.knowledge_base import KnowledgeBase
from src.scholar_search import SmartPaperFetcher

# Fetch papers for actor's learning
fetcher = SmartPaperFetcher()
result = fetcher.search_and_download(
    query=f"{actor.expertise} recent advances",
    max_papers=5
)

# Parse downloaded PDFs
from src.utils.paper_parser import PaperParser
for paper in result['download_results']:
    if paper['pdf_downloaded']:
        parser = PaperParser(paper['pdf_path'])
        content = parser.extract_content()
        
        # Add to knowledge base
        kb = KnowledgeBase(actor_name)
        kb.add_paper(content)
```

## Credits

- **Google Scholar**: Search results source
- **Oxylabs**: Proxy service for rate limit avoidance
- **Playwright**: Browser automation
- **BeautifulSoup**: HTML parsing
- **LLM**: Intelligent URL extraction

## License

Part of the Silvertongue cognitive AI framework.
