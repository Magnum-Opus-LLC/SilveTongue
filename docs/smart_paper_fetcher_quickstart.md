# Smart Paper Fetcher - Quick Start

## What is it?

An intelligent PDF downloading system that uses LLM to extract and download academic papers from Google Scholar search results. It fixes the common problem of failed automated PDF downloads by using multiple strategies and browser automation.

## Installation

```bash
# Basic installation (already included in main project)
cd /Volumes/ext_disk/LLM_Projects/silvertongue

# Optional: Install browser automation for complex downloads
pip install playwright
playwright install
```

## Quick Usage

### Python API

```python
from src.scholar_search import SmartPaperFetcher

# Create fetcher
fetcher = SmartPaperFetcher()

# Search and download papers
result = fetcher.search_and_download(
    query="large language models reasoning",
    max_papers=5,
    pages=2,
    year_from=2022
)

# Check results
for paper in result['download_results']:
    if paper['pdf_downloaded']:
        print(f"✓ Downloaded: {paper['title']}")
        print(f"  Path: {paper['pdf_path']}")
```

### Command Line

```bash
# Basic search (max 5 papers from 2 pages)
python -m src.scholar_search.smart_paper_fetcher \
    "transformers attention mechanism" \
    --max-papers 5 \
    --pages 2

# Filter by year
python -m src.scholar_search.smart_paper_fetcher \
    "neural architecture search" \
    --year-from 2020

# List cached papers
python -m src.scholar_search.smart_paper_fetcher "" --list-cache

# Storage statistics  
python -m src.scholar_search.smart_paper_fetcher "" --stats
```

## How It Works

1. **Search**: Queries Google Scholar (with optional Oxylabs proxy)
2. **LLM Analysis**: Uses LLM to intelligently identify PDF URLs from search results
3. **Multi-Strategy Download**:
   - Direct PDF links (ArXiv, etc.)
   - Page parsing (ResearchGate, IEEE, ACM)
   - Browser automation (complex pages with JavaScript)
4. **Storage**: Saves to `array/data/pdf_cache/` with metadata bookkeeping
5. **Reuse**: Deduplicates and shares PDFs across actors

## Storage Location

All PDFs are stored in:
```
array/data/pdf_cache/
├── pdf_metadata.json          # Bookkeeping database
├── 1_paper_title.pdf          # Downloaded papers
├── 2_another_paper.pdf
└── search_results_*.json      # Archived search results
```

## Features

✅ **LLM-Guided Extraction**: Intelligently identifies PDF URLs  
✅ **Platform Support**: ArXiv, ResearchGate, IEEE, ACM, institutional repos  
✅ **Browser Automation**: Handles complex JavaScript pages  
✅ **Metadata Tracking**: Full bookkeeping for reuse  
✅ **Deduplication**: Skips already-downloaded papers  
✅ **Rate Limiting**: Polite delays to avoid blocking  
✅ **Proxy Support**: Oxylabs integration for Google Scholar  

## Examples

See [examples/smart_paper_fetcher_example.py](../examples/smart_paper_fetcher_example.py) for comprehensive examples.

## Configuration

### Optional: Oxylabs Proxy (Recommended)

To avoid rate limiting from Google Scholar:

```bash
export PROXY_USER_NAME="your_oxylabs_username"
export PROXY_PASS="your_oxylabs_password"
```

### Array Directory

Configured in `config.json`:
```json
{
  "array_dir": "/path/to/array"
}
```

## Troubleshooting

### No PDFs Downloaded

1. **Install browser automation** (for complex pages):
   ```bash
   pip install playwright
   playwright install
   ```

2. **Check storage permissions**:
   ```bash
   ls -la array/data/pdf_cache/
   ```

3. **Enable browser mode** in code:
   ```python
   fetcher = SmartPaperFetcher(use_browser=True)
   ```

### Rate Limited by Google Scholar

1. Set up Oxylabs proxy (see Configuration above)
2. Reduce search pages: `--pages 1`
3. Increase delays (edit `scholar_caller.py`)

### LLM Not Finding URLs

1. Check LLM API is working
2. Try more specific queries
3. Verify search results have valid links

## Integration with Silvertongue

The Smart Paper Fetcher integrates seamlessly with the cognitive AI framework:

```python
from src.actions import cognitive_loop
from src.scholar_search import SmartPaperFetcher

# Actor learns from papers
def learn_from_papers(actor_name, topic):
    fetcher = SmartPaperFetcher()
    
    # Download papers on topic
    result = fetcher.search_and_download(
        query=f"{topic} recent advances",
        max_papers=5
    )
    
    # Parse and add to knowledge base
    for paper in result['download_results']:
        if paper['pdf_downloaded']:
            # Parse PDF (use paper_parser.py)
            # Add to knowledge base
            # Generate training examples
            pass
```

## Architecture

```
SmartPaperFetcher (orchestrator)
    ├── ScholarCaller (search Google Scholar)
    ├── PDFExtractor (LLM-guided URL extraction)
    └── BrowserDownloader (Playwright automation)
            ↓
    array/data/pdf_cache/ (shared storage)
```

## Documentation

- **Full Documentation**: [docs/smart_paper_fetcher.md](../docs/smart_paper_fetcher.md)
- **API Reference**: See docstrings in source files
- **Examples**: [examples/smart_paper_fetcher_example.py](../examples/smart_paper_fetcher_example.py)

## Status

✅ **Working**: LLM-guided extraction, multi-strategy downloads, storage  
⚠️ **Optional**: Browser automation (requires Playwright)  
📋 **Planned**: OCR, citation networks, semantic filtering  

Enjoy intelligent paper downloading! 🔬📄
