"""
PDF Stock Management for the Silvertongue Dashboard.

Manages the PDF cache index for fast lookups and matching of cached PDFs.
"""

import re
import json
import time
from pathlib import Path
from datetime import datetime

from .utils import get_pdf_cache_dir, get_pdf_stock_path


# In-memory cache for PDF stock
_pdf_stock_cache = {'stock': None, 'loaded_at': 0}


def normalize_title(title: str) -> str:
    """Normalize a title for matching - lowercase, remove punctuation, collapse spaces."""
    if not title:
        return ''
    # Lowercase
    normalized = title.lower()
    # Remove punctuation except hyphens
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    # Collapse multiple spaces/underscores to single space
    normalized = re.sub(r'[\s_]+', ' ', normalized).strip()
    return normalized


def sanitize_title_for_filename(title: str) -> str:
    """Sanitize a paper title to match PDF cache naming convention."""
    # Remove special characters and normalize spaces
    sanitized = re.sub(r'[^\w\s-]', '', title)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    # Limit length
    return sanitized[:100]


def load_pdf_stock() -> dict:
    """Load the PDF stock index from disk."""
    stock_path = get_pdf_stock_path()
    if stock_path.exists():
        try:
            with open(stock_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'pdfs': {}, 'title_index': {}, 'updated_at': None}


def save_pdf_stock(stock: dict):
    """Save the PDF stock index to disk."""
    stock_path = get_pdf_stock_path()
    stock['updated_at'] = datetime.now().isoformat()
    with open(stock_path, 'w') as f:
        json.dump(stock, f, indent=2)


def rebuild_pdf_stock() -> dict:
    """Rebuild the PDF stock index by scanning the cache directory."""
    pdf_cache_dir = get_pdf_cache_dir()

    stock = {
        # filename -> {path, size_mb, original_title, normalized_title, added_at}
        'pdfs': {},
        'title_index': {},    # normalized_title -> filename (for quick lookup)
        'updated_at': datetime.now().isoformat(),
        'total_count': 0,
        'total_size_mb': 0
    }

    if not pdf_cache_dir.exists():
        return stock

    total_size = 0
    for pdf_file in pdf_cache_dir.glob('*.pdf'):
        filename = pdf_file.name
        stem = pdf_file.stem

        # Extract title from filename (format: hash_title.pdf)
        if '_' in stem:
            title_part = stem.split('_', 1)[1]
        else:
            title_part = stem

        # Convert underscores back to spaces for the original title
        original_title = title_part.replace('_', ' ')
        normalized = normalize_title(original_title)

        try:
            size_mb = round(pdf_file.stat().st_size / (1024 * 1024), 2)
        except:
            size_mb = 0

        total_size += size_mb

        stock['pdfs'][filename] = {
            'path': str(pdf_file),
            'size_mb': size_mb,
            'original_title': original_title,
            'normalized_title': normalized,
            'references': []  # Papers that matched this PDF
        }

        # Index by normalized title for quick lookup
        if normalized:
            stock['title_index'][normalized] = filename

    stock['total_count'] = len(stock['pdfs'])
    stock['total_size_mb'] = round(total_size, 2)

    # Save to disk
    save_pdf_stock(stock)
    print(
        f"[PDF Stock] Rebuilt index: {stock['total_count']} PDFs, {stock['total_size_mb']} MB total")

    return stock


def get_pdf_stock(force_rebuild: bool = False) -> dict:
    """Get the PDF stock, loading from disk or rebuilding if needed."""
    now = time.time()

    # Check if we need to reload (every 5 minutes or if forced)
    if force_rebuild or _pdf_stock_cache['stock'] is None or (now - _pdf_stock_cache['loaded_at']) > 300:
        stock_path = get_pdf_stock_path()

        if not force_rebuild and stock_path.exists():
            # Load existing stock
            _pdf_stock_cache['stock'] = load_pdf_stock()
            _pdf_stock_cache['loaded_at'] = now
        else:
            # Rebuild stock
            _pdf_stock_cache['stock'] = rebuild_pdf_stock()
            _pdf_stock_cache['loaded_at'] = now

    return _pdf_stock_cache['stock']


def clear_pdf_stock_cache():
    """Clear the in-memory PDF stock cache."""
    _pdf_stock_cache['stock'] = None
    _pdf_stock_cache['loaded_at'] = 0


def find_cached_pdf_by_path(source_file: str) -> dict:
    """
    Find a cached PDF by its exact source_file path.
    Returns dict with 'found', 'filename', 'path', 'size_mb'.
    """
    if not source_file:
        return {'found': False}

    pdf_path = Path(source_file)
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        return {
            'found': True,
            'filename': pdf_path.name,
            'path': str(pdf_path),
            'size_mb': round(size_mb, 2)
        }
    return {'found': False}


def find_pdf_in_stock(title: str, knowledge_bits: list = None) -> dict:
    """
    Find a cached PDF using the stock index.
    Uses multiple matching strategies:
    1. Direct path from knowledge_bits source_file
    2. Exact normalized title match
    3. Fuzzy title match (significant word overlap)

    Returns dict with 'found', 'filename', 'path', 'size_mb', 'match_type'.
    """
    # First, try to find via source_file in knowledge_bits
    if knowledge_bits:
        for bit in knowledge_bits:
            source_file = bit.get('source_file', '')
            if source_file:
                result = find_cached_pdf_by_path(source_file)
                if result['found']:
                    result['match_type'] = 'source_file'
                    return result

    # Load the stock index
    stock = get_pdf_stock()
    if not stock or not stock.get('pdfs'):
        return {'found': False}

    # Normalize the search title
    normalized_search = normalize_title(title)
    if not normalized_search:
        return {'found': False}

    # Strategy 2: Exact normalized title match
    if normalized_search in stock['title_index']:
        filename = stock['title_index'][normalized_search]
        pdf_info = stock['pdfs'].get(filename)
        if pdf_info:
            return {
                'found': True,
                'filename': filename,
                'path': pdf_info['path'],
                'size_mb': pdf_info['size_mb'],
                'match_type': 'exact_title'
            }

    # Strategy 3: Fuzzy matching - check word overlap
    search_words = set(normalized_search.split())
    # Filter to significant words (length > 3)
    search_words = {w for w in search_words if len(w) > 3}

    if len(search_words) >= 2:
        best_match = None
        best_score = 0

        for filename, pdf_info in stock['pdfs'].items():
            pdf_normalized = pdf_info.get('normalized_title', '')
            pdf_words = set(pdf_normalized.split())
            pdf_words = {w for w in pdf_words if len(w) > 3}

            if not pdf_words:
                continue

            # Calculate overlap score
            overlap = search_words & pdf_words
            if len(overlap) >= 2:
                # Score based on how much of both titles match
                score = len(overlap) / max(len(search_words), len(pdf_words))

                # Bonus for matching first few significant words
                search_first = [w for w in normalized_search.split()[
                    :3] if len(w) > 3]
                pdf_first = [w for w in pdf_normalized.split()[:3]
                             if len(w) > 3]
                if search_first and pdf_first and search_first[0] == pdf_first[0]:
                    score += 0.3

                if score > best_score and score >= 0.5:  # Require at least 50% match
                    best_score = score
                    best_match = (filename, pdf_info)

        if best_match:
            filename, pdf_info = best_match
            return {
                'found': True,
                'filename': filename,
                'path': pdf_info['path'],
                'size_mb': pdf_info['size_mb'],
                'match_type': f'fuzzy_{int(best_score*100)}%'
            }

    return {'found': False}


# Legacy function name for compatibility
def find_cached_pdf(title: str, knowledge_bits: list = None) -> dict:
    """Find a cached PDF - wrapper for find_pdf_in_stock."""
    return find_pdf_in_stock(title, knowledge_bits)
