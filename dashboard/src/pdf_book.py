# filepath: /Users/ruihe/mnt/silvertongueArray/silvertongue/dashboard/src/pdf_book.py
"""
PDF Book - Central Database for PDF Management in Silvertongue.

This module provides a comprehensive PDF indexing and management system that:
- Validates PDFs (detects corrupted/empty files)
- Identifies and tracks duplicates
- Maintains metadata (title, authors, source, study status)
- Tracks which actors have studied which papers
- Provides fast lookups by title, hash, or filename

The PDF Book is stored as a JSON file (pdf_book.json) in the PDF cache directory.
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

from .utils import get_pdf_cache_dir, get_array_dir


# ==================== Constants ====================

PDF_BOOK_FILENAME = 'pdf_book.json'
MIN_VALID_PDF_SIZE = 1024  # 1KB - files smaller than this are likely corrupted
PDF_HEADER = b'%PDF'  # Valid PDFs start with this


# ==================== In-memory Cache ====================

_pdf_book_cache = {
    'book': None,
    'loaded_at': 0,
    'cache_ttl': 300  # 5 minutes
}


# ==================== Helper Functions ====================

def get_pdf_book_path() -> Path:
    """Get the path to the PDF book database file."""
    return get_pdf_cache_dir() / PDF_BOOK_FILENAME


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            first_chunk = f.read(chunk_size)
            hasher.update(first_chunk)
            f.seek(0, 2)
            file_size = f.tell()
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, 2)
                last_chunk = f.read(chunk_size)
                hasher.update(last_chunk)
            hasher.update(str(file_size).encode())
        return hasher.hexdigest()[:16]
    except Exception:
        return ''


def validate_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Validate a PDF file and return its status.
    
    Returns:
        {
            'valid': bool,
            'status': 'valid' | 'empty' | 'too_small' | 'corrupted' | 'not_pdf' | 'missing',
            'size_bytes': int,
            'has_content': bool,
            'error': str (if any)
        }
    """
    result = {
        'valid': False,
        'status': 'unknown',
        'size_bytes': 0,
        'has_content': False,
        'error': None
    }
    
    try:
        if not file_path.exists():
            result['status'] = 'missing'
            result['error'] = 'File does not exist'
            return result
        
        size = file_path.stat().st_size
        result['size_bytes'] = size
        
        if size == 0:
            result['status'] = 'empty'
            result['error'] = 'File is empty (0 bytes)'
            return result
        
        if size < MIN_VALID_PDF_SIZE:
            result['status'] = 'too_small'
            result['error'] = f'File too small ({size} bytes), likely corrupted'
            return result
        
        with open(file_path, 'rb') as f:
            header = f.read(10)
            
        if not header.startswith(PDF_HEADER):
            result['status'] = 'not_pdf'
            result['error'] = 'File does not have valid PDF header'
            return result
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read(50000)
                has_stream = b'stream' in content or b'endstream' in content
                has_obj = b'obj' in content and b'endobj' in content
                result['has_content'] = has_stream and has_obj
        except Exception:
            result['has_content'] = True
        
        result['valid'] = True
        result['status'] = 'valid'
        return result
        
    except Exception as e:
        result['status'] = 'corrupted'
        result['error'] = str(e)
        return result


def normalize_title(title: str) -> str:
    """Normalize a title for matching."""
    if not title:
        return ''
    normalized = title.lower()
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    normalized = re.sub(r'[\s_]+', ' ', normalized).strip()
    return normalized


def extract_title_from_filename(filename: str) -> str:
    """Extract a human-readable title from a PDF filename."""
    stem = Path(filename).stem
    if '_' in stem:
        title_part = stem.split('_', 1)[1]
    else:
        title_part = stem
    title = title_part.replace('_', ' ')
    title = re.sub(r'\s+', ' ', title).strip()
    return title


# ==================== PDF Book Class ====================

class PDFBook:
    """Central database for PDF management."""
    
    def __init__(self):
        self.book_path = get_pdf_book_path()
        self.pdf_cache_dir = get_pdf_cache_dir()
        self._data = None
    
    def _create_empty_book(self) -> dict:
        """Create an empty PDF book structure."""
        return {
            'version': 2,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'stats': {
                'total_pdfs': 0,
                'valid_pdfs': 0,
                'invalid_pdfs': 0,
                'duplicates': 0,
                'studied_pdfs': 0,
                'total_size_mb': 0.0
            },
            'pdfs': {},
            'hash_index': {},
            'title_index': {}
        }
    
    def load(self, force: bool = False) -> dict:
        """Load the PDF book from disk."""
        now = time.time()
        
        if not force and _pdf_book_cache['book'] is not None:
            if (now - _pdf_book_cache['loaded_at']) < _pdf_book_cache['cache_ttl']:
                self._data = _pdf_book_cache['book']
                return self._data
        
        if self.book_path.exists():
            try:
                with open(self.book_path, 'r') as f:
                    self._data = json.load(f)
                if self._data.get('version', 1) < 2:
                    self._migrate_v1_to_v2()
            except Exception as e:
                print(f"[PDF Book] Error loading: {e}, creating new")
                self._data = self._create_empty_book()
        else:
            self._data = self._create_empty_book()
        
        _pdf_book_cache['book'] = self._data
        _pdf_book_cache['loaded_at'] = now
        
        return self._data
    
    def _ensure_loaded(self):
        """Ensure _data is loaded, but don't overwrite if already populated."""
        if self._data is None:
            self.load()
        return self._data

    def save(self):
        """Save the PDF book to disk."""
        if self._data is None:
            return
        
        self._data['updated_at'] = datetime.now().isoformat()
        self._update_stats()
        
        self.book_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.book_path, 'w') as f:
            json.dump(self._data, f, indent=2)
        
        _pdf_book_cache['book'] = self._data
        _pdf_book_cache['loaded_at'] = time.time()
    
    def _update_stats(self):
        """Recalculate statistics."""
        pdfs = self._data.get('pdfs', {})
        
        total = len(pdfs)
        valid = sum(1 for p in pdfs.values() if p.get('status') == 'valid')
        invalid = sum(1 for p in pdfs.values() if p.get('status') != 'valid')
        duplicates = sum(1 for p in pdfs.values() if p.get('is_duplicate'))
        studied = sum(1 for p in pdfs.values() 
                     if p.get('study_info', {}).get('studied'))
        total_size = sum(p.get('size_mb', 0) for p in pdfs.values())
        
        self._data['stats'] = {
            'total_pdfs': total,
            'valid_pdfs': valid,
            'invalid_pdfs': invalid,
            'duplicates': duplicates,
            'studied_pdfs': studied,
            'total_size_mb': round(total_size, 2)
        }
    
    def _migrate_v1_to_v2(self):
        """Migrate from v1 (pdf_stock) format to v2 format."""
        old_data = self._data
        new_data = self._create_empty_book()
        
        for filename, info in old_data.get('pdfs', {}).items():
            new_data['pdfs'][filename] = {
                'path': info.get('path', ''),
                'size_mb': info.get('size_mb', 0),
                'file_hash': '',
                'title': info.get('original_title', ''),
                'normalized_title': info.get('normalized_title', ''),
                'status': 'valid',
                'is_duplicate': False,
                'duplicate_of': None,
                'metadata': {
                    'authors': '',
                    'source': '',
                    'url': '',
                    'abstract': '',
                    'added_at': old_data.get('updated_at', '')
                },
                'study_info': {
                    'studied': False,
                    'studied_by': [],
                    'study_count': 0,
                    'last_studied': '',
                    'topics': []
                }
            }
        
        new_data['title_index'] = old_data.get('title_index', {})
        new_data['version'] = 2
        
        self._data = new_data
        self.save()
        print("[PDF Book] Migrated from v1 to v2 format")
    
    def rebuild(self, validate: bool = True, detect_duplicates: bool = True) -> dict:
        """Rebuild the PDF book by scanning the cache directory."""
        print("[PDF Book] Starting rebuild...")
        
        self._data = self._create_empty_book()
        
        if not self.pdf_cache_dir.exists():
            self.save()
            return {'total': 0, 'valid': 0, 'invalid': 0, 'duplicates': 0}
        
        stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'duplicates': 0,
            'scanned': 0
        }
        
        hash_to_filename = {}
        
        for pdf_file in self.pdf_cache_dir.glob('*.pdf'):
            stats['scanned'] += 1
            filename = pdf_file.name
            
            try:
                size_mb = round(pdf_file.stat().st_size / (1024 * 1024), 2)
            except Exception:
                size_mb = 0
            
            title = extract_title_from_filename(filename)
            normalized = normalize_title(title)
            
            if validate:
                validation = validate_pdf(pdf_file)
                status = validation['status']
                if not validation['valid']:
                    stats['invalid'] += 1
                else:
                    stats['valid'] += 1
            else:
                status = 'valid'
                stats['valid'] += 1
            
            file_hash = ''
            is_duplicate = False
            duplicate_of = None
            
            if detect_duplicates and status == 'valid':
                file_hash = compute_file_hash(pdf_file)
                if file_hash:
                    if file_hash in hash_to_filename:
                        is_duplicate = True
                        duplicate_of = hash_to_filename[file_hash]
                        stats['duplicates'] += 1
                    else:
                        hash_to_filename[file_hash] = filename
                        self._data['hash_index'][file_hash] = filename
            
            self._data['pdfs'][filename] = {
                'path': str(pdf_file),
                'size_mb': size_mb,
                'file_hash': file_hash,
                'title': title,
                'normalized_title': normalized,
                'status': status,
                'is_duplicate': is_duplicate,
                'duplicate_of': duplicate_of,
                'metadata': {
                    'authors': '',
                    'source': '',
                    'url': '',
                    'abstract': '',
                    'added_at': datetime.now().isoformat()
                },
                'study_info': {
                    'studied': False,
                    'studied_by': [],
                    'study_count': 0,
                    'last_studied': '',
                    'topics': []
                }
            }
            
            if normalized and not is_duplicate:
                self._data['title_index'][normalized] = filename
            
            stats['total'] += 1
        
        self._load_study_info()
        self.save()
        
        print(f"[PDF Book] Rebuild complete: {stats['total']} PDFs, "
              f"{stats['valid']} valid, {stats['invalid']} invalid, "
              f"{stats['duplicates']} duplicates")
        
        return stats
    
    def _load_study_info(self):
        """Load study information from all actors' cognitive loops."""
        array_dir = get_array_dir()
        actors_dir = array_dir / 'actors'
        
        if not actors_dir.exists():
            return
        
        for actor_dir in actors_dir.iterdir():
            if not actor_dir.is_dir() or actor_dir.name.startswith('.'):
                continue
            
            actor_name = actor_dir.name
            loops_dir = array_dir / 'data' / actor_name / 'cognitive_loops'
            
            if not loops_dir.exists():
                continue
            
            for loop_file in loops_dir.glob('loop_*.json'):
                try:
                    with open(loop_file, 'r') as f:
                        data = json.load(f)
                    
                    state = data.get('state', {})
                    loop_topic = state.get('topic', 'Unknown')
                    
                    for paper_entry in state.get('studied_papers', []):
                        paper = paper_entry.get('paper', {}) or {}
                        title = paper.get('title', '')
                        
                        if not title:
                            continue
                        
                        pdf_entry = self.find_by_title(title)
                        if not pdf_entry:
                            continue
                        
                        filename = pdf_entry['filename']
                        study_info = self._data['pdfs'][filename]['study_info']
                        
                        study_info['studied'] = True
                        if actor_name not in study_info['studied_by']:
                            study_info['studied_by'].append(actor_name)
                        study_info['study_count'] += 1
                        
                        studied_at = paper_entry.get('studied_at', '')
                        if studied_at > study_info['last_studied']:
                            study_info['last_studied'] = studied_at
                        
                        if loop_topic and loop_topic not in study_info['topics']:
                            study_info['topics'].append(loop_topic)
                            
                except Exception:
                    continue
    
    def find_by_title(self, title: str) -> Optional[dict]:
        """Find a PDF by title (exact or fuzzy match)."""
        self._ensure_loaded()
        
        normalized = normalize_title(title)
        if not normalized:
            return None
        
        if normalized in self._data['title_index']:
            filename = self._data['title_index'][normalized]
            if filename in self._data['pdfs']:
                entry = self._data['pdfs'][filename].copy()
                entry['filename'] = filename
                return entry
        
        search_words = {w for w in normalized.split() if len(w) > 3}
        if len(search_words) < 2:
            return None
        
        best_match = None
        best_score = 0
        
        for filename, pdf_info in self._data['pdfs'].items():
            if pdf_info.get('is_duplicate'):
                continue
            
            pdf_normalized = pdf_info.get('normalized_title', '')
            pdf_words = {w for w in pdf_normalized.split() if len(w) > 3}
            
            if not pdf_words:
                continue
            
            overlap = search_words & pdf_words
            if len(overlap) >= 2:
                score = len(overlap) / max(len(search_words), len(pdf_words))
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_match = (filename, pdf_info)
        
        if best_match:
            filename, pdf_info = best_match
            entry = pdf_info.copy()
            entry['filename'] = filename
            entry['match_score'] = best_score
            return entry
        
        return None
    
    def find_by_hash(self, file_hash: str) -> Optional[dict]:
        """Find a PDF by its file hash."""
        self._ensure_loaded()
        
        if file_hash in self._data['hash_index']:
            filename = self._data['hash_index'][file_hash]
            if filename in self._data['pdfs']:
                entry = self._data['pdfs'][filename].copy()
                entry['filename'] = filename
                return entry
        
        return None
    
    def get_pdf(self, filename: str) -> Optional[dict]:
        """Get a PDF entry by filename."""
        self.load()
        
        if filename in self._data['pdfs']:
            entry = self._data['pdfs'][filename].copy()
            entry['filename'] = filename
            return entry
        
        return None
    
    def update_metadata(self, filename: str, metadata: dict) -> bool:
        """Update metadata for a PDF."""
        self._ensure_loaded()
        
        if filename not in self._data['pdfs']:
            return False
        
        self._data['pdfs'][filename]['metadata'].update(metadata)
        self.save()
        return True
    
    def mark_studied(self, filename: str, actor_name: str, topic: str = '') -> bool:
        """Mark a PDF as studied by an actor."""
        self._ensure_loaded()
        
        if filename not in self._data['pdfs']:
            return False
        
        study_info = self._data['pdfs'][filename]['study_info']
        study_info['studied'] = True
        if actor_name not in study_info['studied_by']:
            study_info['studied_by'].append(actor_name)
        study_info['study_count'] += 1
        study_info['last_studied'] = datetime.now().isoformat()
        if topic and topic not in study_info['topics']:
            study_info['topics'].append(topic)
        
        self.save()
        return True
    
    def get_all(self, include_invalid: bool = False, 
                include_duplicates: bool = False) -> List[dict]:
        """Get all PDFs as a list."""
        self._ensure_loaded()
        
        result = []
        for filename, pdf_info in self._data['pdfs'].items():
            if not include_invalid and pdf_info.get('status') != 'valid':
                continue
            if not include_duplicates and pdf_info.get('is_duplicate'):
                continue
            
            entry = pdf_info.copy()
            entry['filename'] = filename
            result.append(entry)
        
        result.sort(key=lambda x: x.get('title', '').lower())
        return result
    
    def get_stats(self) -> dict:
        """Get statistics about the PDF book."""
        self._ensure_loaded()
        return self._data['stats'].copy()
    
    def get_invalid_pdfs(self) -> List[dict]:
        """Get list of invalid/corrupted PDFs."""
        self._ensure_loaded()
        
        result = []
        for filename, pdf_info in self._data['pdfs'].items():
            if pdf_info.get('status') != 'valid':
                entry = pdf_info.copy()
                entry['filename'] = filename
                result.append(entry)
        
        return result
    
    def get_duplicates(self) -> List[dict]:
        """Get list of duplicate PDFs."""
        self._ensure_loaded()
        
        result = []
        for filename, pdf_info in self._data['pdfs'].items():
            if pdf_info.get('is_duplicate'):
                entry = pdf_info.copy()
                entry['filename'] = filename
                result.append(entry)
        
        return result
    
    def remove_invalid(self, dry_run: bool = True) -> List[str]:
        """Remove invalid PDFs from the cache."""
        self._ensure_loaded()
        
        to_remove = []
        for filename, pdf_info in self._data['pdfs'].items():
            if pdf_info.get('status') != 'valid':
                to_remove.append(filename)
        
        if not dry_run:
            for filename in to_remove:
                try:
                    pdf_path = self.pdf_cache_dir / filename
                    if pdf_path.exists():
                        pdf_path.unlink()
                    del self._data['pdfs'][filename]
                except Exception as e:
                    print(f"[PDF Book] Error removing {filename}: {e}")
            self.save()
        
        return to_remove
    
    def remove_duplicates(self, dry_run: bool = True) -> List[str]:
        """Remove duplicate PDFs from the cache."""
        self._ensure_loaded()
        
        to_remove = []
        for filename, pdf_info in self._data['pdfs'].items():
            if pdf_info.get('is_duplicate'):
                to_remove.append(filename)
        
        if not dry_run:
            for filename in to_remove:
                try:
                    pdf_path = self.pdf_cache_dir / filename
                    if pdf_path.exists():
                        pdf_path.unlink()
                    del self._data['pdfs'][filename]
                except Exception as e:
                    print(f"[PDF Book] Error removing {filename}: {e}")
            self.save()
        
        return to_remove


# ==================== Module-level Functions ====================

_pdf_book = None


def get_pdf_book() -> PDFBook:
    """Get the global PDFBook instance."""
    global _pdf_book
    if _pdf_book is None:
        _pdf_book = PDFBook()
    return _pdf_book


def rebuild_pdf_book(validate: bool = True, detect_duplicates: bool = True) -> dict:
    """Rebuild the PDF book index."""
    return get_pdf_book().rebuild(validate, detect_duplicates)


def get_pdf_book_stats() -> dict:
    """Get PDF book statistics."""
    return get_pdf_book().get_stats()


def find_pdf(title: str) -> Optional[dict]:
    """Find a PDF by title."""
    return get_pdf_book().find_by_title(title)


def get_all_pdfs(include_invalid: bool = False, 
                 include_duplicates: bool = False) -> List[dict]:
    """Get all valid PDFs."""
    return get_pdf_book().get_all(include_invalid, include_duplicates)


def clear_pdf_book_cache():
    """Clear the in-memory cache."""
    _pdf_book_cache['book'] = None
    _pdf_book_cache['loaded_at'] = 0
