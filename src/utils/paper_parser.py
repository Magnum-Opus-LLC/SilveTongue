"""
Paper Parser Utility

Parses papers/articles from various formats (PDF, TXT, MD) and extracts text content.
Supports chunking for LLM processing.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Optional PDF support - graceful fallback if not installed
try:
    import fitz  # PyMuPDF
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False


class PaperParser:
    """Parse papers from various formats into structured text."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf'}
    
    def __init__(self):
        self.supported = self.SUPPORTED_EXTENSIONS.copy()
        if not HAS_PDF_SUPPORT:
            self.supported.discard('.pdf')
    
    def parse_file(self, file_path: Path) -> Dict:
        """
        Parse a single file and return structured content.
        
        Returns:
            {
                "file_path": str,
                "file_name": str,
                "file_type": str,
                "parsed_at": str,
                "content": str,
                "chunks": List[str],
                "metadata": {
                    "size_bytes": int,
                    "char_count": int,
                    "word_count": int,
                    "page_count": int (for PDFs)
                }
            }
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        
        if ext not in self.supported:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {self.supported}")
        
        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext in {'.txt', '.md'}:
            return self._parse_text(file_path)
    
    def _parse_text(self, file_path: Path) -> Dict:
        """Parse plain text or markdown files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        chunks = self._chunk_text(content)
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower(),
            "parsed_at": datetime.now().isoformat(),
            "content": content,
            "chunks": chunks,
            "metadata": {
                "size_bytes": file_path.stat().st_size,
                "char_count": len(content),
                "word_count": len(content.split()),
                "page_count": 1
            }
        }
    
    def _parse_pdf(self, file_path: Path) -> Dict:
        """Parse PDF files using PyMuPDF."""
        if not HAS_PDF_SUPPORT:
            raise ImportError("PDF support requires PyMuPDF. Install with: pip install pymupdf")
        
        doc = fitz.open(file_path)
        pages = []
        full_text = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages.append({
                "page_num": page_num + 1,
                "text": text
            })
            full_text.append(text)
        
        content = "\n\n".join(full_text)
        chunks = self._chunk_text(content)
        
        doc.close()
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": ".pdf",
            "parsed_at": datetime.now().isoformat(),
            "content": content,
            "chunks": chunks,
            "pages": pages,
            "metadata": {
                "size_bytes": file_path.stat().st_size,
                "char_count": len(content),
                "word_count": len(content.split()),
                "page_count": len(pages)
            }
        }
    
    def _chunk_text(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks suitable for LLM processing.
        
        Args:
            text: Full text content
            chunk_size: Target size of each chunk in characters
            overlap: Overlap between chunks to maintain context
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for punct in ['. ', '! ', '? ', '.\n']:
                        sent_break = text.rfind(punct, start + chunk_size // 2, end)
                        if sent_break > 0:
                            end = sent_break + len(punct)
                            break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return chunks
    
    def scan_directory(self, directory: Path, file_types: List[str] = None, 
                       recursive: bool = True) -> List[Path]:
        """
        Scan a directory for parseable files.
        
        Args:
            directory: Directory to scan
            file_types: List of extensions to include (e.g., ['.pdf', '.txt'])
            recursive: Whether to scan subdirectories
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if file_types is None:
            file_types = list(self.supported)
        
        # Normalize extensions
        file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                      for ext in file_types]
        
        files = []
        
        if recursive:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    if Path(filename).suffix.lower() in file_types:
                        files.append(Path(root) / filename)
        else:
            for item in directory.iterdir():
                if item.is_file() and item.suffix.lower() in file_types:
                    files.append(item)
        
        return sorted(files)


# Global parser instance
paper_parser = PaperParser()
