"""
Scholar Searcher - Unified interface for Google Scholar searches

Provides a high-level interface for Lingkui to search academic literature,
analyze results, and integrate with the learning system.

NOTE: Uses direct HTTP requests (no proxy required) to avoid rate limiting issues.
For heavy usage, consider adding proxy support or using the async functions with valid proxy.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import re

from src.scholar_search.scholar_caller import search_google_scholar, parse_html
from src.scholar_search.analyze_words import analyze_words


@dataclass
class ScholarPaper:
    """Represents a paper from Google Scholar search."""
    title: str
    link: str
    citation_count: int
    publication_info: str
    authors: str = ""
    year: str = ""
    source: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScholarPaper':
        """Create ScholarPaper from dictionary (scholar_caller format)."""
        pub_info = data.get('publication_info', '')
        authors = ""
        year = ""
        source = ""
        
        if pub_info:
            parts = pub_info.split(" - ")
            if len(parts) >= 1:
                authors = parts[0].strip()
            if len(parts) >= 2:
                # Try to extract year from second part
                year_match = re.search(r'\b(19|20)\d{2}\b', parts[1])
                if year_match:
                    year = year_match.group(0)
                source = parts[1].strip()
        
        citation = data.get('citation', '0')
        citation_count = int(citation) if str(citation).isdigit() else 0
        
        return cls(
            title=data.get('title', ''),
            link=data.get('link', ''),
            citation_count=citation_count,
            publication_info=pub_info,
            authors=authors,
            year=year,
            source=source
        )
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ScholarSearcher:
    """
    Unified interface for Google Scholar searches.
    
    Usage:
        searcher = ScholarSearcher()
        
        # Simple search
        papers = searcher.search("machine learning transformers")
        
        # Advanced search
        papers = searcher.search_advanced(
            query="neural networks",
            exact_phrase="attention mechanism",
            year_from=2020,
            year_to=2024
        )
        
        # Get survey with analysis
        survey = searcher.survey("deep learning", depth=3)
    """
    
    def __init__(self, save_results: bool = False, results_dir: str = None):
        """
        Initialize ScholarSearcher.
        
        Args:
            save_results: Whether to automatically save search results
            results_dir: Directory to save results (default: data/scholar_results)
        """
        self.save_results = save_results
        self.results_dir = Path(results_dir) if results_dir else Path(__file__).parent.parent.parent / 'data' / 'scholar_results'
        self._search_history = []
    
    def search(self, query: str, depth: int = 2, year_from: int = None, 
               year_to: int = None) -> List[ScholarPaper]:
        """
        Simple search for papers on Google Scholar.
        
        Args:
            query: Search query string
            depth: Number of pages to fetch (10 results per page)
            year_from: Filter papers from this year
            year_to: Filter papers up to this year
            
        Returns:
            List of ScholarPaper objects sorted by citation count
        """
        return self.search_advanced(
            query=query,
            depth=depth,
            year_from=year_from,
            year_to=year_to
        )
    
    def search_advanced(
        self,
        query: str = "",
        exact_phrase: str = "",
        any_words: str = "",
        exclude_words: str = "",
        where_words: str = "any",
        author: str = "",
        publication: str = "",
        year_from: int = None,
        year_to: int = None,
        depth: int = 2,
        include_patents: bool = True,
        include_citations: bool = True
    ) -> List[ScholarPaper]:
        """
        Advanced search with Google Scholar's full query options.
        
        Args:
            query: Words to search for (as_q)
            exact_phrase: Exact phrase to match (as_epq)
            any_words: Any of these words (as_oq)
            exclude_words: Exclude these words (as_eq)
            where_words: Where to search - 'any' or 'title' (as_occt)
            author: Filter by author (as_sauthors)
            publication: Filter by publication (as_publication)
            year_from: Start year (as_ylo)
            year_to: End year (as_yhi)
            depth: Number of pages to fetch
            include_patents: Include patents in results
            include_citations: Include citations in results
            
        Returns:
            List of ScholarPaper objects
        """
        # Build search query string with advanced options
        search_query = query
        
        if exact_phrase:
            search_query += f' "{exact_phrase}"'
        if any_words:
            search_query += f' ({" OR ".join(any_words.split())})'
        if exclude_words:
            for word in exclude_words.split():
                search_query += f' -{word}'
        if author:
            search_query += f' author:{author}'
        
        # Use synchronous search (no proxy needed)
        results = search_google_scholar(
            query=search_query.strip(),
            pages=depth
        )
        
        # Filter by year if specified
        papers = [ScholarPaper.from_dict(r) for r in results]
        
        if year_from or year_to:
            filtered = []
            for p in papers:
                if p.year and p.year.isdigit():
                    year_int = int(p.year)
                    if year_from and year_int < year_from:
                        continue
                    if year_to and year_int > year_to:
                        continue
                filtered.append(p)
            papers = filtered
        
        # Record search
        self._record_search(query or exact_phrase or any_words, papers)
        
        # Save if configured
        if self.save_results:
            self._save_results(query or "search", papers)
        
        return papers
    
    def get_survey(
        self,
        query: str,
        depth: int = 2,
        year_from: int = None,
        year_to: int = None
    ) -> Dict[str, Any]:
        """
        Get a comprehensive survey of search results including:
        - Top papers
        - Word frequency analysis
        - Citation histogram
        
        Args:
            query: Search query
            depth: Number of pages to fetch
            year_from: Start year filter
            year_to: End year filter
            
        Returns:
            Dict with 'top_papers', 'word_analysis', 'citation_histogram'
        """
        # Use synchronous search
        papers = self.search(query, depth=depth, year_from=year_from, year_to=year_to)
        
        # Word analysis on titles
        titles = [p.title for p in papers]
        word_freq = analyze_words(titles)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Citation histogram
        citations = [p.citation_count for p in papers]
        histogram = self._generate_histogram(citations)
        
        return {
            'top_papers': papers[:15],
            'word_analysis': sorted_words,
            'citation_histogram': histogram,
            'query': query,
            'searched_at': datetime.now().isoformat()
        }
    
    def _generate_histogram(self, citations: List[int]) -> Dict[str, float]:
        """Generate citation histogram."""
        if not citations:
            return {}
        
        import math
        min_val = min(citations)
        max_val = max(citations)
        n = len(citations)
        
        num_bins = max(round(math.sqrt(n)), 5)
        bin_size = 1 if max_val == min_val else math.ceil((max_val - min_val) / num_bins)
        
        bins = {}
        for citation in citations:
            bin_index = (citation - min_val) // bin_size if bin_size > 0 else 0
            bin_start = bin_index * bin_size + min_val
            bin_end = (bin_index + 1) * bin_size + min_val - 1
            bin_range = f"{bin_start}-{bin_end}"
            bins[bin_range] = bins.get(bin_range, 0) + 1
        
        total = len(citations)
        return {bin_range: (count / total) * 100 for bin_range, count in bins.items()}
    
    def analyze_titles(self, papers: List[ScholarPaper], top_n: int = 20) -> List[tuple]:
        """
        Analyze word frequency in paper titles.
        
        Args:
            papers: List of papers to analyze
            top_n: Number of top words to return
            
        Returns:
            List of (word, percentage) tuples
        """
        titles = [p.title for p in papers]
        word_freq = analyze_words(titles)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
    
    def filter_papers(
        self,
        papers: List[ScholarPaper],
        min_citations: int = None,
        max_citations: int = None,
        year_from: int = None,
        year_to: int = None,
        title_contains: str = None
    ) -> List[ScholarPaper]:
        """
        Filter a list of papers by various criteria.
        
        Args:
            papers: Papers to filter
            min_citations: Minimum citation count
            max_citations: Maximum citation count
            year_from: From year
            year_to: To year
            title_contains: Title must contain this string
            
        Returns:
            Filtered list of papers
        """
        result = papers
        
        if min_citations is not None:
            result = [p for p in result if p.citation_count >= min_citations]
        
        if max_citations is not None:
            result = [p for p in result if p.citation_count <= max_citations]
        
        if year_from is not None:
            result = [p for p in result if p.year and p.year.isdigit() and int(p.year) >= year_from]
        
        if year_to is not None:
            result = [p for p in result if p.year and p.year.isdigit() and int(p.year) <= year_to]
        
        if title_contains:
            title_lower = title_contains.lower()
            result = [p for p in result if title_lower in p.title.lower()]
        
        return result
    
    def get_highly_cited(self, papers: List[ScholarPaper], top_n: int = 10) -> List[ScholarPaper]:
        """Get the top N most cited papers."""
        sorted_papers = sorted(papers, key=lambda p: p.citation_count, reverse=True)
        return sorted_papers[:top_n]
    
    def format_as_references(self, papers: List[ScholarPaper]) -> str:
        """Format papers as a reference list."""
        lines = []
        for i, paper in enumerate(papers, 1):
            line = f"{i}. {paper.authors or 'Unknown'}"
            if paper.year:
                line += f" ({paper.year})"
            line += f". {paper.title}."
            if paper.source:
                line += f" {paper.source}."
            line += f" [Cited: {paper.citation_count}]"
            lines.append(line)
        return "\n".join(lines)
    
    def _record_search(self, query: str, papers: List[ScholarPaper]):
        """Record search in history."""
        self._search_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'result_count': len(papers),
            'top_cited': papers[0].citation_count if papers else 0
        })
    
    def _save_results(self, query: str, papers: List[ScholarPaper]):
        """Save search results to file."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
        filename = f"{safe_query}_{timestamp}.json"
        
        data = {
            'query': query,
            'searched_at': datetime.now().isoformat(),
            'result_count': len(papers),
            'papers': [p.to_dict() for p in papers]
        }
        
        with open(self.results_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_search_history(self) -> List[Dict]:
        """Get search history from this session."""
        return self._search_history.copy()


# Global instance for easy access
scholar_searcher = ScholarSearcher()


def quick_search(query: str, depth: int = 2) -> List[ScholarPaper]:
    """Quick helper for simple searches."""
    return scholar_searcher.search(query, depth=depth)


def quick_survey(query: str) -> Dict[str, Any]:
    """Quick helper for survey searches."""
    return scholar_searcher.get_survey(query)
