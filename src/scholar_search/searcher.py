"""
Scholar Searcher - Unified interface for Google Scholar searches

Provides a high-level interface for Lingkui to search academic literature,
analyze results, and integrate with the learning system.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from src.scholar_search.parse_html import SearchResult
from src.scholar_search.search_functions import search_scholar
from src.scholar_search.survey import survey
from src.scholar_search.scholar_caller import search_google_scholar
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
    def from_search_result(cls, result: SearchResult) -> 'ScholarPaper':
        """Create ScholarPaper from SearchResult."""
        # Parse publication info to extract authors, year, source
        pub_info = result.publication_info
        authors = ""
        year = ""
        source = ""
        
        if pub_info:
            parts = pub_info.split(" - ")
            if len(parts) >= 1:
                authors = parts[0].strip()
            if len(parts) >= 2:
                # Try to extract year from second part
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', parts[1])
                if year_match:
                    year = year_match.group(0)
                source = parts[1].strip()
        
        return cls(
            title=result.title,
            link=result.link,
            citation_count=int(result.citation) if result.citation.isdigit() else 0,
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
        # Run async search
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            search_scholar(
                depth=depth,
                as_q=query,
                as_epq=exact_phrase,
                as_oq=any_words,
                as_eq=exclude_words,
                as_occt=where_words,
                as_sauthors=author,
                as_publication=publication,
                as_ylo=year_from,
                as_yhi=year_to,
                as_sdt=0 if include_patents else 7,
                as_vis=1 if include_citations else 0
            )
        )
        
        papers = [ScholarPaper.from_search_result(r) for r in results]
        
        # Record search
        self._record_search(query or exact_phrase or any_words, papers)
        
        # Save if configured
        if self.save_results:
            self._save_results(query or "search", papers)
        
        return papers
    
    def search_simple(self, query: str, pages: int = 5, year_from: int = None) -> List[ScholarPaper]:
        """
        Simple synchronous search using scholar_caller (no proxy required).
        
        Args:
            query: Search query
            pages: Number of pages (10 results each)
            year_from: Filter from year
            
        Returns:
            List of ScholarPaper objects
        """
        results = search_google_scholar(query, pages=pages)
        
        papers = []
        for r in results:
            papers.append(ScholarPaper(
                title=r['title'],
                link=r['link'],
                citation_count=int(r['citation']) if r['citation'].isdigit() else 0,
                publication_info=r['publication_info'],
                authors="",
                year="",
                source=""
            ))
        
        # Record and optionally save
        self._record_search(query, papers)
        if self.save_results:
            self._save_results(query, papers)
        
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
            Dict with 'top_results', 'word_analysis', 'citation_histogram'
        """
        result = survey(
            depth=depth,
            as_q=query,
            as_ylo=year_from,
            as_yhi=year_to
        )
        
        # Convert top results to ScholarPaper
        papers = []
        for r in result.get('top_results', []):
            papers.append(ScholarPaper(
                title=r.get('title', ''),
                link=r.get('link', ''),
                citation_count=int(r.get('citation', 0)) if str(r.get('citation', '0')).isdigit() else 0,
                publication_info=r.get('publication_info', ''),
                authors="",
                year="",
                source=""
            ))
        
        return {
            'top_papers': papers,
            'word_analysis': result.get('word_analysis', []),
            'citation_histogram': result.get('citation_histogram', {}),
            'query': query,
            'searched_at': datetime.now().isoformat()
        }
    
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
