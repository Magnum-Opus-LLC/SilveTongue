"""
Scholar Search Module

Google Scholar search and analysis tools for academic research.

Components:
- ScholarSearcher: Main interface for searching Google Scholar
- ScholarPaper: Data class for individual papers
- search_scholar: Async function for fetching search results
- survey: Get search results with word analysis and citation histogram
- SearchResult: Data class for raw search results
- SmartPaperFetcher: Intelligent PDF extraction and download system
- PDFExtractor: LLM-powered PDF URL extraction
- BrowserDownloader: Browser automation for complex downloads
"""

from src.scholar_search.parse_html import SearchResult, parse_html
from src.scholar_search.search_functions import search_scholar
from src.scholar_search.survey import survey
from src.scholar_search.main_search import search_api, survey_api
from src.scholar_search.analyze_words import analyze_words
from src.scholar_search.generate_histogram import generate_histogram
from src.scholar_search.scholar_caller import search_google_scholar
from src.scholar_search.searcher import (
    ScholarSearcher,
    ScholarPaper,
    scholar_searcher,
    quick_search,
    quick_survey
)
from src.scholar_search.smart_paper_fetcher import SmartPaperFetcher
from src.scholar_search.pdf_extractor import PDFExtractor
from src.scholar_search.browser_downloader import BrowserDownloader, PLAYWRIGHT_AVAILABLE

__all__ = [
    # Main interface
    'ScholarSearcher',
    'ScholarPaper',
    'scholar_searcher',
    'quick_search',
    'quick_survey',
    # Smart PDF downloading
    'SmartPaperFetcher',
    'PDFExtractor',
    'BrowserDownloader',
    'PLAYWRIGHT_AVAILABLE',
    # Low-level functions
    'SearchResult',
    'parse_html',
    'search_scholar',
    'survey',
    'search_api',
    'survey_api',
    'analyze_words',
    'generate_histogram',
    'search_google_scholar'
]
