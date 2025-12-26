import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from typing import List, Dict, Optional


class ResearchManager:
    """
    Research manager with web search and academic paper search capabilities.

    Supports:
    - DuckDuckGo web search
    - Google Scholar academic search
    - Web scraping
    """

    def __init__(self):
        self.ddgs = DDGS()
        self._scholar_searcher = None

    @property
    def scholar(self):
        """Lazy-load ScholarSearcher to avoid circular imports."""
        if self._scholar_searcher is None:
            from src.scholar_search import ScholarSearcher
            self._scholar_searcher = ScholarSearcher(save_results=True)
        return self._scholar_searcher

    def search(self, query, max_results=5):
        """
        Perform a web search using DuckDuckGo.
        """
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def search_scholar(self, query: str, depth: int = 2,
                       year_from: int = None, year_to: int = None) -> List[Dict]:
        """
        Search Google Scholar for academic papers.

        Args:
            query: Search query
            depth: Number of pages (10 results each)
            year_from: Filter papers from this year
            year_to: Filter papers to this year

        Returns:
            List of paper dictionaries with title, link, citations, etc.
        """
        try:
            papers = self.scholar.search(
                query,
                depth=depth,
                year_from=year_from,
                year_to=year_to
            )
            return [p.to_dict() for p in papers]
        except Exception as e:
            print(f"Error during scholar search: {e}")
            return []

    def search_scholar_advanced(
        self,
        query: str = "",
        exact_phrase: str = "",
        author: str = "",
        publication: str = "",
        year_from: int = None,
        year_to: int = None,
        depth: int = 2
    ) -> List[Dict]:
        """
        Advanced Google Scholar search with filters.

        Args:
            query: General search terms
            exact_phrase: Exact phrase to match
            author: Filter by author name
            publication: Filter by publication/journal
            year_from: Start year
            year_to: End year
            depth: Number of pages to fetch

        Returns:
            List of paper dictionaries
        """
        try:
            papers = self.scholar.search_advanced(
                query=query,
                exact_phrase=exact_phrase,
                author=author,
                publication=publication,
                year_from=year_from,
                year_to=year_to,
                depth=depth
            )
            return [p.to_dict() for p in papers]
        except Exception as e:
            print(f"Error during advanced scholar search: {e}")
            return []

    def get_scholar_survey(self, query: str, depth: int = 2) -> Dict:
        """
        Get a comprehensive survey of academic papers including:
        - Top papers by citations
        - Word frequency analysis
        - Citation distribution

        Args:
            query: Search query
            depth: Search depth

        Returns:
            Survey dictionary with analysis
        """
        try:
            survey = self.scholar.get_survey(query, depth=depth)
            # Convert papers to dicts
            survey['top_papers'] = [p.to_dict()
                                    for p in survey.get('top_papers', [])]
            return survey
        except Exception as e:
            print(f"Error during scholar survey: {e}")
            return {}

    def scrape_content(self, url):
        """
        Scrape the main text content from a URL.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Break into lines and remove leading/trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Limit content length to avoid token limits
            return text[:5000]

        except Exception as e:
            return f"Error scraping {url}: {e}"

    def conduct_research(self, query: str, include_scholar: bool = False) -> str:
        """
        High-level method to search and summarize findings.

        Args:
            query: Research query
            include_scholar: Also search Google Scholar for academic papers

        Returns:
            Formatted research findings
        """
        print(f"Researching: {query}")
        findings = []

        # Web search
        search_results = self.search(query)

        findings.append("=== Web Search Results ===\n")
        for result in search_results[:3]:  # Process top 3
            title = result.get('title')
            link = result.get('href')
            snippet = result.get('body')
            findings.append(
                f"Source: {title}\nURL: {link}\nSnippet: {snippet}\n")

        # Scholar search if requested
        if include_scholar:
            findings.append("\n=== Academic Papers (Google Scholar) ===\n")
            papers = self.search_scholar(query, depth=1)

            for paper in papers[:5]:  # Top 5 papers
                findings.append(
                    f"📄 {paper.get('title', 'Unknown')}\n"
                    f"   Citations: {paper.get('citation_count', 0)}\n"
                    f"   {paper.get('publication_info', '')}\n"
                    f"   Link: {paper.get('link', '')}\n"
                )

        return "\n".join(findings)

    def conduct_academic_research(self, query: str, depth: int = 2,
                                  year_from: int = None) -> str:
        """
        Focused academic research using Google Scholar.

        Args:
            query: Research query
            depth: Search depth (pages)
            year_from: Only papers from this year onwards

        Returns:
            Formatted academic research findings
        """
        print(f"Academic Research: {query}")

        survey = self.get_scholar_survey(query, depth=depth)

        findings = [f"=== Academic Research: {query} ===\n"]

        # Top papers
        findings.append("📚 Top Papers by Citations:\n")
        for i, paper in enumerate(survey.get('top_papers', [])[:10], 1):
            findings.append(
                f"{i}. [{paper.get('citation_count', 0)} citations] {paper.get('title', 'Unknown')}\n"
                f"   {paper.get('publication_info', '')}\n"
            )

        # Word analysis
        findings.append("\n📊 Key Terms in Research:\n")
        for word, percentage in survey.get('word_analysis', [])[:10]:
            findings.append(f"   • {word}: {percentage:.1f}%\n")

        # Citation histogram
        findings.append("\n📈 Citation Distribution:\n")
        for bin_range, percentage in list(survey.get('citation_histogram', {}).items())[:5]:
            findings.append(
                f"   {bin_range}: {'█' * int(percentage / 5)} {percentage:.1f}%\n")

        return "".join(findings)


# Global instance
researcher = ResearchManager()
