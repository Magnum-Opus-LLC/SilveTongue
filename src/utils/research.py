import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

class ResearchManager:
    def __init__(self):
        self.ddgs = DDGS()

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
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit content length to avoid token limits
            return text[:5000] 
            
        except Exception as e:
            return f"Error scraping {url}: {e}"

    def conduct_research(self, query):
        """
        High-level method to search and summarize findings.
        """
        print(f"Researching: {query}")
        search_results = self.search(query)
        
        findings = []
        for result in search_results[:3]: # Process top 3
            title = result.get('title')
            link = result.get('href')
            snippet = result.get('body')
            
            findings.append(f"Source: {title}\nURL: {link}\nSnippet: {snippet}\n")
            
        return "\n".join(findings)

# Global instance
researcher = ResearchManager()
