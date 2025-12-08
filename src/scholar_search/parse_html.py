# parse_html.py
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List


@dataclass
class SearchResult:
    title: str
    link: str
    citation: str
    publication_info: str


def parse_html(html_content: str) -> List[SearchResult]:
    soup = BeautifulSoup(html_content, "html.parser")
    results = []
    # Each search result container
    for item in soup.select(".gs_ri"):
        title_elem = item.select_one(".gs_rt a")
        title = title_elem.get_text(strip=True) if title_elem else ""
        link = title_elem.get("href", "") if title_elem else ""
        citation_elem = item.select_one(".gs_fl a:nth-of-type(3)")
        if citation_elem:
            citation_text = citation_elem.get_text(strip=True)
            match = re.search(r"Cited by (\d+)", citation_text)
            citation = match.group(1) if match else "0"
        else:
            citation = "0"
        pub_elem = item.select_one(".gs_a")
        publication_info = pub_elem.get_text(strip=True) if pub_elem else ""
        results.append(SearchResult(title=title, link=link,
                       citation=citation, publication_info=publication_info))
    return results
