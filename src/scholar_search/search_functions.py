# search_functions.py
import aiohttp
import asyncio
from typing import List, Optional
from urllib.parse import quote
from src.scholar_search.parse_html import parse_html, SearchResult


async def fetch_page(session: aiohttp.ClientSession, url: str, proxy: str, headers: dict) -> List[SearchResult]:
    try:
        async with session.get(url, proxy=proxy, headers=headers, timeout=30) as response:
            text = await response.text()
            results = parse_html(text)
            return results
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []


async def search_scholar(
    depth: Optional[int] = 2,
    as_q: Optional[str] = "",
    as_epq: Optional[str] = "",
    as_oq: Optional[str] = "",
    as_eq: Optional[str] = "",
    as_occt: Optional[str] = "any",
    as_sauthors: Optional[str] = "",
    as_publication: Optional[str] = "",
    as_ylo: Optional[int] = None,
    as_yhi: Optional[int] = None,
    as_sdt: Optional[int] = 7,
    as_rr: Optional[int] = None,
    as_vis: Optional[int] = 1
) -> List[SearchResult]:
    depth = depth or 2
    as_sdt = as_sdt or 7
    as_vis = as_vis or 1
    as_occt = as_occt or "any"
    tasks = []
    base_url = "https://scholar.google.com/scholar"
    # Set the proxy URL as per your configuration
    proxy_url = "http://customer-scholar_scarpper:feb14project_Test@us-pr.oxylabs.io:10000"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
        "Referer": "https://www.google.com/"
    }
    async with aiohttp.ClientSession() as session:
        for page in range(depth):
            start = page * 10
            url = (
                f"{base_url}?as_q={quote(as_q or '')}"
                f"&as_epq={quote(as_epq or '')}"
                f"&as_oq={quote(as_oq or '')}"
                f"&as_eq={quote(as_eq or '')}"
                f"&as_occt={quote(as_occt)}"
                f"&as_sauthors={quote(as_sauthors or '')}"
                f"&as_publication={quote(as_publication or '')}"
                f"&as_ylo={as_ylo if as_ylo is not None else ''}"
                f"&as_yhi={as_yhi if as_yhi is not None else ''}"
                f"&hl=en&as_sdt={as_sdt}"
                f"&as_rr={as_rr if as_rr is not None else ''}"
                f"&as_vis={as_vis}&start={start}"
            )
            tasks.append(fetch_page(session, url, proxy_url, headers))
        pages_results = await asyncio.gather(*tasks)
    # Flatten the list of results from all pages
    flat_results = [item for sublist in pages_results for item in sublist]
    # Sort by citation count descending
    flat_results.sort(key=lambda x: int(x.citation)
                      if x.citation.isdigit() else 0, reverse=True)
    return flat_results
