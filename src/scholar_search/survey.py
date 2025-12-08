# survey.py
import asyncio
from typing import Dict, Any
from src.scholar_search.search_functions import search_scholar
from src.scholar_search.analyze_words import analyze_words
from src.scholar_search.generate_histogram import generate_histogram
from src.scholar_search.parse_html import SearchResult


def survey(
    depth: int = 2,
    as_q: str = "",
    as_epq: str = "",
    as_oq: str = "",
    as_eq: str = "",
    as_occt: str = "any",
    as_sauthors: str = "",
    as_publication: str = "",
    as_ylo: int = None,
    as_yhi: int = None,
    as_sdt: int = 7,
    as_rr: int = None,
    as_vis: int = 1
) -> Dict[str, Any]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    flat_results = loop.run_until_complete(
        search_scholar(
            depth, as_q, as_epq, as_oq, as_eq, as_occt,
            as_sauthors, as_publication, as_ylo, as_yhi, as_sdt, as_rr, as_vis
        )
    )
    loop.close()
    # Top 15 search results
    top_results = flat_results[:15]
    titles = [res.title for res in top_results]
    word_analysis_dict = analyze_words(titles)
    citations = [int(res.citation) if res.citation.isdigit()
                 else 0 for res in flat_results]
    citation_hist = generate_histogram(citations)
    # Sort word analysis by percentage descending and take top 20
    sorted_words = sorted(word_analysis_dict.items(),
                          key=lambda x: x[1], reverse=True)[:20]
    return {
        "top_results": [res.__dict__ for res in top_results],
        "word_analysis": sorted_words,
        "citation_histogram": citation_hist
    }
