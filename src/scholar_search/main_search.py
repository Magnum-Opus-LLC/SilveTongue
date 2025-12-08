# api_functions.py
import asyncio
from typing import List, Dict, Any
from src.scholar_search.search_functions import search_scholar
from src.scholar_search.survey import survey


async def search_api(
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
) -> List[dict]:
    """
    Returns a list of search results from Google Scholar.
    Each result is represented as a dictionary.
    """
    # Run the asynchronous search_scholar function in an event loop
    results = await search_scholar(
            depth, as_q, as_epq, as_oq, as_eq, as_occt,
            as_sauthors, as_publication, as_ylo, as_yhi, as_sdt, as_rr, as_vis
        )
    # Convert each SearchResult dataclass to a dictionary
    return [result.__dict__ for result in results]


def survey_api(
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
    """
    Returns a survey summary which includes:
      - Top search results (as list of dictionaries)
      - Word analysis (top words and their percentage)
      - Citation histogram (bins and percentages)
    """
    return survey(
        depth, as_q, as_epq, as_oq, as_eq, as_occt,
        as_sauthors, as_publication, as_ylo, as_yhi, as_sdt, as_rr, as_vis
    )
