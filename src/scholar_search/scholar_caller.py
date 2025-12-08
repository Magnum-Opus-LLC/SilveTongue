import requests
from bs4 import BeautifulSoup
import re
import json
import logging
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")


def parse_html(html_content):
    """
    Parse the HTML content returned by Google Scholar and extract search results.
    Each result contains a title, link, citation count, and publication info.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    results = []

    # Each search result is contained within an element with class "gs_ri"
    for result in soup.select(".gs_ri"):
        # Extract the title and link from the anchor tag in the .gs_rt element
        title_elem = result.select_one(".gs_rt a")
        if title_elem:
            title = title_elem.get_text(strip=True)
            link = title_elem.get("href", "")
        else:
            title = ""
            link = ""

        # Look for a link with text like "Cited by 123" within the .gs_fl element
        citation = "0"
        for a in result.select(".gs_fl a"):
            text = a.get_text(strip=True)
            m = re.search(r"Cited by (\d+)", text)
            if m:
                citation = m.group(1)
                break

        # Extract publication info (typically authors, source, year) from the .gs_a element
        pub_info_elem = result.select_one(".gs_a")
        publication_info = pub_info_elem.get_text(
            strip=True) if pub_info_elem else ""

        results.append({
            "title": title,
            "link": link,
            "citation": citation,
            "publication_info": publication_info
        })

    return results


def fetch_proxy_list():
    """
    Fetch the proxy list from a public proxy provider and return a list of proxies.
    (This mimics your proxy_manager.rs functionality.)
    """
    proxy_list_url = "https://spys.me/proxy.txt"
    response = requests.get(proxy_list_url)
    text = response.text
    proxies = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            ip_port = parts[0]
            if ":" in ip_port:
                proxies.append(ip_port)
    return proxies


def fetch_page(url, proxy=None):
    """
    Fetch a single page from Google Scholar.
    If a proxy is provided, it is used for the request.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/74.0.3729.169 Safari/537.36"
        ),
        "Referer": "https://www.google.com/"
    }
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        response = requests.get(url, headers=headers,
                                proxies=proxies, timeout=10)
        response.raise_for_status()
        logging.info(f"Fetched URL: {url} (Status: {response.status_code})")
        return response.text
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None


def search_google_scholar(query, pages=11, proxy=None, save_json_path=None):
    """
    Perform a search on Google Scholar for the provided query.

    - query: the search query string.
    - pages: number of pages to fetch (each page returns 10 results).
    - proxy: (optional) a proxy string (e.g., "http://user:pass@host:port").
    - save_json_path: (optional) file path where the JSON results will be saved.

    Returns a list of dictionaries with keys: title, link, citation, publication_info.
    """
    encoded_query = quote(query)
    base_url = "https://scholar.google.com/scholar"
    all_results = []
    urls = []

    # Construct URLs for each page (Google Scholar uses the "start" parameter)
    for page in range(pages):
        # Note: Using page*10+1 mimics your Rust URL construction.
        start = page * 10 + 1
        url = f"{base_url}?start={start}&q={encoded_query}&as_ylo=2023&hl=en&as_sdt=0,34"
        urls.append(url)
        logging.info(f"Constructed URL: {url}")

    # Fetch pages concurrently using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(
            fetch_page, url, proxy): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            html = future.result()
            if html:
                try:
                    results = parse_html(html)
                    all_results.extend(results)
                except Exception as e:
                    logging.error(f"Error parsing HTML from {url}: {e}")

    # Sort the results by citation count (descending)
    all_results.sort(key=lambda x: int(
        x["citation"]) if x["citation"].isdigit() else 0, reverse=True)

    # Save results as JSON if a file path is provided
    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logging.info(f"Results saved to {save_json_path}")

    return all_results


if __name__ == "__main__":
    # Example usage: prompt the user for a query and an optional proxy.
    query = input("Enter your search query: ")
    proxy_input = input(
        "Enter proxy (e.g., http://user:pass@host:port) or leave blank for none: ").strip() or None

    # You could also use the fetch_proxy_list() function to get a list of proxies:
    # proxies = fetch_proxy_list()
    # proxy_input = proxies[0] if proxies else None

    results = search_google_scholar(
        query, proxy=proxy_input, save_json_path="results.json")
    print(json.dumps(results, indent=2))
