import requests
from bs4 import BeautifulSoup
import re
import json
import logging
import time
import random
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# Rate limiting settings - can be faster with paid proxy
MIN_DELAY = 2.0  # Minimum seconds between requests
MAX_DELAY = 4.0  # Maximum seconds between requests
MAX_RETRIES = 3  # Maximum retry attempts
RETRY_DELAY = 5.0  # Seconds to wait after a 429 error before retry

# User agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

# Oxylabs proxy configuration
# Credentials from environment variables: PROXY_USER_NAME, PROXY_PASS
OXYLABS_COUNTRIES = ['US', 'GB', 'DE', 'FR', 'CA', 'AU', 'JP']  # Rotate through countries


def get_oxylabs_proxy(country=None):
    """
    Get Oxylabs proxy URL with credentials from environment variables.
    
    Args:
        country: Country code (e.g., 'US', 'GB'). If None, picks random.
    
    Returns:
        Proxy URL string or None if credentials not set.
    """
    username = os.environ.get('PROXY_USER_NAME')
    password = os.environ.get('PROXY_PASS')
    
    if not username or not password:
        logging.debug("Oxylabs credentials not found in environment variables")
        return None
    
    if country is None:
        country = random.choice(OXYLABS_COUNTRIES)
    
    proxy_url = f"http://customer-{username}-cc-{country}:{password}@pr.oxylabs.io:7777"
    return proxy_url


def test_oxylabs_connection():
    """Test if Oxylabs proxy is working."""
    proxy = get_oxylabs_proxy('US')
    if not proxy:
        logging.warning("Oxylabs credentials not configured")
        return False
    
    try:
        proxies = {"http": proxy, "https": proxy}
        response = requests.get("https://ip.oxylabs.io/location", proxies=proxies, timeout=15)
        if response.status_code == 200:
            logging.info(f"Oxylabs proxy working. Location: {response.text.strip()}")
            return True
    except Exception as e:
        logging.error(f"Oxylabs proxy test failed: {e}")
    return False


# Global proxy manager (for fallback to free proxies if needed)
class ProxyManager:
    """Manages proxy rotation for avoiding rate limits."""
    
    def __init__(self):
        self._proxies = []
        self._current_index = 0
        self._failed_proxies = set()
        self._last_fetch = 0
        self._fetch_interval = 300  # Refresh proxy list every 5 minutes
        self._use_oxylabs = True  # Prefer Oxylabs
        self._oxylabs_country_index = 0
    
    def get_oxylabs_proxy(self):
        """Get Oxylabs proxy, rotating through countries."""
        country = OXYLABS_COUNTRIES[self._oxylabs_country_index % len(OXYLABS_COUNTRIES)]
        self._oxylabs_country_index += 1
        return get_oxylabs_proxy(country)
    
    def fetch_proxies(self):
        """Fetch fresh proxy list from multiple sources (fallback only)."""
        all_proxies = []
        
        sources = [
            "https://spys.me/proxy.txt",
        ]
        
        for source in sources:
            try:
                response = requests.get(source, timeout=10)
                if response.status_code == 200:
                    for line in response.text.splitlines():
                        line = line.strip()
                        if line and ":" in line and not line.startswith("#"):
                            parts = line.split()
                            ip_port = parts[0] if parts else line
                            if re.match(r'\d+\.\d+\.\d+\.\d+:\d+', ip_port):
                                proxy = f"http://{ip_port}"
                                if proxy not in self._failed_proxies:
                                    all_proxies.append(proxy)
            except Exception as e:
                logging.debug(f"Failed to fetch proxies from {source}: {e}")
        
        self._proxies = list(set(all_proxies))
        random.shuffle(self._proxies)
        self._current_index = 0
        self._last_fetch = time.time()
        
        logging.info(f"Loaded {len(self._proxies)} fallback proxies")
        return len(self._proxies)
    
    def get_proxy(self):
        """Get next proxy - prefers Oxylabs, falls back to free proxies."""
        # First try Oxylabs
        if self._use_oxylabs:
            proxy = self.get_oxylabs_proxy()
            if proxy:
                return proxy
        
        # Fallback to free proxies
        # Refresh if needed
        if time.time() - self._last_fetch > self._fetch_interval or not self._proxies:
            self.fetch_proxies()
        
        if not self._proxies:
            return None
        
        proxy = self._proxies[self._current_index % len(self._proxies)]
        self._current_index += 1
        return proxy
    
    def mark_failed(self, proxy):
        """Mark a proxy as failed."""
        if proxy:
            self._failed_proxies.add(proxy)
            if proxy in self._proxies:
                self._proxies.remove(proxy)
    
    def reset(self):
        """Reset failed proxies list."""
        self._failed_proxies.clear()
        self._proxies.clear()
        self._current_index = 0


# Global proxy manager instance
proxy_manager = ProxyManager()


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
    Fetch the proxy list using the proxy manager.
    Returns list of proxy strings.
    """
    proxy_manager.fetch_proxies()
    return proxy_manager._proxies


def fetch_page(url, proxy=None, retry_count=0, use_oxylabs=True):
    """
    Fetch a single page from Google Scholar with retry logic and Oxylabs proxy.
    
    Args:
        url: URL to fetch
        proxy: Specific proxy to use (or None for Oxylabs)
        retry_count: Current retry attempt
        use_oxylabs: Whether to use Oxylabs proxy (recommended)
    """
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    # Get proxy - prefer Oxylabs, fall back to provided or rotation
    current_proxy = proxy
    if current_proxy is None and use_oxylabs:
        current_proxy = proxy_manager.get_proxy()  # This now returns Oxylabs first
        if current_proxy and 'oxylabs' in current_proxy:
            logging.debug(f"  Using Oxylabs proxy (country rotation)")
        elif current_proxy:
            logging.debug(f"  Using fallback proxy: {current_proxy[:30]}...")
    
    proxies = {"http": current_proxy, "https": current_proxy} if current_proxy else None
    
    try:
        response = requests.get(url, headers=headers,
                                proxies=proxies, timeout=20)
        response.raise_for_status()
        logging.info(f"Fetched URL successfully (Status: {response.status_code})")
        return response.text
        
    except requests.exceptions.HTTPError as e:
        status_code = getattr(response, 'status_code', None)
        
        if status_code == 429:
            # Rate limited - wait and try with different country
            logging.warning(f"Rate limited (429). Waiting {RETRY_DELAY}s and switching proxy country...")
            
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                new_proxy = proxy_manager.get_proxy()  # Gets new country
                return fetch_page(url, proxy=new_proxy, retry_count=retry_count + 1, 
                                use_oxylabs=use_oxylabs)
            else:
                logging.error(f"Max retries exceeded for {url}")
                return None
        else:
            logging.error(f"HTTP Error {status_code} fetching {url}: {e}")
            return None
            
    except requests.exceptions.ProxyError as e:
        logging.warning(f"Proxy error: {e}")
        
        if retry_count < MAX_RETRIES:
            time.sleep(2)
            new_proxy = proxy_manager.get_proxy()
            return fetch_page(url, proxy=new_proxy, retry_count=retry_count + 1,
                            use_oxylabs=use_oxylabs)
        return None
        
    except requests.exceptions.Timeout as e:
        logging.warning(f"Timeout: {e}")
            
        if retry_count < MAX_RETRIES:
            time.sleep(2)
            return fetch_page(url, proxy=None, retry_count=retry_count + 1,
                            use_oxylabs=use_oxylabs)
        return None
        
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        if retry_count < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return fetch_page(url, proxy=None, retry_count=retry_count + 1,
                            use_oxylabs=use_oxylabs)
        return None


def search_google_scholar(query, pages=3, proxy=None, save_json_path=None, year_from=None, use_oxylabs=True):
    """
    Perform a search on Google Scholar for the provided query.

    - query: the search query string.
    - pages: number of pages to fetch (each page returns 10 results). Default reduced to 3.
    - proxy: (optional) a proxy string (e.g., "http://user:pass@host:port").
    - save_json_path: (optional) file path where the JSON results will be saved.
    - year_from: (optional) filter papers from this year onwards.
    - use_oxylabs: (optional) whether to use Oxylabs proxy. Defaults to True.

    Returns a list of dictionaries with keys: title, link, citation, publication_info.
    
    Note: Set PROXY_USER_NAME and PROXY_PASS environment variables for Oxylabs.
    """
    encoded_query = quote(query)
    base_url = "https://scholar.google.com/scholar"
    all_results = []
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    # Build year filter
    year_param = f"&as_ylo={year_from}" if year_from else ""
    
    logging.info(f"Starting Google Scholar search: '{query}' ({pages} pages)")
    
    # Check Oxylabs configuration
    if use_oxylabs and not proxy:
        test_proxy = get_oxylabs_proxy()
        if test_proxy:
            logging.info("Using Oxylabs proxy with country rotation")
        else:
            logging.warning("Oxylabs credentials not found. Set PROXY_USER_NAME and PROXY_PASS env vars.")
            logging.warning("Falling back to direct connection (may get rate limited)")

    # Fetch pages SEQUENTIALLY with delays to avoid rate limiting
    for page in range(pages):
        start = page * 10
        url = f"{base_url}?start={start}&q={encoded_query}{year_param}&hl=en&as_sdt=0,5"
        
        logging.info(f"Fetching page {page + 1}/{pages}...")
        
        html = fetch_page(url, proxy, use_oxylabs=use_oxylabs)
        
        if html:
            try:
                results = parse_html(html)
                all_results.extend(results)
                logging.info(f"  Found {len(results)} results on page {page + 1}")
                consecutive_failures = 0  # Reset on success
            except Exception as e:
                logging.error(f"Error parsing HTML from {url}: {e}")
        else:
            logging.warning(f"  Failed to fetch page {page + 1}")
            consecutive_failures += 1
            
            # If we get rate limited repeatedly, back off longer
            if consecutive_failures >= max_consecutive_failures:
                logging.warning(f"  Too many failures. Stopping search early.")
                break
            
            if page < pages - 1:
                # Exponential backoff on failures
                backoff_time = RETRY_DELAY * (consecutive_failures + 1)
                logging.info(f"  Backing off for {backoff_time:.1f}s before next request...")
                time.sleep(backoff_time)
        
        # Add 5-second delay between pages to avoid rate limiting
        if page < pages - 1 and html:
            delay = random.uniform(MIN_DELAY, MAX_DELAY)
            logging.info(f"  Waiting {delay:.1f}s before next page...")
            time.sleep(delay)

    # Sort the results by citation count (descending)
    all_results.sort(key=lambda x: int(
        x["citation"]) if x["citation"].isdigit() else 0, reverse=True)
    
    logging.info(f"Search complete. Total results: {len(all_results)}")

    # Save results as JSON if a file path is provided
    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logging.info(f"Results saved to {save_json_path}")

    return all_results


def search_google_scholar_fast(query, pages=2, proxy=None, save_json_path=None):
    """
    Fast version using concurrent requests - USE WITH CAUTION.
    More likely to trigger rate limiting.
    """
    encoded_query = quote(query)
    base_url = "https://scholar.google.com/scholar"
    all_results = []
    urls = []

    for page in range(pages):
        start = page * 10
        url = f"{base_url}?start={start}&q={encoded_query}&hl=en&as_sdt=0,5"
        urls.append(url)

    # Fetch with limited concurrency
    with ThreadPoolExecutor(max_workers=2) as executor:
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
            # Small delay between processing
            time.sleep(1)

    all_results.sort(key=lambda x: int(
        x["citation"]) if x["citation"].isdigit() else 0, reverse=True)

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

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
