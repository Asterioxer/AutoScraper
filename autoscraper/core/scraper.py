import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from autoscraper.utils.logger import info, error

def fetch_page(url, retries=3, backoff=1, timeout=10):
    """Fetch a page with retries and exponential backoff."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://google.com/",
        "Connection": "keep-alive",
    }
    for attempt in range(1, retries + 1):
        try:
            info(f"Fetching {url} (attempt {attempt})")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            error(f"Error fetching {url}: {e}")
            if attempt == retries:
                error(f"Max retries reached for {url}. Skipping.")
                return None
            wait = backoff * (2 ** (attempt - 1))
            info(f"Retrying in {wait} seconds...")
            time.sleep(wait)



def scrape(url: str, selector: str, retries=3, backoff=1, timeout=10):
    """Scrape elements matching selector from a single page with retry."""
    html = fetch_page(url, retries, backoff, timeout)
    if not html:
        return []
    soup = BeautifulSoup(html, 'lxml')
    elements = soup.select(selector)
    return [el.get_text(strip=True) for el in elements]

def scrape_with_pagination(base_url: str, selectors: dict, pagination_selector: str = None,
                           max_pages: int = 5, retries=3, backoff=1, timeout=10):
    all_data = {k: [] for k in selectors.keys()}
    page_url = base_url
    pages_scraped = 0

    while page_url and pages_scraped < max_pages:
        info(f"Scraping page {pages_scraped + 1}: {page_url}")
        html = fetch_page(page_url, retries, backoff, timeout)
        if not html:
            error(f"Skipping page due to fetch failure: {page_url}")
            break

        soup = BeautifulSoup(html, 'lxml')

        for key, sel in selectors.items():
            elements = soup.select(sel)
            info(f"Found {len(elements)} elements for selector '{sel}' on page {pages_scraped + 1}")
            all_data[key].extend([el.get_text(strip=True) for el in elements])

        pages_scraped += 1

        if pagination_selector:
            next_link = soup.select_one(pagination_selector)
            if next_link and next_link.get('href'):
                page_url = urljoin(page_url, next_link['href'])
                info(f"Next page URL resolved to: {page_url}")
            else:
                page_url = None
        else:
            page_url = None

    items_count = max(len(v) for v in all_data.values()) if all_data else 0
    combined = []
    for i in range(items_count):
        row = {k: (v[i] if i < len(v) else None) for k, v in all_data.items()}
        combined.append(row)

    return combined
