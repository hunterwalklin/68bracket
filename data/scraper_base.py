"""Base scraper with rate limiting, caching, and HTML parsing."""

import hashlib
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import RAW_DIR, REQUEST_DELAY, REQUEST_TIMEOUT, USER_AGENT, CACHE_EXPIRY_DAYS


class ScraperBase:
    """Base class for all scrapers with rate limiting and local caching."""

    _last_request_time = 0.0  # class-level to share across instances

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        os.makedirs(RAW_DIR, exist_ok=True)

    def _cache_path(self, url: str) -> str:
        """Generate a deterministic cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        safe_name = url.replace("https://", "").replace("http://", "")
        safe_name = safe_name.replace("/", "_").replace("?", "_")[:80]
        return os.path.join(RAW_DIR, f"{safe_name}_{url_hash}.html")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cached file exists and is within expiry window."""
        if not os.path.exists(cache_path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - mtime < timedelta(days=CACHE_EXPIRY_DAYS)

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - ScraperBase._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        ScraperBase._last_request_time = time.time()

    def fetch(self, url: str, force_refresh: bool = False) -> str:
        """Fetch URL content with caching and rate limiting.

        Returns the HTML string.
        """
        cache_path = self._cache_path(url)

        if not force_refresh and self._is_cache_valid(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        max_retries = 5
        for attempt in range(max_retries):
            self._rate_limit()
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s, 480s
                print(f"  Rate limited (429), waiting {wait}s before retry...")
                time.sleep(wait)
                ScraperBase._last_request_time = time.time()
                continue
            resp.raise_for_status()

            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

            return resp.text

        raise requests.exceptions.HTTPError(f"Still rate limited after {max_retries} retries: {url}")

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML string into BeautifulSoup object."""
        return BeautifulSoup(html, "lxml")

    def fetch_and_parse(self, url: str, force_refresh: bool = False) -> BeautifulSoup:
        """Fetch URL and return parsed BeautifulSoup object."""
        html = self.fetch(url, force_refresh=force_refresh)
        return self.parse_html(html)

    @staticmethod
    def table_to_rows(table) -> list[dict]:
        """Convert an HTML table element to a list of dicts.

        Handles Sports-Reference's comment-wrapped tables and multi-header rows.
        """
        if table is None:
            return []

        headers = []
        header_row = table.find("thead")
        if header_row:
            # Use the last header row (SR sometimes has multi-level headers)
            th_elements = header_row.find_all("tr")[-1].find_all("th")
            headers = [th.get("data-stat", th.get_text(strip=True)) for th in th_elements]

        rows = []
        tbody = table.find("tbody")
        if tbody is None:
            return rows

        for tr in tbody.find_all("tr"):
            # Skip separator/header rows within tbody
            if tr.get("class") and "thead" in tr.get("class", []):
                continue

            cells = tr.find_all(["td", "th"])
            if not cells:
                continue

            row = {}
            for i, cell in enumerate(cells):
                key = cell.get("data-stat", headers[i] if i < len(headers) else f"col_{i}")
                # Extract link href if present (useful for team URLs)
                link = cell.find("a")
                if link and link.get("href"):
                    row[f"{key}_link"] = link["href"]
                row[key] = cell.get_text(strip=True)
            rows.append(row)

        return rows

    @staticmethod
    def unwrap_comment_tables(soup: BeautifulSoup) -> BeautifulSoup:
        """Sports-Reference hides some tables in HTML comments. Unwrap them."""
        import re
        comments = soup.find_all(string=lambda text: isinstance(text, type(soup.new_string("").__class__)) and False)
        # Use regex to find comments containing tables
        for comment in soup.find_all(string=lambda t: t and isinstance(t, str) and "<table" in str(t)):
            pass  # BeautifulSoup Comment handling

        # More robust approach: find comments via regex on raw HTML
        html_str = str(soup)
        comment_pattern = re.compile(r'<!--(.*?)-->', re.DOTALL)
        for match in comment_pattern.finditer(html_str):
            comment_content = match.group(1)
            if "<table" in comment_content:
                # Parse the table from the comment and inject it
                comment_soup = BeautifulSoup(comment_content, "lxml")
                for table in comment_soup.find_all("table"):
                    table_id = table.get("id", "")
                    # Only add if not already present
                    if table_id and not soup.find("table", id=table_id):
                        soup.body.append(table) if soup.body else None

        return soup
