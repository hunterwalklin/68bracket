"""Scrape Bart Torvik T-Rank and quadrant data via Playwright."""

import hashlib
import os
import re
from datetime import datetime, timedelta
from urllib.parse import unquote_plus

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import RAW_DIR, CACHE_EXPIRY_DAYS, TRAINING_SEASONS


TORVIK_TRANK_URL = "https://barttorvik.com/trank.php?year={year}"
TORVIK_QUAD_URL = "https://barttorvik.com/quadrants.php?year={year}"

# Torvik team name → SR school_id (after URL-decoding and cleaning game info)
TORVIK_TO_SR = {
    "Connecticut": "connecticut",
    "UConn": "connecticut",
    "LSU": "louisiana-state",
    "UCF": "central-florida",
    "SMU": "southern-methodist",
    "TCU": "texas-christian",
    "VCU": "virginia-commonwealth",
    "UNLV": "nevada-las-vegas",
    "BYU": "brigham-young",
    "USC": "southern-california",
    "UAB": "alabama-birmingham",
    "UCLA": "ucla",
    "UNC": "north-carolina",
    "Ole Miss": "mississippi",
    "Pitt": "pittsburgh",
    "Texas A&M": "texas-am",
    "Miami FL": "miami-fl",
    "Miami OH": "miami-oh",
    "Saint Mary's": "saint-marys-ca",
    "St. John's": "st-johns-ny",
    "St. Bonaventure": "st-bonaventure",
    "Saint Peter's": "saint-peters",
    "Saint Joseph's": "saint-josephs",
    "Saint Louis": "saint-louis",
    "Saint Francis": "saint-francis-pa",
    "St. Thomas": "st-thomas-mn",
    "Mount St. Mary's": "mount-st-marys",
    "Loyola Chicago": "loyola-il",
    "Loyola Maryland": "loyola-md",
    "LIU": "long-island-university",
    "FDU": "fairleigh-dickinson",
    "FGCU": "florida-gulf-coast",
    "ETSU": "east-tennessee-state",
    "SIUE": "southern-illinois-edwardsville",
    "UNC Wilmington": "north-carolina-wilmington",
    "UNC Greensboro": "north-carolina-greensboro",
    "UNC Asheville": "north-carolina-asheville",
    "UMBC": "maryland-baltimore-county",
    "UMKC": "missouri-kansas-city",
    "UIC": "illinois-chicago",
    "UMass": "massachusetts",
    "UMass Lowell": "massachusetts-lowell",
    "NJIT": "njit",
    "IUPUI": "iupui",
    "IU Indianapolis": "indiana-purdue-indianapolis",
    "UT Arlington": "texas-arlington",
    "UT Martin": "tennessee-martin",
    "UT Rio Grande Valley": "texas-rio-grande-valley",
    "UTEP": "texas-el-paso",
    "UTSA": "texas-san-antonio",
    "SIU Edwardsville": "southern-illinois-edwardsville",
    "UNI": "northern-iowa",
    "NIU": "northern-illinois",
    "SFA": "stephen-f-austin",
    "Purdue Fort Wayne": "purdue-fort-wayne",
    "Cal St. Northridge": "cal-state-northridge",
    "Cal St. Fullerton": "cal-state-fullerton",
    "Cal St. Bakersfield": "cal-state-bakersfield",
    "Sam Houston St.": "sam-houston-state",
    "Sam Houston": "sam-houston-state",
    "McNeese St.": "mcneese-state",
    "McNeese": "mcneese-state",
    "Nicholls St.": "nicholls-state",
    "Nicholls": "nicholls-state",
    "Grambling St.": "grambling-state",
    "Grambling": "grambling-state",
    "UC San Diego": "california-san-diego",
    "UC Irvine": "california-irvine",
    "UC Davis": "california-davis",
    "UC Riverside": "california-riverside",
    "UC Santa Barbara": "california-santa-barbara",
    "Southern Miss": "southern-mississippi",
    "Charleston": "college-of-charleston",
    "Omaha": "nebraska-omaha",
    "Little Rock": "arkansas-little-rock",
    "Long Island": "long-island-university",
    "Le Moyne": "le-moyne",
    "Queens": "queens-nc",
    "Stonehill": "stonehill",
    "Lindenwood": "lindenwood",
    "Seattle": "seattle",
    "FAU": "florida-atlantic",
    "Louisiana": "louisiana-lafayette",
    "App St.": "appalachian-state",
    "Appalachian St.": "appalachian-state",
    "Prairie View": "prairie-view",
    "Prairie View A&M": "prairie-view",
    "Chicago St.": "chicago-state",
    "Albany": "albany-ny",
    "Bowling Green": "bowling-green-state",
    "Cal Baptist": "california-baptist",
    "Central Connecticut": "central-connecticut-state",
    "East Texas A&M": "texas-am-commerce",
    "FIU": "florida-international",
    "Houston Christian": "houston-baptist",
    "IU Indy": "indiana-purdue-indianapolis",
    "IU Indianapolis": "indiana-purdue-indianapolis",
    "NC State": "north-carolina-state",
    "Penn": "pennsylvania",
    "The Citadel": "citadel",
    "USC Upstate": "south-carolina-upstate",
    "Utah Tech": "dixie-state",
    "VMI": "virginia-military-institute",
    "Grambling": "grambling",
    "Grambling St.": "grambling",
    "Texas A&M Corpus Christi": "texas-am-corpus-christi",
    "Texas A&M Corpus Chris": "texas-am-corpus-christi",
    "Texas AM Corpus Chris": "texas-am-corpus-christi",
    "Texas AM Corpus Christi": "texas-am-corpus-christi",
    "East Texas AM": "texas-am-commerce",
    "Prairie View AM": "prairie-view",
    "UT Rio Grande Valley": "texas-rio-grande-valley",
    "Purdue Fort Wayne": "purdue-fort-wayne",
    "Long Beach St.": "long-beach-state",
    "San Jose St.": "san-jose-state",
}


def torvik_slug_to_school_id(slug: str) -> str:
    """Convert a Torvik URL slug to SR school_id format.

    Args:
        slug: URL-encoded slug like "Michigan+St.", "Saint+Mary%27s"

    Returns:
        SR school_id like "michigan-state", "saint-marys-ca"
    """
    # URL-decode: + → space, %27 → apostrophe
    name = unquote_plus(slug)

    # Check direct mapping first
    if name in TORVIK_TO_SR:
        return TORVIK_TO_SR[name]

    # Convert "St." suffix to "State" for SR format
    # e.g. "Michigan St." → "michigan-state"
    name_lower = name.lower()
    name_lower = re.sub(r"\bst\.\s*$", "state", name_lower)
    name_lower = re.sub(r"\bst\.\s", "state ", name_lower)

    # Standard conversion: lowercase, spaces → hyphens, drop punctuation
    school_id = name_lower
    school_id = re.sub(r"[.'()]", "", school_id)
    school_id = re.sub(r"[&]", "", school_id)
    school_id = re.sub(r"\s+", "-", school_id)
    school_id = re.sub(r"-+", "-", school_id)
    school_id = school_id.strip("-")

    return school_id


def parse_record(text: str) -> tuple[int, int]:
    """Parse a 'W-L' record string into (wins, losses)."""
    text = text.strip()
    m = re.match(r"^(\d+)-(\d+)$", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def _clean_team_text(text: str) -> str:
    """Remove game info appended to team names like '(A) 72 Northwestern'."""
    # Strip trailing game context: (A), (H), (N) followed by score and opponent
    text = re.sub(r"\s*\([AHN]\)\s*\d+.*$", "", text)
    return text.strip()


class TorvikvScraper:
    """Scrape Bart Torvik data using Playwright for Cloudflare bypass."""

    def __init__(self):
        self.force_refresh = False
        os.makedirs(RAW_DIR, exist_ok=True)

    def _cache_path(self, url: str) -> str:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        safe = url.replace("https://", "").replace("/", "_").replace("?", "_")[:60]
        return os.path.join(RAW_DIR, f"torvik_{safe}_{url_hash}.html")

    def _is_cache_valid(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now() - mtime < timedelta(days=CACHE_EXPIRY_DAYS)

    def _fetch_with_playwright(self, url: str) -> str:
        """Fetch a page using Playwright to bypass Cloudflare."""
        cache = self._cache_path(url)
        if not self.force_refresh and self._is_cache_valid(cache):
            with open(cache, "r", encoding="utf-8") as f:
                return f.read()

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            html = page.content()
            browser.close()

        with open(cache, "w", encoding="utf-8") as f:
            f.write(html)

        return html

    def scrape_trank(self, season: int) -> pd.DataFrame:
        """Scrape T-Rank ratings page for a season.

        Returns DataFrame with: team, school_id, season, trank_rank,
        adj_oe, adj_de, barthag, wab
        """
        url = TORVIK_TRANK_URL.format(year=season)
        html = self._fetch_with_playwright(url)
        soup = BeautifulSoup(html, "lxml")

        table = soup.find("table")
        if not table:
            print(f"  Warning: no trank table found for {season}")
            return pd.DataFrame()

        trs = table.find_all("tr")

        # Find header row (contains "Rk")
        header_idx = 0
        col_map = {}
        for i, tr in enumerate(trs):
            cells = tr.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if "Rk" in texts:
                header_idx = i
                for j, t in enumerate(texts):
                    t_lower = t.lower().strip()
                    if t_lower == "rk":
                        col_map["rank"] = j
                    elif t_lower == "adjoe":
                        col_map["adj_oe"] = j
                    elif t_lower == "adjde":
                        col_map["adj_de"] = j
                    elif t_lower == "barthag":
                        col_map["barthag"] = j
                    elif t_lower == "wab":
                        col_map["wab"] = j
                break

        if "rank" not in col_map:
            print(f"  Warning: could not find Rk column for {season}")
            return pd.DataFrame()

        rows = []
        for tr in trs[header_idx + 1:]:
            cells = tr.find_all("td")
            if len(cells) < 8:
                continue

            link = tr.find("a", href=re.compile(r"team\.php"))
            if not link:
                continue

            href = link.get("href", "")
            m = re.search(r"team=([^&]+)", href)
            if not m:
                continue

            slug = m.group(1)
            team_name = _clean_team_text(link.get_text(strip=True))
            school_id = torvik_slug_to_school_id(slug)

            try:
                rank = int(cells[col_map["rank"]].get_text(strip=True))
            except (ValueError, IndexError):
                continue

            row = {
                "team": team_name,
                "school_id": school_id,
                "season": season,
                "trank_rank": rank,
            }

            for field in ["adj_oe", "adj_de", "barthag"]:
                try:
                    row[field] = float(cells[col_map[field]].get_text(strip=True))
                except (ValueError, IndexError, KeyError):
                    row[field] = 0.0

            # WAB is like "+8.61" or "-2.3"
            try:
                wab_text = cells[col_map["wab"]].get_text(strip=True)
                row["wab"] = float(wab_text.replace("+", ""))
            except (ValueError, IndexError, KeyError):
                row["wab"] = 0.0

            rows.append(row)

        if not rows:
            print(f"  Warning: no trank rows parsed for {season}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        print(f"  trank {season}: {len(df)} teams")
        return df

    def scrape_quadrants(self, season: int) -> pd.DataFrame:
        """Scrape quadrant records page for a season.

        Returns DataFrame with: school_id, season, q1a_wins, q1a_losses,
        q1_wins, q1_losses, q2_wins, q2_losses, q3_wins, q3_losses,
        q4_wins, q4_losses
        """
        url = TORVIK_QUAD_URL.format(year=season)
        html = self._fetch_with_playwright(url)
        soup = BeautifulSoup(html, "lxml")

        table = soup.find("table")
        if not table:
            print(f"  Warning: no quad table found for {season}")
            return pd.DataFrame()

        trs = table.find_all("tr")

        # Find header row
        header_idx = 0
        col_map = {}
        for i, tr in enumerate(trs):
            cells = tr.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]
            if "Q1" in texts:
                header_idx = i
                for j, t in enumerate(texts):
                    if t == "Q1A":
                        col_map["q1a"] = j
                    elif t == "Q1":
                        col_map["q1"] = j
                    elif t == "Q2":
                        col_map["q2"] = j
                    elif t == "Q3":
                        col_map["q3"] = j
                    elif t == "Q4":
                        col_map["q4"] = j
                break

        if "q1" not in col_map:
            print(f"  Warning: could not find Q1 column for {season}")
            return pd.DataFrame()

        rows = []
        for tr in trs[header_idx + 1:]:
            cells = tr.find_all("td")
            if len(cells) < 5:
                continue

            link = tr.find("a", href=re.compile(r"team\.php"))
            if not link:
                continue

            href = link.get("href", "")
            m = re.search(r"team=([^&]+)", href)
            if not m:
                continue

            slug = m.group(1)
            school_id = torvik_slug_to_school_id(slug)

            row = {"school_id": school_id, "season": season}

            for field in ["q1a", "q1", "q2", "q3", "q4"]:
                if field in col_map:
                    w, l = parse_record(cells[col_map[field]].get_text(strip=True))
                    row[f"{field}_wins"] = w
                    row[f"{field}_losses"] = l
                else:
                    row[f"{field}_wins"] = 0
                    row[f"{field}_losses"] = 0

            rows.append(row)

        if not rows:
            print(f"  Warning: no quad rows parsed for {season}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        print(f"  quads {season}: {len(df)} teams")
        return df

    def scrape_all_seasons(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Scrape both trank and quadrant data for all seasons, merged."""
        if seasons is None:
            seasons = TRAINING_SEASONS

        # Filter to valid seasons (NET era, skip 2020)
        valid = [s for s in seasons if s >= 2019 and s != 2020]

        all_dfs = []
        for season in tqdm(valid, desc="Scraping Torvik"):
            try:
                trank_df = self.scrape_trank(season)
                quad_df = self.scrape_quadrants(season)

                if trank_df.empty:
                    continue

                if not quad_df.empty:
                    # Merge quads into trank by school_id + season
                    quad_cols = [c for c in quad_df.columns if c not in trank_df.columns
                                 or c in ("school_id", "season")]
                    merged = trank_df.merge(
                        quad_df[quad_cols], on=["school_id", "season"], how="left"
                    )
                    # Fill missing quad data with 0
                    for col in quad_df.columns:
                        if col not in ("school_id", "season") and col in merged.columns:
                            merged[col] = merged[col].fillna(0).astype(int)
                    all_dfs.append(merged)
                else:
                    all_dfs.append(trank_df)

            except Exception as e:
                print(f"\n  Failed season {season}: {e}. Keeping data from other seasons.")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)
