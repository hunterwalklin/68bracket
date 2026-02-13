"""Scrape bracketology projections from CBS Sports (Jerry Palm)."""

import hashlib
import json
import os
import re
from datetime import datetime, timedelta

import pandas as pd
from bs4 import BeautifulSoup

from config import (
    RAW_DIR, PROCESSED_DIR, CACHE_EXPIRY_DAYS,
    CBS_BRACKETOLOGY_URL,
)


# CBS display name → SR school_id (only entries that need manual mapping)
CBS_TO_SR = {
    "UConn": "connecticut",
    "Connecticut": "connecticut",
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
    "Miami (FL)": "miami-fl",
    "Miami (OH)": "miami-oh",
    "Miami FL": "miami-fl",
    "Miami OH": "miami-oh",
    "Miami": "miami-fl",
    "Saint Mary's": "saint-marys-ca",
    "Saint Mary's (CA)": "saint-marys-ca",
    "St. Mary's": "saint-marys-ca",
    "St. John's": "st-johns-ny",
    "St. John's (NY)": "st-johns-ny",
    "St. Bonaventure": "st-bonaventure",
    "Saint Bonaventure": "st-bonaventure",
    "Saint Peter's": "saint-peters",
    "Saint Joseph's": "saint-josephs",
    "Saint Louis": "saint-louis",
    "Saint Francis": "saint-francis-pa",
    "St. Thomas": "st-thomas-mn",
    "Mount St. Mary's": "mount-st-marys",
    "Loyola Chicago": "loyola-il",
    "Loyola Maryland": "loyola-md",
    "LIU": "long-island-university",
    "LIU Brooklyn": "long-island-university",
    "Long Island": "long-island-university",
    "FDU": "fairleigh-dickinson",
    "FGCU": "florida-gulf-coast",
    "ETSU": "east-tennessee-state",
    "East Tennessee State": "east-tennessee-state",
    "East Tennessee St.": "east-tennessee-state",
    "UNC Wilmington": "north-carolina-wilmington",
    "UNC Greensboro": "north-carolina-greensboro",
    "UNC Asheville": "north-carolina-asheville",
    "UNCW": "north-carolina-wilmington",
    "UMBC": "maryland-baltimore-county",
    "UMKC": "missouri-kansas-city",
    "UIC": "illinois-chicago",
    "UMass": "massachusetts",
    "UMass Lowell": "massachusetts-lowell",
    "NJIT": "njit",
    "IUPUI": "iupui",
    "IU Indianapolis": "iupui",
    "IU Indy": "iupui",
    "UT Arlington": "texas-arlington",
    "UT Martin": "tennessee-martin",
    "UT Rio Grande Valley": "texas-pan-american",
    "UTEP": "texas-el-paso",
    "UTSA": "texas-san-antonio",
    "UNI": "northern-iowa",
    "NIU": "northern-illinois",
    "SFA": "stephen-f-austin",
    "Stephen F. Austin": "stephen-f-austin",
    "Purdue Fort Wayne": "ipfw",
    "Cal State Northridge": "cal-state-northridge",
    "Cal State Fullerton": "cal-state-fullerton",
    "Cal State Bakersfield": "cal-state-bakersfield",
    "Cal St. Northridge": "cal-state-northridge",
    "Cal St. Fullerton": "cal-state-fullerton",
    "Sam Houston": "sam-houston-state",
    "Sam Houston State": "sam-houston-state",
    "McNeese": "mcneese-state",
    "McNeese State": "mcneese-state",
    "McNeese St.": "mcneese-state",
    "Nicholls": "nicholls-state",
    "Nicholls State": "nicholls-state",
    "Grambling": "grambling",
    "Grambling State": "grambling",
    "UC San Diego": "california-san-diego",
    "UC Irvine": "california-irvine",
    "UC Davis": "california-davis",
    "UC Riverside": "california-riverside",
    "UC Santa Barbara": "california-santa-barbara",
    "Southern Miss": "southern-mississippi",
    "Charleston": "college-of-charleston",
    "Omaha": "nebraska-omaha",
    "Little Rock": "arkansas-little-rock",
    "FAU": "florida-atlantic",
    "FIU": "florida-international",
    "Louisiana": "louisiana-lafayette",
    "App State": "appalachian-state",
    "Appalachian State": "appalachian-state",
    "Appalachian St.": "appalachian-state",
    "Prairie View": "prairie-view",
    "Prairie View A&M": "prairie-view",
    "Albany": "albany-ny",
    "Bowling Green": "bowling-green-state",
    "Cal Baptist": "california-baptist",
    "California Baptist": "california-baptist",
    "Central Connecticut": "central-connecticut-state",
    "East Texas A&M": "texas-am-commerce",
    "Houston Christian": "houston-baptist",
    "NC State": "north-carolina-state",
    "N.C. State": "north-carolina-state",
    "Penn": "pennsylvania",
    "The Citadel": "citadel",
    "USC Upstate": "south-carolina-upstate",
    "VMI": "virginia-military-institute",
    "Texas A&M-CC": "texas-am-corpus-christi",
    "Texas A&M Corpus Christi": "texas-am-corpus-christi",
    "Long Beach State": "long-beach-state",
    "Long Beach St.": "long-beach-state",
    "San Jose State": "san-jose-state",
    "San Jose St.": "san-jose-state",
    "Detroit Mercy": "detroit-mercy",
    "Detroit": "detroit-mercy",
    "Loyola Marymount": "loyola-marymount",
    "Seattle": "seattle",
    "Seattle U": "seattle",
    "Le Moyne": "le-moyne",
    "Queens": "queens-nc",
    "Stonehill": "stonehill",
    "Lindenwood": "lindenwood",
    "Southeast Missouri": "southeast-missouri-state",
    "Southeast Missouri State": "southeast-missouri-state",
    "Southeast Missouri St.": "southeast-missouri-state",
    "Utah Tech": "dixie-state",
    "Tennessee-Martin": "tennessee-martin",
    "TN Martin": "tennessee-martin",
    "Tennessee Martin": "tennessee-martin",
    "Bethune-Cookman": "bethune-cookman",
    "Morgan State": "morgan-state",
    "Merrimack": "merrimack",
    "Howard": "howard",
    "Vermont": "vermont",
    "Siena": "siena",
    "San Diego State": "san-diego-state",
    "San Diego St.": "san-diego-state",
    "Santa Clara": "santa-clara",
    "Troy": "troy",
    "Akron": "akron",
    "Belmont": "belmont",
    "Utah Valley": "utah-valley",
    "High Point": "high-point",
    "Portland State": "portland-state",
    "Portland St.": "portland-state",
    "Austin Peay": "austin-peay",
    "Wright State": "wright-state",
    "Navy": "navy",
    "North Dakota State": "north-dakota-state",
    "North Dakota St.": "north-dakota-state",
    "South Dakota State": "south-dakota-state",
    "South Dakota St.": "south-dakota-state",
    "MTSU": "middle-tennessee",
    "Middle Tennessee": "middle-tennessee",
}


def _normalize_name(name: str) -> str:
    """Lowercase/strip helper to match team names to school_id."""
    name = name.strip()
    # Remove common suffixes like "(FL)", conference tags, seed numbers
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)
    name = re.sub(r"\s*#\s*$", "", name)
    name = re.sub(r"\s*\*\s*$", "", name)
    return name


def _name_to_school_id(name: str) -> str:
    """Convert a CBS display name to SR school_id format."""
    name = _normalize_name(name)

    # Check direct mapping first
    if name in CBS_TO_SR:
        return CBS_TO_SR[name]

    # Standard conversion: lowercase, St. → State, spaces → hyphens
    sid = name.lower()
    sid = re.sub(r"\bst\.\s*$", "state", sid)
    sid = re.sub(r"\bst\.\s", "state ", sid)
    sid = re.sub(r"[.'()\"]", "", sid)
    sid = re.sub(r"[&]", "", sid)
    sid = re.sub(r"\s+", "-", sid)
    sid = re.sub(r"-+", "-", sid)
    sid = sid.strip("-")

    return sid


class BracketologyScraper:
    """Scrape bracketology projections using Playwright for JS-rendered pages."""

    def __init__(self):
        self.force_refresh = False
        os.makedirs(RAW_DIR, exist_ok=True)

    def _cache_path(self, url: str) -> str:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        safe = url.replace("https://", "").replace("/", "_").replace("?", "_")[:60]
        return os.path.join(RAW_DIR, f"bracketology_{safe}_{url_hash}.html")

    def _is_cache_valid(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now() - mtime < timedelta(days=CACHE_EXPIRY_DAYS)

    def _fetch_with_playwright(self, url: str) -> str:
        """Fetch a page using Playwright to render JS content."""
        cache = self._cache_path(url)
        if not self.force_refresh and self._is_cache_valid(cache):
            with open(cache, "r", encoding="utf-8") as f:
                return f.read()

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            # CBS may need extra time for bracket JS to render
            page.wait_for_timeout(3000)
            html = page.content()
            browser.close()

        with open(cache, "w", encoding="utf-8") as f:
            f.write(html)

        return html

    def scrape_cbs(self) -> pd.DataFrame:
        """Scrape CBS Sports bracketology page for team seeds and regions.

        Returns DataFrame with: team, school_id, seed, region, source
        """
        url = CBS_BRACKETOLOGY_URL
        html = self._fetch_with_playwright(url)
        soup = BeautifulSoup(html, "lxml")

        teams = []

        # Strategy 1: Look for bracket matchup rows with "(seed) Team" pattern
        # CBS typically renders bracket as matchup cards within region containers
        teams = self._parse_bracket_structure(soup)

        if not teams:
            # Strategy 2: Look for seed-team text patterns anywhere in page
            teams = self._parse_seed_team_text(soup)

        if not teams:
            # Strategy 3: Parse from text content using regex
            teams = self._parse_from_text(soup)

        if not teams:
            print("  Warning: could not parse CBS bracketology page")
            print("  Cached HTML available at:", self._cache_path(url))
            return pd.DataFrame()

        df = pd.DataFrame(teams)
        df["source"] = "CBS - Palm"
        print(f"  CBS bracketology: {len(df)} teams parsed")
        return df

    def _parse_bracket_structure(self, soup: BeautifulSoup) -> list[dict]:
        """Parse bracket from structured HTML elements (region divs, matchup rows)."""
        teams = []

        # Look for region containers — CBS typically uses heading elements or
        # data attributes to identify regions
        region_keywords = ["east", "west", "south", "midwest"]

        # Try finding region headers and parsing teams within each section
        all_text = soup.get_text()

        # Look for elements that contain region names as headers
        for el in soup.find_all(["h1", "h2", "h3", "h4", "div", "span", "section"]):
            text = el.get_text(strip=True).lower()
            for region in region_keywords:
                if region in text and "region" in text:
                    # Found a region header — look for seed/team data nearby
                    parent = el.find_parent(["div", "section"]) or el.parent
                    if parent:
                        region_teams = self._extract_teams_from_element(
                            parent, region.capitalize()
                        )
                        teams.extend(region_teams)

        # Deduplicate by school_id
        seen = set()
        unique = []
        for t in teams:
            if t["school_id"] not in seen:
                seen.add(t["school_id"])
                unique.append(t)

        return unique

    def _extract_teams_from_element(self, el, region: str) -> list[dict]:
        """Extract (seed, team) pairs from an element's text."""
        teams = []
        text = el.get_text("\n")

        # Pattern: "(seed) Team Name" or "seed. Team Name" or "seed Team Name"
        # Common CBS patterns: "1 Duke", "(1) Duke", "No. 1 Duke"
        patterns = [
            r"\((\d{1,2})\)\s+([A-Z][A-Za-z\s.&'()\-]+)",
            r"(?:No\.\s*)?(\d{1,2})\s+([A-Z][A-Za-z\s.&'()\-]+)",
        ]

        for pattern in patterns:
            for m in re.finditer(pattern, text):
                seed = int(m.group(1))
                name = m.group(2).strip()
                # Stop at common delimiters
                name = re.split(r"\s{2,}|\n|\t|vs\.?|,", name)[0].strip()
                if 1 <= seed <= 16 and len(name) > 2:
                    school_id = _name_to_school_id(name)
                    teams.append({
                        "team": _normalize_name(name),
                        "school_id": school_id,
                        "seed": seed,
                        "region": region,
                    })

        return teams

    def _parse_seed_team_text(self, soup: BeautifulSoup) -> list[dict]:
        """Scan all text for seed-team patterns, inferring regions from context."""
        teams = []
        current_region = ""
        region_keywords = {
            "east": "East", "west": "West",
            "south": "South", "midwest": "Midwest",
        }

        for el in soup.find_all(["div", "span", "td", "li", "p", "h2", "h3", "h4"]):
            text = el.get_text(strip=True)
            if not text:
                continue

            # Check for region header
            text_lower = text.lower()
            for key, val in region_keywords.items():
                if key in text_lower and ("region" in text_lower or len(text) < 30):
                    current_region = val
                    break

            # Look for seed-team pattern
            m = re.match(r"^\(?(\d{1,2})\)?\s+(.+)$", text)
            if m:
                seed = int(m.group(1))
                name = m.group(2).strip()
                name = re.split(r"\s{2,}|\t|vs\.?", name)[0].strip()
                if 1 <= seed <= 16 and len(name) > 2:
                    school_id = _name_to_school_id(name)
                    teams.append({
                        "team": _normalize_name(name),
                        "school_id": school_id,
                        "seed": seed,
                        "region": current_region or "Unknown",
                    })

        return teams

    def _parse_from_text(self, soup: BeautifulSoup) -> list[dict]:
        """Last resort: parse entire page text for seed-team patterns."""
        teams = []
        text = soup.get_text("\n")
        current_region = ""

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Region detection
            lower = line.lower()
            for key, val in [("east", "East"), ("west", "West"),
                             ("south", "South"), ("midwest", "Midwest")]:
                if key in lower and ("region" in lower or len(line) < 30):
                    current_region = val

            # Seed-team pattern
            m = re.match(r"^\(?(\d{1,2})\)?\s+([A-Z].+)$", line)
            if m:
                seed = int(m.group(1))
                name = m.group(2).strip()
                name = re.split(r"\s{2,}|\t|vs\.?|,", name)[0].strip()
                if 1 <= seed <= 16 and len(name) > 2:
                    school_id = _name_to_school_id(name)
                    teams.append({
                        "team": _normalize_name(name),
                        "school_id": school_id,
                        "seed": seed,
                        "region": current_region or "Unknown",
                    })

        return teams

    def save(self, df: pd.DataFrame) -> str:
        """Save bracketology data as JSON to processed dir."""
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, "bracketology.json")

        data = {
            "updated": str(datetime.now().date()),
            "sources": {},
        }

        for source in df["source"].unique():
            source_df = df[df["source"] == source]
            data["sources"][source] = {
                "teams": source_df[["team", "school_id", "seed", "region"]]
                .to_dict(orient="records")
            }

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Bracketology saved to {out_path} ({len(df)} teams)")
        return out_path

    def scrape_all(self) -> pd.DataFrame:
        """Scrape all bracketology sources and save."""
        df = self.scrape_cbs()
        if not df.empty:
            self.save(df)
        return df
