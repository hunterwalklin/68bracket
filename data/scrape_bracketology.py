"""Scrape bracketology projections from Bracket Matrix (bracketmatrix.com).

Bracket Matrix aggregates 100+ bracketology sources into a single static HTML
table, replacing the old CBS/ESPN Playwright scrapers that consistently timed out.
"""

import json
import os
import re
import warnings
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from config import RAW_DIR, PROCESSED_DIR, BRACKET_MATRIX_URL
from data.scraper_base import ScraperBase


# Display name → SR school_id (only entries that need manual mapping)
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
    "Miami (FLA.)": "miami-fl",
    "Miami (Ohio)": "miami-oh",
    "Saint Mary's": "saint-marys-ca",
    "Saint Mary's (CA)": "saint-marys-ca",
    "St. Mary's": "saint-marys-ca",
    "St. Mary\u2019s (CA)": "saint-marys-ca",
    "St. Mary's (CA)": "saint-marys-ca",
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
    "E. Tennessee State": "east-tennessee-state",
    "E. Tennessee St.": "east-tennessee-state",
    "UNC Wilmington": "north-carolina-wilmington",
    "UNC Greensboro": "north-carolina-greensboro",
    "UNC Asheville": "north-carolina-asheville",
    "UNCW": "north-carolina-wilmington",
    "NC-Wilmington": "north-carolina-wilmington",
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
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)
    name = re.sub(r"\s*#\s*$", "", name)
    name = re.sub(r"\s*\*\s*$", "", name)
    return name


def _name_to_school_id(name: str) -> str:
    """Convert a display name to SR school_id format."""
    name = _normalize_name(name)

    if name in CBS_TO_SR:
        return CBS_TO_SR[name]

    sid = name.lower()
    sid = re.sub(r"\bst\.\s*$", "state", sid)
    sid = re.sub(r"\bst\.\s", "state ", sid)
    sid = re.sub(r"[.'()\"]", "", sid)
    sid = re.sub(r"[&]", "", sid)
    sid = re.sub(r"\s+", "-", sid)
    sid = re.sub(r"-+", "-", sid)
    sid = sid.strip("-")

    return sid


class BracketologyScraper(ScraperBase):
    """Scrape consensus bracketology from Bracket Matrix.

    Bracket Matrix aggregates 100+ bracketology sources into a single HTML
    table. This replaces the old CBS/ESPN Playwright-based scrapers.
    """

    def _fetch_bracket_matrix(self) -> str:
        """Fetch Bracket Matrix HTML, bypassing expired SSL cert."""
        url = BRACKET_MATRIX_URL
        cache_path = self._cache_path(url)

        if not self.force_refresh and self._is_cache_valid(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        # Bracket Matrix has an expired SSL cert — disable verification
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.session.verify = False
            html = self.fetch(url)
            self.session.verify = True

        return html

    def _parse_source_names(self, rows) -> dict[int, str]:
        """Parse source names from header row, accounting for colspans.

        Returns mapping of data-column index -> source name.
        """
        header_cells = rows[0].find_all(["td", "th"])
        col_names = []
        for cell in header_cells:
            text = cell.get_text(strip=True)
            colspan = int(cell.get("colspan", 1))
            col_names.append(text)
            for _ in range(colspan - 1):
                col_names.append("")

        # Individual source columns start at col 6 (after seed, team, conf,
        # avg seed, # brackets, and an empty gap column)
        source_map = {}
        for i in range(6, len(col_names)):
            if col_names[i]:
                source_map[i] = col_names[i]
        return source_map

    def _parse_table(self, html: str) -> tuple[list[dict], dict[int, str]]:
        """Parse the Bracket Matrix HTML table into team records.

        Table structure (single <table>):
          Row 0: Source names (with colspans)
          Row 1: Update dates
          Row 2: Empty separator
          Row 3+: Data — col 0=seed, 1=team, 2=conf, 3=avg seed,
                  4=# brackets, 5=gap, 6+=per-source seeds
          Stops at "OTHER AT-LARGES" separator row.

        Returns (teams list, source_map) where source_map is
        col_index -> source_name for individual sources.
        """
        soup = self.parse_html(html)
        table = soup.find("table")
        if not table:
            print("  Warning: no table found on Bracket Matrix page")
            return [], {}

        rows = table.find_all("tr")
        source_map = self._parse_source_names(rows)
        teams = []

        for row in rows[3:]:  # skip header rows
            cells = row.find_all(["td", "th"])
            if len(cells) < 5:
                continue

            seed_text = cells[0].get_text(strip=True)
            team_name = cells[1].get_text(strip=True)

            # Stop at "OTHER AT-LARGES" separator
            if "OTHER AT-LARGE" in team_name.upper():
                break

            if not seed_text.isdigit():
                continue

            seed = int(seed_text)
            if seed < 1 or seed > 16:
                continue

            avg_seed_text = cells[3].get_text(strip=True)
            try:
                avg_seed = float(avg_seed_text)
            except ValueError:
                avg_seed = float(seed)

            # Parse individual source seeds
            per_source = {}
            for col_idx, src_name in source_map.items():
                if col_idx < len(cells):
                    val = cells[col_idx].get_text(strip=True)
                    if val.isdigit():
                        per_source[src_name] = int(val)

            school_id = _name_to_school_id(team_name)
            teams.append({
                "team": _normalize_name(team_name),
                "school_id": school_id,
                "seed": round(avg_seed),
                "avg_seed": avg_seed,
                "region": "",
                "per_source": per_source,
            })

        return teams, source_map

    def save(self, teams: list[dict], source_names: list[str]) -> str:
        """Save bracketology data as JSON to processed dir.

        Outputs one entry per individual source in the sources dict,
        plus a "Bracket Matrix (Avg)" consensus entry.
        """
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, "bracketology.json")

        data = {
            "updated": str(datetime.now().date()),
            "sources": {},
        }

        # Consensus average entry (keep float for 2-decimal display)
        data["sources"]["BM Avg"] = {
            "teams": [
                {
                    "team": t["team"],
                    "school_id": t["school_id"],
                    "seed": t["avg_seed"],
                    "region": "",
                }
                for t in teams
            ]
        }

        # Individual source entries
        for src_name in source_names:
            src_teams = []
            for t in teams:
                seed = t["per_source"].get(src_name)
                if seed is not None:
                    src_teams.append({
                        "team": t["team"],
                        "school_id": t["school_id"],
                        "seed": seed,
                        "region": "",
                    })
            if src_teams:
                data["sources"][src_name] = {"teams": src_teams}

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        n_sources = len(data["sources"]) - 1  # exclude avg
        print(f"  Bracketology saved to {out_path} "
              f"({len(teams)} teams, {n_sources} sources)")
        return out_path

    def scrape_all(self) -> pd.DataFrame:
        """Scrape Bracket Matrix and save."""
        html = self._fetch_bracket_matrix()
        teams, source_map = self._parse_table(html)

        if not teams:
            print("  Warning: no teams parsed from Bracket Matrix")
            return pd.DataFrame()

        source_names = [name for _, name in sorted(source_map.items())]
        print(f"  Bracket Matrix: {len(teams)} teams, "
              f"{len(source_names)} sources parsed")
        self.save(teams, source_names)

        # Return DataFrame for backward compat (uses avg seed)
        df = pd.DataFrame([
            {"team": t["team"], "school_id": t["school_id"],
             "seed": t["seed"], "region": ""}
            for t in teams
        ])
        df["source"] = "Bracket Matrix"
        return df
