"""Scrape NCAA NET Nitty Gritty data from WarrenNolan.com."""

import re

import pandas as pd
from tqdm import tqdm

from config import WARRENNOLAN_NITTY_URL, WARRENNOLAN_COMPARE_URL, TRAINING_SEASONS
from data.scraper_base import ScraperBase


# WarrenNolan URL slugs that don't match SR school_id via simple lowercasing
WARRENNOLAN_TO_SR = {
    "Saint-Marys-College": "saint-marys-ca",
    "Saint-Johns": "st-johns-ny",
    "Ole-Miss": "mississippi",
    "Texas-AM": "texas-am",
    "SIUE": "southern-illinois-edwardsville",
    "UNCW": "north-carolina-wilmington",
    "LSU": "louisiana-state",
    "UCF": "central-florida",
    "SMU": "southern-methodist",
    "TCU": "texas-christian",
    "VCU": "virginia-commonwealth",
    "UNLV": "nevada-las-vegas",
    "BYU": "brigham-young",
    "USC": "southern-california",
    "UAB": "alabama-birmingham",
    "Mount-Saint-Marys": "mount-st-marys",
    "UC-San-Diego": "california-san-diego",
    "UC-Irvine": "california-irvine",
    "UC-Davis": "california-davis",
    "UC-Riverside": "california-riverside",
    "UC-Santa-Barbara": "california-santa-barbara",
    "McNeese": "mcneese-state",
    "Southeast-Missouri": "southeast-missouri-state",
    "Saint-Francis-PA": "saint-francis-pa",
    "Loyola-Chicago": "loyola-il",
    "Loyola-Maryland": "loyola-md",
    "Loyola-Marymount": "loyola-marymount",
    "Miami-(FL)": "miami-fl",
    "UNCG": "north-carolina-greensboro",
    "Nicholls": "nicholls-state",
    "Saint-Bonaventure": "st-bonaventure",
    "Seattle-University": "seattle",
    "Charleston": "college-of-charleston",
    "FAU": "florida-atlantic",
    "Louisiana": "louisiana-lafayette",
    "Omaha": "nebraska-omaha",
    "Long-Island": "long-island-university",
    "UNC-Asheville": "north-carolina-asheville",
    "UNC-Greensboro": "north-carolina-greensboro",
    "UNC-Wilmington": "north-carolina-wilmington",
    "ETSU": "east-tennessee-state",
    "MTSU": "middle-tennessee",
    "FDU": "fairleigh-dickinson",
    "LIU": "long-island-university",
    "FGCU": "florida-gulf-coast",
    "SIU-Edwardsville": "southern-illinois-edwardsville",
    "St-Bonaventure": "st-bonaventure",
    "Saint-Peters": "saint-peters",
    "Saint-Josephs": "saint-josephs",
    "St-Thomas": "st-thomas-mn",
    "UT-Arlington": "texas-arlington",
    "UT-Martin": "tennessee-martin",
    "UT-Rio-Grande-Valley": "texas-rio-grande-valley",
    "UMass": "massachusetts",
    "UMass-Lowell": "massachusetts-lowell",
    "UConn": "connecticut",
    "NIU": "northern-illinois",
    "SFA": "stephen-f-austin",
    "UTEP": "texas-el-paso",
    "UTSA": "texas-san-antonio",
    "IUPUI": "iupui",
    "UMBC": "maryland-baltimore-county",
    "UMKC": "missouri-kansas-city",
    "UIC": "illinois-chicago",
    "UNI": "northern-iowa",
    "NJIT": "njit",
    "App-State": "appalachian-state",
    "Miami-FL": "miami-fl",
    "Miami-OH": "miami-oh",
    "Central-Connecticut": "central-connecticut-state",
    "Southern-Miss": "southern-mississippi",
    "Ole-Miss": "mississippi",
    "Pitt": "pittsburgh",
    "Grambling": "grambling",
    "Purdue-Fort-Wayne": "ipfw",
    "IU-Indianapolis": "iupui",
    "Le-Moyne": "le-moyne",
    "Queens-University": "queens-nc",
    "Stonehill": "stonehill",
    "Lindenwood": "lindenwood",
    "Texas-AM-Commerce": "texas-am-commerce",
    "Texas-AM-Corpus-Christi": "texas-am-corpus-christi",
    "Prairie-View-AM": "prairie-view",
    "Albany": "albany-ny",
    "Bowling-Green": "bowling-green-state",
    "Detroit": "detroit-mercy",
    "East-Texas-AM": "texas-am-commerce",
    "FIU": "florida-international",
    "Grambling-State": "grambling",
    "Houston-Christian": "houston-baptist",
    "IU-Indy": "iupui",
    "Penn": "pennsylvania",
    "Presbyterian-College": "presbyterian",
    "Queens": "queens-nc",
    "Saint-Thomas": "st-thomas-mn",
    "The-Citadel": "citadel",
    "ULM": "louisiana-monroe",
    "UTA": "texas-arlington",
    "Utah-Tech": "dixie-state",
    "UTRGV": "texas-pan-american",
    "VMI": "virginia-military-institute",
    "Little-Rock": "arkansas-little-rock",
}


def warrennolan_slug_to_school_id(slug: str) -> str:
    """Convert a WarrenNolan URL slug to SR school_id format.

    Args:
        slug: e.g., "Duke", "Michigan-State", "Saint-Marys-College"

    Returns:
        Lowercase-hyphenated SR school_id: "duke", "michigan-state", "saint-marys-ca"
    """
    if slug in WARRENNOLAN_TO_SR:
        return WARRENNOLAN_TO_SR[slug]
    # Default: just lowercase the slug
    return slug.lower()


def parse_record(text: str) -> tuple[int, int]:
    """Parse a 'W-L' record string into (wins, losses).

    Args:
        text: e.g., "8-2", "12-4", "35-4"

    Returns:
        (wins, losses) as integers, or (0, 0) if unparseable
    """
    text = text.strip()
    m = re.match(r"^(\d+)-(\d+)$", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


class NittyGrittyScraper(ScraperBase):
    """Scrape NET Nitty Gritty data from WarrenNolan.com."""

    # Map header text (lowercased, stripped) to our field names
    HEADER_MAP = {
        "net": "net",
        "sos": "sos",
        "road": "road",
        "q1": "q1",
        "q2": "q2",
        "q3": "q3",
        "q4": "q4",
        "avgnet ws": "avg_net_wins",
        "avgnetws": "avg_net_wins",
        "avg net ws": "avg_net_wins",
        "avgnet ls": "avg_net_losses",
        "avgnetls": "avg_net_losses",
        "avg net ls": "avg_net_losses",
    }

    def _build_col_map(self, table) -> dict[str, int]:
        """Build a mapping from field name to column index using header text."""
        header_row = table.find("tr")
        if not header_row:
            return {}

        col_map = {}
        for i, cell in enumerate(header_row.find_all(["th", "td"])):
            text = cell.get_text(strip=True).lower().replace(" ", "")
            # Normalize common variations
            for header_key, field_name in self.HEADER_MAP.items():
                if text == header_key.replace(" ", ""):
                    col_map[field_name] = i
                    break
        return col_map

    def scrape_nitty(self, season: int) -> pd.DataFrame:
        """Scrape NET Nitty Gritty data for a single season.

        Returns DataFrame with: team, school_id, season, net_ranking, net_sos,
        q1_wins, q1_losses, q2_wins, q2_losses, q3_wins, q3_losses,
        q4_wins, q4_losses, road_wins, road_losses, avg_net_wins, avg_net_losses
        """
        url = WARRENNOLAN_NITTY_URL.format(year=season)
        soup = self.fetch_and_parse(url)

        # Find the data table — try common class names
        table = None
        for cls in ["normal-grid", "sortable-onload"]:
            table = soup.find("table", class_=re.compile(cls))
            if table:
                break

        # Fallback: find any table with enough rows
        if table is None:
            for t in soup.find_all("table"):
                if len(t.find_all("tr")) > 50:
                    table = t
                    break

        if table is None:
            print(f"  Warning: no NET table found for {season}")
            return pd.DataFrame()

        # Build column index map from headers
        col_map = self._build_col_map(table)
        if "net" not in col_map or "q1" not in col_map:
            print(f"  Warning: could not map columns for {season}: {col_map}")
            return pd.DataFrame()

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 15:
                continue

            # Find team slug from any cell containing a schedule link
            link = tr.find("a", href=re.compile(r"/schedule/"))
            if not link:
                continue

            slug_match = re.search(r"/schedule/([^/?]+)", link.get("href", ""))
            if not slug_match:
                continue

            slug = slug_match.group(1)
            team_name = link.get_text(strip=True)
            school_id = warrennolan_slug_to_school_id(slug)

            # Extract NET ranking
            try:
                net_ranking = int(cells[col_map["net"]].get_text(strip=True))
            except (ValueError, IndexError):
                continue

            # Extract SOS
            try:
                net_sos = int(cells[col_map["sos"]].get_text(strip=True))
            except (ValueError, IndexError, KeyError):
                net_sos = 0

            # Extract records using header-mapped indices
            road_w, road_l = parse_record(cells[col_map["road"]].get_text(strip=True))
            q1_w, q1_l = parse_record(cells[col_map["q1"]].get_text(strip=True))
            q2_w, q2_l = parse_record(cells[col_map["q2"]].get_text(strip=True))
            q3_w, q3_l = parse_record(cells[col_map["q3"]].get_text(strip=True))
            q4_w, q4_l = parse_record(cells[col_map["q4"]].get_text(strip=True))

            # Extract avg NET
            try:
                avg_net_wins = int(cells[col_map["avg_net_wins"]].get_text(strip=True))
            except (ValueError, IndexError, KeyError):
                avg_net_wins = 0

            try:
                avg_net_losses = int(cells[col_map["avg_net_losses"]].get_text(strip=True))
            except (ValueError, IndexError, KeyError):
                avg_net_losses = 0

            rows.append({
                "team": team_name,
                "school_id": school_id,
                "season": season,
                "net_ranking": net_ranking,
                "net_sos": net_sos,
                "q1_wins": q1_w,
                "q1_losses": q1_l,
                "q2_wins": q2_w,
                "q2_losses": q2_l,
                "q3_wins": q3_w,
                "q3_losses": q3_l,
                "q4_wins": q4_w,
                "q4_losses": q4_l,
                "road_wins": road_w,
                "road_losses": road_l,
                "avg_net_wins": avg_net_wins,
                "avg_net_losses": avg_net_losses,
            })

        if not rows:
            print(f"  Warning: no rows parsed for {season}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        print(f"  {season}: {len(df)} teams scraped")
        return df

    # Map compare-rankings header text to our field names
    COMPARE_HEADER_MAP = {
        "net": "net_ranking",
        "kpi": "kpi",
        "sor": "sor",
        "wab": "wab",
        "bpi": "bpi",
        "pom": "pom",
        "t-rank": "trank",
        "sag": "trank",  # SAG (Sagarin) was used pre-2025, same slot as T-Rank
    }

    def scrape_compare_rankings(self, season: int) -> pd.DataFrame:
        """Scrape compare-rankings page for a single season.

        Returns DataFrame with: team, school_id, season, net_ranking,
        kpi, sor, wab, bpi, pom, trank
        """
        url = WARRENNOLAN_COMPARE_URL.format(year=season)
        soup = self.fetch_and_parse(url)

        # Find the data table
        table = None
        for cls in ["normal-grid", "sortable-onload"]:
            table = soup.find("table", class_=re.compile(cls))
            if table:
                break

        if table is None:
            for t in soup.find_all("table"):
                if len(t.find_all("tr")) > 50:
                    table = t
                    break

        if table is None:
            print(f"  Warning: no compare-rankings table found for {season}")
            return pd.DataFrame()

        # Build column map from headers
        header_row = table.find("tr")
        if not header_row:
            return pd.DataFrame()

        col_map = {}
        for i, cell in enumerate(header_row.find_all(["th", "td"])):
            text = cell.get_text(strip=True).lower().replace(" ", "")
            for header_key, field_name in self.COMPARE_HEADER_MAP.items():
                if text == header_key.replace(" ", ""):
                    col_map[field_name] = i
                    break

        if "net_ranking" not in col_map:
            print(f"  Warning: could not find NET column for {season}: {col_map}")
            return pd.DataFrame()

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 5:
                continue

            # Find team slug from schedule link
            link = tr.find("a", href=re.compile(r"/schedule/"))
            if not link:
                continue

            slug_match = re.search(r"/schedule/([^/?]+)", link.get("href", ""))
            if not slug_match:
                continue

            slug = slug_match.group(1)
            team_name = link.get_text(strip=True)
            school_id = warrennolan_slug_to_school_id(slug)

            row = {
                "team": team_name,
                "school_id": school_id,
                "season": season,
            }

            # Extract each metric as an integer ranking
            for field, idx in col_map.items():
                try:
                    val = cells[idx].get_text(strip=True)
                    # WAB can be a float like "4.5" or negative "-1.2"
                    if field == "wab":
                        row[field] = float(val)
                    else:
                        row[field] = int(val)
                except (ValueError, IndexError):
                    row[field] = 0

            rows.append(row)

        if not rows:
            print(f"  Warning: no compare-rankings rows parsed for {season}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Ensure all expected columns exist
        for field in ["net_ranking", "kpi", "sor", "wab", "bpi", "pom", "trank"]:
            if field not in df.columns:
                df[field] = 0
        print(f"  compare-rankings {season}: {len(df)} teams")
        return df

    def scrape_all_seasons(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Scrape both NET nitty gritty and compare-rankings for all seasons.

        Merges both sources: compare-rankings provides rankings (KPI, SOR, WAB,
        BPI, POM, T-Rank) for all ~365 teams; net-nitty provides detailed records
        (Q1-Q4, road, avg NET) for ~120 teams.
        """
        if seasons is None:
            seasons = TRAINING_SEASONS

        # Filter to NET-era seasons only
        net_seasons = [s for s in seasons if s >= 2019 and s != 2020]

        all_dfs = []
        for season in tqdm(net_seasons, desc="Scraping WarrenNolan"):
            try:
                # Scrape compare-rankings (primary — all teams)
                compare_df = self.scrape_compare_rankings(season)

                # Scrape net-nitty (supplemental — ~120 teams with detailed records)
                nitty_df = self.scrape_nitty(season)

                if compare_df.empty and nitty_df.empty:
                    continue

                if compare_df.empty:
                    all_dfs.append(nitty_df)
                    continue

                if nitty_df.empty:
                    all_dfs.append(compare_df)
                    continue

                # Merge: compare-rankings is the base, net-nitty adds detail columns
                nitty_extra_cols = [
                    "school_id", "season", "net_sos",
                    "q1_wins", "q1_losses", "q2_wins", "q2_losses",
                    "q3_wins", "q3_losses", "q4_wins", "q4_losses",
                    "road_wins", "road_losses", "avg_net_wins", "avg_net_losses",
                ]
                nitty_merge = nitty_df[[c for c in nitty_extra_cols if c in nitty_df.columns]]
                merged = compare_df.merge(nitty_merge, on=["school_id", "season"], how="left")

                # Fill NaN for teams not in net-nitty (lower-ranked teams)
                for col in nitty_merge.columns:
                    if col not in ("school_id", "season"):
                        merged[col] = merged[col].fillna(0).astype(int)

                all_dfs.append(merged)

            except Exception as e:
                print(f"\n  Failed season {season}: {e}. Keeping data from other seasons.")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)
