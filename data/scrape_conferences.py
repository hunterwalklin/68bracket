"""Scrape conference tournament winners (auto-bids) from Sports-Reference."""

import re

import pandas as pd
from tqdm import tqdm

from config import SR_BASE, TRAINING_SEASONS
from data.scraper_base import ScraperBase


class ConferenceScraper(ScraperBase):
    """Scrape conference tournament results to identify auto-bid teams."""

    def scrape_conference_winners(self, season: int) -> pd.DataFrame:
        """Scrape conference tournament winners for a season.

        Returns DataFrame with: team, school_id, season, conference, conf_tourney_winner
        """
        # The main season page lists conference tourney results
        url = f"{SR_BASE}/seasons/{season}.html"
        soup = self.fetch_and_parse(url)
        soup = self.unwrap_comment_tables(soup)

        winners = []

        # Method 1: Look for conference tournament results table
        table = soup.find("table", id="conf-tourney")
        if table:
            rows = self.table_to_rows(table)
            for row in rows:
                team = row.get("school_name", row.get("winner", "")).strip()
                conf = row.get("conf_abbr", row.get("conf", "")).strip()
                school_id = ""
                for key in ["school_name_link", "winner_link"]:
                    if key in row:
                        match = re.search(r"/cbb/schools/([^/]+)/", row[key])
                        if match:
                            school_id = match.group(1)
                            break
                if team:
                    winners.append({
                        "team": team,
                        "school_id": school_id,
                        "season": season,
                        "conference": conf,
                        "conf_tourney_winner": 1,
                    })

        # Method 2: Parse from conference links on the season page
        if not winners:
            winners = self._parse_from_season_page(soup, season)

        # Method 3: Infer from tournament bracket (teams with seeds 14-16 from
        # small conferences are likely auto-bid winners)
        if not winners:
            print(f"  Warning: no conference tourney data found for {season}, "
                  "will infer from bracket")

        if not winners:
            return pd.DataFrame()

        df = pd.DataFrame(winners)
        df = df.drop_duplicates(subset=["team", "season"]).reset_index(drop=True)
        return df

    def _parse_from_season_page(self, soup, season: int) -> list[dict]:
        """Try to extract conference tourney winners from season overview page."""
        winners = []

        # Look for links or text mentioning conference tournament champions
        for header in soup.find_all(["h2", "h3"]):
            text = header.get_text(strip=True).lower()
            if "conference" in text and "tournament" in text:
                # Look at the following table or list
                next_table = header.find_next("table")
                if next_table:
                    rows = self.table_to_rows(next_table)
                    for row in rows:
                        team = row.get("school_name", row.get("champion", "")).strip()
                        conf = row.get("conf_abbr", row.get("conf", "")).strip()
                        school_id = ""
                        for key in row:
                            if "_link" in key and "/cbb/schools/" in row.get(key, ""):
                                match = re.search(r"/cbb/schools/([^/]+)/", row[key])
                                if match:
                                    school_id = match.group(1)
                                    break
                        if team:
                            winners.append({
                                "team": team,
                                "school_id": school_id,
                                "season": season,
                                "conference": conf,
                                "conf_tourney_winner": 1,
                            })

        return winners

    def scrape_all_seasons(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Scrape conference tournament data for all training seasons."""
        if seasons is None:
            seasons = TRAINING_SEASONS

        all_dfs = []
        for season in tqdm(seasons, desc="Scraping conference tournaments"):
            df = self.scrape_conference_winners(season)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)
