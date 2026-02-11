"""Scrape historical NCAA tournament brackets for training labels."""

import re

import pandas as pd
from tqdm import tqdm

from config import SR_TOURNAMENT_URL, TRAINING_SEASONS
from data.scraper_base import ScraperBase


class TournamentScraper(ScraperBase):
    """Scrape NCAA tournament bracket data (seeds and teams) from Sports-Reference."""

    def scrape_bracket(self, season: int) -> pd.DataFrame:
        """Scrape tournament bracket for a single season.

        Returns DataFrame with columns: team, seed, season, school_id, region.
        """
        url = SR_TOURNAMENT_URL.format(season=season)
        soup = self.fetch_and_parse(url)

        # The bracket page uses a div-based layout, not a standard table
        # Look for seed-team pairs in the bracket
        bracket_data = []

        # Method 1: Look for the bracket div structure
        # SR uses <span class="seed"> and <a> for team names
        bracket_div = soup.find("div", id="bracket")
        if bracket_div:
            bracket_data = self._parse_bracket_div(bracket_div, season)

        # Method 2: Fallback - look for a table with team seeds
        if not bracket_data:
            bracket_data = self._parse_bracket_from_links(soup, season)

        if not bracket_data:
            print(f"  Warning: no bracket data found for {season}")
            return pd.DataFrame()

        df = pd.DataFrame(bracket_data)

        # Deduplicate (team might appear in multiple rounds)
        df = df.drop_duplicates(subset=["team", "season"]).reset_index(drop=True)

        return df

    def _parse_bracket_div(self, bracket_div, season: int) -> list[dict]:
        """Parse the bracket div structure from Sports-Reference."""
        data = []
        seen_teams = set()

        # Find all rounds
        for round_div in bracket_div.find_all("div", class_=re.compile(r"round")):
            region_text = ""
            # Try to get region from parent or sibling
            region_header = round_div.find_previous("h3")
            if region_header:
                region_text = region_header.get_text(strip=True)

            for game in round_div.find_all("div"):
                seeds = game.find_all("span", class_="seed")
                links = game.find_all("a")

                for seed_span, link in zip(seeds, links):
                    seed_text = seed_span.get_text(strip=True)
                    try:
                        seed = int(seed_text)
                    except (ValueError, TypeError):
                        continue

                    team_name = link.get_text(strip=True)
                    href = link.get("href", "")
                    school_id_match = re.search(r"/cbb/schools/([^/]+)/", href)
                    school_id = school_id_match.group(1) if school_id_match else ""

                    if team_name and team_name not in seen_teams:
                        seen_teams.add(team_name)
                        data.append({
                            "team": team_name,
                            "seed": seed,
                            "season": season,
                            "school_id": school_id,
                            "region": region_text,
                        })

        return data

    def _parse_bracket_from_links(self, soup, season: int) -> list[dict]:
        """Fallback: parse bracket data from links on the page."""
        data = []
        seen_teams = set()

        # Look for pattern: seed number followed by team link
        # SR format: <span class="seed">1</span> <a href="/cbb/schools/...">TeamName</a>
        for seed_span in soup.find_all("span"):
            seed_text = seed_span.get_text(strip=True)
            try:
                seed = int(seed_text)
                if seed < 1 or seed > 16:
                    continue
            except (ValueError, TypeError):
                continue

            # Find the next sibling or nearby link
            link = seed_span.find_next("a")
            if link is None:
                continue

            href = link.get("href", "")
            if "/cbb/schools/" not in href:
                continue

            team_name = link.get_text(strip=True)
            school_id_match = re.search(r"/cbb/schools/([^/]+)/", href)
            school_id = school_id_match.group(1) if school_id_match else ""

            if team_name and team_name not in seen_teams:
                seen_teams.add(team_name)
                data.append({
                    "team": team_name,
                    "seed": seed,
                    "season": season,
                    "school_id": school_id,
                    "region": "",
                })

        return data

    def scrape_all_seasons(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Scrape bracket data for all training seasons."""
        if seasons is None:
            seasons = TRAINING_SEASONS

        all_dfs = []
        for season in tqdm(seasons, desc="Scraping tournament brackets"):
            df = self.scrape_bracket(season)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        print(f"  Total tournament entries: {len(result)} across {len(all_dfs)} seasons")
        return result
