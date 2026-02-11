"""Scrape team stats from Sports-Reference: basic, advanced, and ratings."""

import re

import pandas as pd
from tqdm import tqdm

from config import SR_SEASONS_URL, SR_ADVANCED_URL, SR_RATINGS_URL, TRAINING_SEASONS
from data.scraper_base import ScraperBase


class TeamStatsScraper(ScraperBase):
    """Scrape basic stats, advanced stats, and ratings for all D1 teams."""

    def scrape_basic_stats(self, season: int) -> pd.DataFrame:
        """Scrape basic school stats (wins, losses, pts, SRS, SOS, etc.)."""
        url = SR_SEASONS_URL.format(season=season)
        soup = self.fetch_and_parse(url)
        soup = self.unwrap_comment_tables(soup)

        table = soup.find("table", id="basic_school_stats")
        if table is None:
            print(f"  Warning: basic_school_stats table not found for {season}")
            return pd.DataFrame()

        rows = self.table_to_rows(table)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["season"] = season

        # Extract school name from the 'school_name' column
        rename_map = {
            "school_name": "team",
            "wins": "wins",
            "losses": "losses",
            "win_loss_pct": "win_pct",
            "srs": "srs",
            "sos": "sos",
            "conf_wins": "conf_wins",
            "conf_losses": "conf_losses",
            "pts_per_g": "pts_per_game",
            "opp_pts_per_g": "opp_pts_per_game",
        }

        # Only rename columns that exist
        existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=existing_renames)

        # Clean team names: remove NCAA/NAIA tournament markers
        if "team" in df.columns:
            df["team"] = df["team"].str.replace(r"\s*NCAA$", "", regex=True)
            df["team"] = df["team"].str.strip()

        # Extract school link for matching
        if "school_name_link" in df.columns:
            df["school_id"] = df["school_name_link"].str.extract(r"/cbb/schools/([^/]+)/")

        # Convert numeric columns
        numeric_cols = ["wins", "losses", "win_pct", "srs", "sos",
                        "conf_wins", "conf_losses", "pts_per_game", "opp_pts_per_game"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute conf_win_pct
        if "conf_wins" in df.columns and "conf_losses" in df.columns:
            total = df["conf_wins"] + df["conf_losses"]
            df["conf_win_pct"] = (df["conf_wins"] / total).where(total > 0, 0.0)

        # Filter out empty/header rows
        df = df.dropna(subset=["team"]).query("team != ''")

        return df

    def scrape_advanced_stats(self, season: int) -> pd.DataFrame:
        """Scrape advanced school stats (ORtg, DRtg, pace, four factors, etc.)."""
        url = SR_ADVANCED_URL.format(season=season)
        soup = self.fetch_and_parse(url)
        soup = self.unwrap_comment_tables(soup)

        table = soup.find("table", id="adv_school_stats")
        if table is None:
            print(f"  Warning: adv_school_stats table not found for {season}")
            return pd.DataFrame()

        rows = self.table_to_rows(table)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["season"] = season

        rename_map = {
            "school_name": "team",
            "pace": "pace",
            "off_rtg": "ortg",
            "def_rtg": "drtg",
            "efg_pct": "efg_pct",
            "tov_pct": "tov_pct",
            "orb_pct": "orb_pct",
            "ft_rate": "ft_rate",
            "opp_efg_pct": "opp_efg_pct",
            "opp_tov_pct": "opp_tov_pct",
            "opp_orb_pct": "opp_orb_pct",
            "opp_ft_rate": "opp_ft_rate",
        }

        existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=existing_renames)

        if "team" in df.columns:
            df["team"] = df["team"].str.replace(r"\s*NCAA$", "", regex=True)
            df["team"] = df["team"].str.strip()

        if "school_name_link" in df.columns:
            df["school_id"] = df["school_name_link"].str.extract(r"/cbb/schools/([^/]+)/")

        # Convert numeric columns
        numeric_cols = ["pace", "ortg", "drtg", "efg_pct", "tov_pct", "orb_pct",
                        "ft_rate", "opp_efg_pct", "opp_tov_pct", "opp_orb_pct", "opp_ft_rate"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute net rating
        if "ortg" in df.columns and "drtg" in df.columns:
            df["nrtg"] = df["ortg"] - df["drtg"]

        df = df.dropna(subset=["team"]).query("team != ''")
        return df

    def scrape_ratings(self, season: int) -> pd.DataFrame:
        """Scrape preseason/final ratings (AP, Coaches polls integrated here if available)."""
        url = SR_RATINGS_URL.format(season=season)
        soup = self.fetch_and_parse(url)
        soup = self.unwrap_comment_tables(soup)

        table = soup.find("table", id="ratings")
        if table is None:
            print(f"  Warning: ratings table not found for {season}")
            return pd.DataFrame()

        rows = self.table_to_rows(table)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["season"] = season

        if "school_name" in df.columns:
            df = df.rename(columns={"school_name": "team"})
            df["team"] = df["team"].str.replace(r"\s*NCAA$", "", regex=True)
            df["team"] = df["team"].str.strip()

        if "school_name_link" in df.columns:
            df["school_id"] = df["school_name_link"].str.extract(r"/cbb/schools/([^/]+)/")

        # Extract conference if present
        if "conf_abbr" in df.columns:
            df = df.rename(columns={"conf_abbr": "conference"})

        df = df.dropna(subset=["team"]).query("team != ''")
        return df

    def scrape_season(self, season: int) -> pd.DataFrame:
        """Scrape all stats for a single season and merge."""
        basic = self.scrape_basic_stats(season)
        advanced = self.scrape_advanced_stats(season)
        ratings = self.scrape_ratings(season)

        if basic.empty:
            return pd.DataFrame()

        # Merge on school_id + season (most reliable key)
        merge_key = ["school_id", "season"] if "school_id" in basic.columns else ["team", "season"]

        df = basic.copy()
        if not advanced.empty:
            adv_cols = [c for c in advanced.columns if c not in basic.columns or c in merge_key]
            df = df.merge(advanced[adv_cols], on=merge_key, how="left")

        if not ratings.empty:
            rat_cols = [c for c in ratings.columns if c not in df.columns or c in merge_key]
            df = df.merge(ratings[rat_cols], on=merge_key, how="left")

        return df

    def scrape_all_seasons(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Scrape stats for all training seasons."""
        if seasons is None:
            seasons = TRAINING_SEASONS

        all_dfs = []
        for season in tqdm(seasons, desc="Scraping team stats"):
            try:
                df = self.scrape_season(season)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"\n  Failed season {season}: {e}. Keeping data from other seasons.")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)
