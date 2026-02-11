"""Scrape AP and Coaches poll data from Sports-Reference."""

import re

import pandas as pd
from tqdm import tqdm

from config import SR_POLLS_URL, TRAINING_SEASONS
from data.scraper_base import ScraperBase


class PollsScraper(ScraperBase):
    """Scrape AP and Coaches poll rankings."""

    def scrape_polls(self, season: int) -> pd.DataFrame:
        """Scrape final AP and Coaches poll rankings for a season.

        Returns DataFrame with: team, school_id, season, ap_rank, ap_weeks_ranked, coaches_rank
        """
        url = SR_POLLS_URL.format(season=season)
        soup = self.fetch_and_parse(url)
        soup = self.unwrap_comment_tables(soup)

        ap_df = self._parse_ap_poll(soup, season)
        coaches_df = self._parse_coaches_poll(soup, season)

        # Merge AP and coaches data
        if ap_df.empty and coaches_df.empty:
            print(f"  Warning: no poll data found for {season}")
            return pd.DataFrame()

        merge_key = ["school_id", "season"] if not ap_df.empty and "school_id" in ap_df.columns else ["team", "season"]

        if ap_df.empty:
            return coaches_df
        if coaches_df.empty:
            return ap_df

        coaches_merge_cols = [c for c in coaches_df.columns if c not in ap_df.columns or c in merge_key]
        df = ap_df.merge(coaches_df[coaches_merge_cols], on=merge_key, how="outer")
        return df

    def _parse_ap_poll(self, soup, season: int) -> pd.DataFrame:
        """Parse AP poll table."""
        table = soup.find("table", id="ap-poll")
        if table is None:
            # Try alternative IDs
            table = soup.find("table", id="polls")
            if table is None:
                return pd.DataFrame()

        rows = self.table_to_rows(table)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["season"] = season

        if "school_name" in df.columns:
            df = df.rename(columns={"school_name": "team"})
            df["team"] = df["team"].str.strip()

        if "school_name_link" in df.columns:
            df["school_id"] = df["school_name_link"].str.extract(r"/cbb/schools/([^/]+)/")

        # Find the final poll rank column (last week's ranking)
        # SR uses week numbers as column headers
        rank_cols = [c for c in df.columns if re.match(r"^\d+$", str(c))]
        if rank_cols:
            # Final ranking is the last week column
            final_col = rank_cols[-1]
            df["ap_rank"] = pd.to_numeric(df[final_col], errors="coerce")
        elif "ranker" in df.columns:
            df["ap_rank"] = pd.to_numeric(df.get("rank", pd.Series()), errors="coerce")

        # Count weeks ranked (number of non-empty week columns)
        if rank_cols:
            df["ap_weeks_ranked"] = df[rank_cols].apply(
                lambda row: sum(1 for v in row if str(v).strip() and str(v).strip() != "—"),
                axis=1
            )
        else:
            df["ap_weeks_ranked"] = 0

        keep_cols = ["team", "school_id", "season", "ap_rank", "ap_weeks_ranked"]
        return df[[c for c in keep_cols if c in df.columns]]

    def _parse_coaches_poll(self, soup, season: int) -> pd.DataFrame:
        """Parse Coaches poll table."""
        table = soup.find("table", id="coaches-poll")
        if table is None:
            return pd.DataFrame()

        rows = self.table_to_rows(table)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["season"] = season

        if "school_name" in df.columns:
            df = df.rename(columns={"school_name": "team"})
            df["team"] = df["team"].str.strip()

        if "school_name_link" in df.columns:
            df["school_id"] = df["school_name_link"].str.extract(r"/cbb/schools/([^/]+)/")

        # Get final ranking
        rank_cols = [c for c in df.columns if re.match(r"^\d+$", str(c))]
        if rank_cols:
            final_col = rank_cols[-1]
            df["coaches_rank"] = pd.to_numeric(df[final_col], errors="coerce")

        keep_cols = ["team", "school_id", "season", "coaches_rank"]
        return df[[c for c in keep_cols if c in df.columns]]

    def scrape_all_seasons(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Scrape poll data for all training seasons."""
        if seasons is None:
            seasons = TRAINING_SEASONS

        all_dfs = []
        for season in tqdm(seasons, desc="Scraping polls"):
            df = self.scrape_polls(season)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)
