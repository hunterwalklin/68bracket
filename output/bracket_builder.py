"""Build bracket structure: assign regions, format matchups."""

import numpy as np
import pandas as pd

from config import REGIONS, SEEDS, TEAMS_PER_SEED


def assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign the 68 teams to 4 regions.

    Rules:
    - Each region gets one team per seed line (1-16)
    - First Four teams (extra at seeds 11 and 16) are placed as play-in games
    - Top 4 overall seeds go to different regions as 1-seeds

    Args:
        df: DataFrame with predicted_seed column, sorted by raw_seed

    Returns:
        DataFrame with 'region' column added
    """
    df = df.copy()
    df["region"] = ""
    df["first_four"] = False

    # Group by predicted seed
    for seed in SEEDS:
        seed_teams = df[df["predicted_seed"] == seed].copy()
        n_expected = TEAMS_PER_SEED[seed]

        if len(seed_teams) == 0:
            continue

        # Standard seeds (4 teams): one per region
        if n_expected == 4:
            # Sort by raw_seed or selection_prob for within-seed ordering
            sort_col = "raw_seed" if "raw_seed" in seed_teams.columns else "selection_prob"
            ascending = True if sort_col == "raw_seed" else False
            seed_teams = seed_teams.sort_values(sort_col, ascending=ascending)

            for i, (idx, _) in enumerate(seed_teams.iterrows()):
                region = REGIONS[i % 4]
                df.loc[idx, "region"] = region

        # First Four seeds (6 teams): 4 in bracket + 2 play-in
        elif n_expected == 6:
            sort_col = "raw_seed" if "raw_seed" in seed_teams.columns else "selection_prob"
            ascending = True if sort_col == "raw_seed" else False
            seed_teams = seed_teams.sort_values(sort_col, ascending=ascending)

            indices = seed_teams.index.tolist()
            # First 4 go directly to regions
            for i in range(min(4, len(indices))):
                df.loc[indices[i], "region"] = REGIONS[i]
            # Remaining are First Four play-in games
            for i in range(4, len(indices)):
                df.loc[indices[i], "region"] = REGIONS[i - 4]  # paired with same region
                df.loc[indices[i], "first_four"] = True

    return df


def build_matchups(df: pd.DataFrame) -> list[dict]:
    """Build first-round matchups from seeded bracket.

    Standard NCAA bracket matchups by seed:
    1 vs 16, 2 vs 15, 3 vs 14, 4 vs 13,
    5 vs 12, 6 vs 11, 7 vs 10, 8 vs 9
    """
    matchup_pairs = [
        (1, 16), (8, 9), (5, 12), (4, 13),
        (6, 11), (3, 14), (7, 10), (2, 15),
    ]

    matchups = []
    for region in REGIONS:
        region_teams = df[df["region"] == region]

        for high_seed, low_seed in matchup_pairs:
            high = region_teams[
                (region_teams["predicted_seed"] == high_seed) &
                (~region_teams["first_four"])
            ]
            low = region_teams[
                (region_teams["predicted_seed"] == low_seed) &
                (~region_teams["first_four"])
            ]

            high_name = high.iloc[0]["team"] if not high.empty else "TBD"
            low_name = low.iloc[0]["team"] if not low.empty else "TBD"

            matchups.append({
                "region": region,
                "high_seed": high_seed,
                "low_seed": low_seed,
                "high_team": high_name,
                "low_team": low_name,
            })

    return matchups


def build_first_four(df: pd.DataFrame) -> list[dict]:
    """Build First Four play-in matchups."""
    ff_teams = df[df["first_four"]].copy()
    games = []

    for seed in [11, 16]:
        seed_ff = ff_teams[ff_teams["predicted_seed"] == seed]
        if len(seed_ff) >= 2:
            teams = seed_ff.sort_values("raw_seed" if "raw_seed" in seed_ff.columns else "selection_prob")
            for i in range(0, len(teams) - 1, 2):
                games.append({
                    "seed": seed,
                    "team1": teams.iloc[i]["team"],
                    "team2": teams.iloc[i + 1]["team"],
                    "region": teams.iloc[i]["region"],
                })

    return games
