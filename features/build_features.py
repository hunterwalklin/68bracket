"""Merge all data sources and engineer features for modeling."""

import os

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DIR, POWER_CONFERENCES, ALL_FEATURES,
    BASIC_STAT_FEATURES, ADVANCED_STAT_FEATURES,
    TRAINING_SEASONS, PREDICTION_SEASON,
)
from features.team_names import merge_on_team


def compute_conference_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Compute conference-level aggregate strength metric.

    Uses mean SRS of teams in each conference as a proxy for conference strength.
    """
    if "conference" not in df.columns or "srs" not in df.columns:
        df["conf_strength"] = 0.0
        return df

    conf_avg = (
        df.groupby(["conference", "season"])["srs"]
        .mean()
        .reset_index()
        .rename(columns={"srs": "conf_strength"})
    )
    df = df.merge(conf_avg, on=["conference", "season"], how="left")
    df["conf_strength"] = df["conf_strength"].fillna(0.0)
    return df


def add_power_conference_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flag for teams in power conferences."""
    if "conference" not in df.columns:
        df["power_conf"] = 0
        return df

    df["power_conf"] = df["conference"].isin(POWER_CONFERENCES).astype(int)
    return df


def build_features(
    team_stats: pd.DataFrame,
    tournament: pd.DataFrame,
    polls: pd.DataFrame,
    conferences: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources and build the full feature matrix.

    Args:
        team_stats: Basic + advanced stats per team per season
        tournament: Historical bracket data (seeds) - training labels
        polls: AP and Coaches poll data
        conferences: Conference tournament winners

    Returns:
        DataFrame with all features and target columns (made_tournament, seed)
    """
    df = team_stats.copy()

    # Merge poll data
    if not polls.empty:
        df = merge_on_team(df, polls, season_col="season", how="left")
        # Fill unranked teams
        df["ap_rank"] = df.get("ap_rank", pd.Series(dtype=float)).fillna(0)
        df["ap_weeks_ranked"] = df.get("ap_weeks_ranked", pd.Series(dtype=float)).fillna(0)
        df["coaches_rank"] = df.get("coaches_rank", pd.Series(dtype=float)).fillna(0)
    else:
        df["ap_rank"] = 0
        df["ap_weeks_ranked"] = 0
        df["coaches_rank"] = 0

    # Merge conference tournament winners
    if not conferences.empty:
        df = merge_on_team(df, conferences[["team", "school_id", "season", "conf_tourney_winner"]],
                           season_col="season", how="left")
        df["conf_tourney_winner"] = df.get("conf_tourney_winner", pd.Series(dtype=float)).fillna(0).astype(int)
    else:
        df["conf_tourney_winner"] = 0

    # Add conference features
    df = add_power_conference_flag(df)
    df = compute_conference_strength(df)

    # Add tournament labels (target variables)
    if not tournament.empty:
        tournament_labeled = tournament[["school_id", "season", "seed"]].copy()
        tournament_labeled["made_tournament"] = 1

        if "school_id" in df.columns:
            df = df.merge(tournament_labeled, on=["school_id", "season"], how="left")
        else:
            df = merge_on_team(df, tournament_labeled, season_col="season", how="left")

        df["made_tournament"] = df.get("made_tournament", pd.Series(dtype=float)).fillna(0).astype(int)
        df["seed"] = df.get("seed", pd.Series(dtype=float))
    else:
        df["made_tournament"] = 0
        df["seed"] = np.nan

    # Ensure all expected feature columns exist
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    # Convert feature columns to numeric
    for feat in ALL_FEATURES:
        df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)

    return df


def save_features(df: pd.DataFrame, filename: str = "features.parquet"):
    """Save the feature matrix to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_parquet(path, index=False)
    print(f"  Saved features to {path} ({len(df)} rows)")
    return path


def load_features(filename: str = "features.parquet") -> pd.DataFrame:
    """Load the feature matrix from disk."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    return pd.read_parquet(path)


def get_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract training data from the feature matrix.

    Returns:
        X: Feature matrix (all seasons in TRAINING_SEASONS)
        y_selection: Binary target for tournament selection
        y_seed: Seed target (NaN for non-tournament teams)
    """
    train_mask = df["season"].isin(TRAINING_SEASONS)
    train = df[train_mask].copy()

    X = train[ALL_FEATURES]
    y_selection = train["made_tournament"]
    y_seed = train["seed"]

    return X, y_selection, y_seed


def get_prediction_data(df: pd.DataFrame, season: int = PREDICTION_SEASON) -> pd.DataFrame:
    """Extract prediction-season data from the feature matrix."""
    return df[df["season"] == season].copy()
