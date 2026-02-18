"""Merge all data sources and engineer features for modeling."""

import os

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DIR, ALL_FEATURES, COMMITTEE_FEATURES,
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


def build_features(
    team_stats: pd.DataFrame,
    tournament: pd.DataFrame,
    nitty_gritty: pd.DataFrame,
    torvik: pd.DataFrame,
    conferences: pd.DataFrame,
    espn_box: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all data sources and build the full feature matrix.

    Args:
        team_stats: Basic + advanced stats per team per season
        tournament: Historical bracket data (seeds) - training labels
        nitty_gritty: WarrenNolan data (NET ranking, KPI, SOR, BPI, POM rankings)
        torvik: Bart Torvik data (WAB, Barthag, AdjOE, AdjDE, quadrant records)
        conferences: Conference tournament winners
        espn_box: ESPN box score data for KenPom-style HCA (optional, display-only)

    Returns:
        DataFrame with all features and target columns (made_tournament, seed)
    """
    df = team_stats.copy()

    # Merge WarrenNolan rankings (NET, KPI, SOR, BPI, POM)
    if not nitty_gritty.empty:
        df = merge_on_team(df, nitty_gritty, season_col="season", how="left")

    # Merge Torvik data (WAB, Barthag, AdjOE, AdjDE, quadrant records)
    # Drop overlapping columns from WarrenNolan so Torvik's fuller-coverage data wins
    if not torvik.empty:
        overlap_cols = [c for c in torvik.columns
                        if c in df.columns and c not in ("team", "school_id", "season")]
        if overlap_cols:
            df = df.drop(columns=overlap_cols)
        df = merge_on_team(df, torvik, season_col="season", how="left")

    # Fill missing committee features with 0
    for feat in COMMITTEE_FEATURES:
        df[feat] = df.get(feat, pd.Series(dtype=float)).fillna(0.0)

    # Merge conference tournament winners
    if not conferences.empty:
        df = merge_on_team(df, conferences[["team", "school_id", "season", "conf_tourney_winner"]],
                           season_col="season", how="left")
        df["conf_tourney_winner"] = df.get("conf_tourney_winner", pd.Series(dtype=float)).fillna(0).astype(int)
    else:
        df["conf_tourney_winner"] = 0

    # Add conference features
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

    # Merge KenPom-style HCA features (display-only — NOT added to ALL_FEATURES)
    if espn_box is not None and not espn_box.empty:
        from features.hca import build_hca_features
        # Pass NET rankings for quality weighting
        net_df = df[["school_id", "season", "net_ranking"]].copy() if "net_ranking" in df.columns else None
        hca_df = build_hca_features(espn_box, net_rankings=net_df)
        if not hca_df.empty and "school_id" in df.columns:
            hca_cols = [c for c in hca_df.columns if c not in ("school_id", "season")]
            # Drop any pre-existing HCA columns to avoid _x/_y suffixes
            existing_hca = [c for c in hca_cols if c in df.columns]
            if existing_hca:
                df = df.drop(columns=existing_hca)
            df = df.merge(hca_df, on=["school_id", "season"], how="left")
            # Fill missing HCA values with defaults
            df["hca_score"] = df.get("hca_score", pd.Series(dtype=float)).fillna(0.0)
            df["hca_points"] = df.get("hca_points", pd.Series(dtype=float)).fillna(3.5)
            for col in ["foul_advantage", "scoring_advantage",
                        "turnover_advantage", "block_advantage",
                        "espn_home_wins", "espn_home_losses",
                        "espn_road_wins", "espn_road_losses",
                        "home_games", "road_games"]:
                df[col] = df.get(col, pd.Series(dtype=float)).fillna(0.0)

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
