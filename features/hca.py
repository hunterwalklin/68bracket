"""KenPom-style Home Court Advantage calculation from ESPN box scores.

Per KenPom's research, fouls drive HCA more than any other box score stat.
This module computes home/road splits for fouls, turnovers, blocks, and
scoring margin to produce a composite HCA score.

Each component is z-score normalized within the season before weighting,
so that fouls (range ~0-9) aren't drowned out by scoring (range ~0-35).
A quality multiplier prevents bad teams with lopsided schedules from
dominating — HCA is scaled by the team's overall strength percentile.

Components & weights:
- Foul advantage (40%)  — opponent fouls minus own fouls, home vs road
- Scoring advantage (40%) — point margin, home vs road
- Turnover advantage (15%) — non-steal turnovers, home vs road
- Block advantage (5%) — blocks, home vs road
"""

import numpy as np
import pandas as pd


# Component weights (sum to 1.0)
FOUL_WEIGHT = 0.40
SCORING_WEIGHT = 0.40
TURNOVER_WEIGHT = 0.15
BLOCK_WEIGHT = 0.05

# Minimum games required
MIN_HOME_GAMES = 5
MIN_ROAD_GAMES = 4

# League average HCA in points
LEAGUE_AVG_HCA_PTS = 3.5

# KenPom-calibrated HCA range: best ≈ 4.5, worst ≈ 2.5 (2018 data)
# hca_points = HCA_PTS_MIN + hca_score * (HCA_PTS_MAX - HCA_PTS_MIN)
HCA_PTS_MIN = 2.5
HCA_PTS_MAX = 4.5

# Quality blending: raw_hca * (QUALITY_FLOOR + quality_pct * (1 - QUALITY_FLOOR))
# Floor of 0.3 means even a bottom-tier team keeps 30% of its raw HCA;
# a top-tier team gets 100%.
QUALITY_FLOOR = 0.3


def build_hca_features(espn_box: pd.DataFrame,
                       net_rankings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute KenPom-style HCA features from ESPN box score data.

    Args:
        espn_box: DataFrame with columns: game_id, date, season, school_id,
                  espn_id, opponent_espn_id, home_away, points, opp_points,
                  fouls, opp_fouls, turnovers, opp_turnovers, steals,
                  opp_steals, blocks, opp_blocks
        net_rankings: Optional DataFrame with school_id, season, net_ranking
                      for quality weighting. If None, quality multiplier is 1.0.

    Returns:
        DataFrame with columns: school_id, season, hca_score, hca_points,
        foul_advantage, scoring_advantage, turnover_advantage, block_advantage,
        home_pts_margin, road_pts_margin, home_foul_margin, road_foul_margin
    """
    if espn_box.empty:
        return pd.DataFrame()

    df = espn_box.copy()

    # Filter to rows with a valid school_id mapping
    df = df[df["school_id"].astype(str).str.len() > 0].copy()

    # Compute per-game margins
    df["pts_margin"] = df["points"] - df["opp_points"]
    df["foul_margin"] = df["opp_fouls"] - df["fouls"]  # positive = opponent fouled more
    df["to_margin"] = df["opp_turnovers"] - df["turnovers"]
    df["blk_margin"] = df["blocks"] - df["opp_blocks"]
    df["is_win"] = (df["points"] > df["opp_points"]).astype(int)

    # Split home and road
    home = df[df["home_away"] == "home"]
    road = df[df["home_away"] == "road"]

    # Aggregate per team per season
    home_agg = (
        home.groupby(["school_id", "season"])
        .agg(
            home_games=("game_id", "count"),
            espn_home_wins=("is_win", "sum"),
            home_pts_margin=("pts_margin", "mean"),
            home_foul_margin=("foul_margin", "mean"),
            home_to_margin=("to_margin", "mean"),
            home_blk_margin=("blk_margin", "mean"),
        )
        .reset_index()
    )

    road_agg = (
        road.groupby(["school_id", "season"])
        .agg(
            road_games=("game_id", "count"),
            espn_road_wins=("is_win", "sum"),
            road_pts_margin=("pts_margin", "mean"),
            road_foul_margin=("foul_margin", "mean"),
            road_to_margin=("to_margin", "mean"),
            road_blk_margin=("blk_margin", "mean"),
        )
        .reset_index()
    )

    # Merge home and road
    merged = home_agg.merge(road_agg, on=["school_id", "season"], how="outer")

    # Fill NaN game counts with 0
    merged["home_games"] = merged["home_games"].fillna(0).astype(int)
    merged["road_games"] = merged["road_games"].fillna(0).astype(int)
    merged["espn_home_wins"] = merged["espn_home_wins"].fillna(0).astype(int)
    merged["espn_road_wins"] = merged["espn_road_wins"].fillna(0).astype(int)
    merged["espn_home_losses"] = merged["home_games"] - merged["espn_home_wins"]
    merged["espn_road_losses"] = merged["road_games"] - merged["espn_road_wins"]

    # Compute home court advantages (home margin - road margin)
    # Positive = team performs better at home
    merged["foul_advantage"] = merged["home_foul_margin"].fillna(0) - merged["road_foul_margin"].fillna(0)
    merged["scoring_advantage"] = merged["home_pts_margin"].fillna(0) - merged["road_pts_margin"].fillna(0)
    merged["turnover_advantage"] = merged["home_to_margin"].fillna(0) - merged["road_to_margin"].fillna(0)
    merged["block_advantage"] = merged["home_blk_margin"].fillna(0) - merged["road_blk_margin"].fillna(0)

    # Apply minimums: teams without enough games get defaults
    below_threshold = (
        (merged["home_games"] < MIN_HOME_GAMES)
        | (merged["road_games"] < MIN_ROAD_GAMES)
    )

    # Merge quality data (NET rankings) if provided
    if net_rankings is not None and not net_rankings.empty:
        nr = net_rankings[["school_id", "season", "net_ranking"]].copy()
        nr["net_ranking"] = pd.to_numeric(nr["net_ranking"], errors="coerce")
        merged = merged.merge(nr, on=["school_id", "season"], how="left")
    else:
        merged["net_ranking"] = np.nan

    # Compute weighted composite per season
    results = []
    for season in merged["season"].unique():
        season_mask = merged["season"] == season
        season_df = merged[season_mask].copy()

        # Apply threshold — below minimum gets NaN for ranking purposes
        threshold_mask = below_threshold[season_mask]

        # Z-score normalize each component within the season (qualified only)
        components = ["foul_advantage", "scoring_advantage",
                      "turnover_advantage", "block_advantage"]
        z_cols = []
        for comp in components:
            z_col = f"{comp}_z"
            z_cols.append(z_col)
            vals = season_df[comp].copy()
            vals[threshold_mask] = np.nan
            qualified = vals.dropna()
            if len(qualified) > 1:
                mean, std = qualified.mean(), qualified.std()
                if std > 0:
                    season_df[z_col] = (vals - mean) / std
                else:
                    season_df[z_col] = 0.0
            else:
                season_df[z_col] = 0.0

        # Weighted composite of z-scored components
        season_df["raw_hca"] = (
            season_df["foul_advantage_z"] * FOUL_WEIGHT
            + season_df["scoring_advantage_z"] * SCORING_WEIGHT
            + season_df["turnover_advantage_z"] * TURNOVER_WEIGHT
            + season_df["block_advantage_z"] * BLOCK_WEIGHT
        )
        season_df.loc[threshold_mask, "raw_hca"] = np.nan

        # Quality multiplier: scale raw HCA by team quality
        # Teams ranked 1 get multiplier ~1.0; teams ranked 365 get ~QUALITY_FLOOR
        if season_df["net_ranking"].notna().any():
            max_rank = season_df["net_ranking"].max()
            if max_rank > 0:
                quality_pct = 1.0 - (season_df["net_ranking"] - 1) / max(max_rank - 1, 1)
                quality_pct = quality_pct.clip(0, 1).fillna(0.5)
            else:
                quality_pct = 0.5
            season_df["quality_mult"] = QUALITY_FLOOR + quality_pct * (1 - QUALITY_FLOOR)
        else:
            season_df["quality_mult"] = 1.0

        season_df["adjusted_hca"] = season_df["raw_hca"] * season_df["quality_mult"]
        season_df.loc[threshold_mask, "adjusted_hca"] = np.nan

        # Percentile rank among qualified teams (0.0-1.0)
        qualified = season_df["adjusted_hca"].dropna()
        if len(qualified) > 1:
            season_df["hca_score"] = season_df["adjusted_hca"].rank(pct=True)
        else:
            season_df["hca_score"] = 0.5

        # HCA points: linear map from percentile to KenPom-calibrated range
        season_df["hca_points"] = HCA_PTS_MIN + season_df["hca_score"] * (HCA_PTS_MAX - HCA_PTS_MIN)

        # Fill below-threshold teams with defaults
        season_df.loc[threshold_mask, "hca_score"] = 0.0
        season_df.loc[threshold_mask, "hca_points"] = LEAGUE_AVG_HCA_PTS
        season_df.loc[threshold_mask, "foul_advantage"] = 0.0
        season_df.loc[threshold_mask, "scoring_advantage"] = 0.0
        season_df.loc[threshold_mask, "turnover_advantage"] = 0.0
        season_df.loc[threshold_mask, "block_advantage"] = 0.0

        results.append(season_df)

    if not results:
        return pd.DataFrame()

    result = pd.concat(results, ignore_index=True)

    # Select output columns
    output_cols = [
        "school_id", "season",
        "hca_score", "hca_points",
        "foul_advantage", "scoring_advantage", "turnover_advantage", "block_advantage",
        "home_pts_margin", "road_pts_margin", "home_foul_margin", "road_foul_margin",
        "home_games", "road_games",
        "espn_home_wins", "espn_home_losses", "espn_road_wins", "espn_road_losses",
    ]
    # Ensure all output columns exist
    for col in output_cols:
        if col not in result.columns:
            result[col] = 0.0

    return result[output_cols].copy()
