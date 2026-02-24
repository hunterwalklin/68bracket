"""KenPom-style Home Court Advantage calculation from ESPN box scores.

Per KenPom's research, fouls drive HCA more than any other box score stat.
This module computes home/road splits for fouls, turnovers, steals, and
scoring margin to produce a composite HCA score.

Each component is z-score normalized within the season before weighting,
so that fouls (range ~0-9) aren't drowned out by scoring (range ~0-35).

Multi-season smoothing (exponential decay, up to 6 seasons) reduces noise
from single-season samples (~15 home games). Travel distance provides an
additional adjustment based on opponent distance.

Components & weights:
- Foul advantage (55%)  — opponent fouls minus own fouls, home vs road
- Scoring advantage (25%) — point margin, home vs road
- Turnover advantage (15%) — non-steal turnovers, home vs road
- Other/steals advantage (5%) — steals, home vs road
"""

import json
import math
import os

import numpy as np
import pandas as pd

from config import DATA_DIR


# Component weights (sum to 1.0)
FOUL_WEIGHT = 0.55
SCORING_WEIGHT = 0.25
TURNOVER_WEIGHT = 0.15
OTHER_WEIGHT = 0.05

# Minimum games required
MIN_HOME_GAMES = 5
MIN_ROAD_GAMES = 4

# League average HCA in points
LEAGUE_AVG_HCA_PTS = 3.2

# Narrowed HCA range: best ≈ 3.8, worst ≈ 2.5 (research-calibrated)
HCA_PTS_MIN = 2.5
HCA_PTS_MAX = 3.8

# Multi-season smoothing
SEASON_DECAY = 0.5       # weight = SEASON_DECAY ^ years_back
MAX_PRIOR_SEASONS = 5    # blend up to 6 seasons total (current + 5 prior)

# Travel distance
TRAVEL_PTS_PER_1000MI = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/lon points."""
    R = 3959  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_team_locations():
    """Load team_locations.json, normalize keys to school_id format.

    Returns:
        dict: school_id -> {"lat": float, "lon": float}
    """
    from features.team_names import normalize_team_name

    path = os.path.join(DATA_DIR, "team_locations.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    locations = {}
    for name, coords in raw.items():
        sid = normalize_team_name(name)
        if sid:
            locations[sid] = coords
    return locations


def _load_espn_to_school():
    """Load espn_logos.json to build espn_id -> school_id reverse mapping.

    Returns:
        dict: espn_id (str) -> school_id (str)
    """
    path = os.path.join(DATA_DIR, "espn_logos.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        logos = json.load(f)
    # espn_logos.json: school_id_slug -> espn_id
    return {str(v): k for k, v in logos.items()}


def _compute_travel_distances(espn_box: pd.DataFrame) -> pd.DataFrame:
    """Compute average opponent travel distance for each team's home games.

    For each team's home games, computes the great-circle distance from
    the opponent's campus to the team's campus.

    Returns:
        DataFrame with columns: school_id, season, avg_opp_travel_mi
    """
    locations = _load_team_locations()
    espn_to_school = _load_espn_to_school()

    if not locations or not espn_to_school:
        return pd.DataFrame(columns=["school_id", "season", "avg_opp_travel_mi"])

    home_games = espn_box[espn_box["home_away"] == "home"].copy()
    if home_games.empty:
        return pd.DataFrame(columns=["school_id", "season", "avg_opp_travel_mi"])

    # Resolve opponent school_id from opponent_espn_id
    home_games["opp_school_id"] = home_games["opponent_espn_id"].astype(str).map(espn_to_school)

    records = []
    for (school_id, season), group in home_games.groupby(["school_id", "season"]):
        home_loc = locations.get(school_id)
        if not home_loc:
            continue
        distances = []
        for _, row in group.iterrows():
            opp_sid = row.get("opp_school_id")
            opp_loc = locations.get(opp_sid) if opp_sid else None
            if opp_loc:
                d = _haversine(opp_loc["lat"], opp_loc["lon"],
                               home_loc["lat"], home_loc["lon"])
                distances.append(d)
        if distances:
            records.append({
                "school_id": school_id,
                "season": season,
                "avg_opp_travel_mi": np.mean(distances),
            })

    if not records:
        return pd.DataFrame(columns=["school_id", "season", "avg_opp_travel_mi"])
    return pd.DataFrame(records)


def _smooth_across_seasons(per_season_df: pd.DataFrame, target_season: int,
                           components: list[str]) -> pd.DataFrame:
    """Exponential-decay blend of component values across up to 6 seasons.

    Weight for each prior season = SEASON_DECAY ^ years_back.
    Handles the COVID gap (2020 season missing).

    Args:
        per_season_df: DataFrame with school_id, season, and component columns
        target_season: The season to produce smoothed values for
        components: List of column names to smooth

    Returns:
        DataFrame with school_id, smoothed component columns, seasons_blended count
        (only teams present in target_season)
    """
    # Seasons to consider: target_season and up to MAX_PRIOR_SEASONS before it
    candidate_seasons = []
    for offset in range(MAX_PRIOR_SEASONS + 1):
        s = target_season - offset
        if s == 2020:
            continue  # COVID gap — no data
        candidate_seasons.append(s)
        if len(candidate_seasons) > MAX_PRIOR_SEASONS + 1:
            break

    # Get teams that exist in the target season
    target_teams = per_season_df[per_season_df["season"] == target_season]["school_id"].unique()
    if len(target_teams) == 0:
        return pd.DataFrame(columns=["school_id"] + components + ["seasons_blended"])

    result_rows = []
    for sid in target_teams:
        team_data = per_season_df[per_season_df["school_id"] == sid]
        weighted_sums = {c: 0.0 for c in components}
        total_weight = 0.0
        seasons_found = 0
        for s in candidate_seasons:
            row = team_data[team_data["season"] == s]
            if row.empty:
                continue
            years_back = target_season - s
            # Adjust for COVID gap — don't count 2020 as a gap
            if target_season > 2020 and s < 2020:
                years_back -= 1
            weight = SEASON_DECAY ** years_back
            for c in components:
                val = row.iloc[0].get(c, np.nan)
                if pd.notna(val):
                    weighted_sums[c] += val * weight
            total_weight += weight
            seasons_found += 1

        if total_weight > 0:
            row_data = {"school_id": sid, "seasons_blended": seasons_found}
            for c in components:
                row_data[c] = weighted_sums[c] / total_weight
            result_rows.append(row_data)

    if not result_rows:
        return pd.DataFrame(columns=["school_id"] + components + ["seasons_blended"])
    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_hca_features(espn_box: pd.DataFrame,
                       net_rankings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute KenPom-style HCA features from ESPN box score data.

    Args:
        espn_box: DataFrame with columns: game_id, date, season, school_id,
                  espn_id, opponent_espn_id, home_away, points, opp_points,
                  fouls, opp_fouls, turnovers, opp_turnovers, steals,
                  opp_steals, blocks, opp_blocks
        net_rankings: Ignored (kept for backward compatibility).

    Returns:
        DataFrame with columns: school_id, season, hca_score, hca_points,
        foul_advantage, scoring_advantage, turnover_advantage, other_advantage,
        block_advantage, travel_advantage_pts, avg_opp_travel_mi,
        home_pts_margin, road_pts_margin, home_foul_margin, road_foul_margin,
        home_games, road_games, espn_home_wins, espn_home_losses,
        espn_road_wins, espn_road_losses, seasons_blended
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
    df["stl_margin"] = df["steals"] - df["opp_steals"]
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
            home_stl_margin=("stl_margin", "mean"),
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
            road_stl_margin=("stl_margin", "mean"),
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
    merged["other_advantage"] = merged["home_stl_margin"].fillna(0) - merged["road_stl_margin"].fillna(0)

    # Backward compat: block_advantage kept as 0.0
    merged["block_advantage"] = 0.0

    # Apply minimums: teams without enough games get defaults
    below_threshold = (
        (merged["home_games"] < MIN_HOME_GAMES)
        | (merged["road_games"] < MIN_ROAD_GAMES)
    )

    # Compute travel distances
    travel_df = _compute_travel_distances(espn_box)

    # Multi-season smoothing and composite scoring per target season
    components = ["foul_advantage", "scoring_advantage",
                  "turnover_advantage", "other_advantage"]
    all_seasons = sorted(merged["season"].unique())

    results = []
    for target_season in all_seasons:
        season_mask = merged["season"] == target_season
        season_df = merged[season_mask].copy()
        threshold_mask = below_threshold[season_mask].values

        # Smooth component values across seasons
        smoothed = _smooth_across_seasons(merged, target_season, components)

        if not smoothed.empty:
            # Merge smoothed values back, overwriting single-season raw values
            season_df = season_df.drop(columns=components, errors="ignore")
            season_df = season_df.merge(
                smoothed[["school_id"] + components + ["seasons_blended"]],
                on="school_id", how="left",
            )
            for c in components:
                season_df[c] = season_df[c].fillna(0.0)
            season_df["seasons_blended"] = season_df["seasons_blended"].fillna(1).astype(int)
        else:
            season_df["seasons_blended"] = 1

        # Z-score normalize each component within the season (qualified only)
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

        # Weighted composite of z-scored components — no quality multiplier
        season_df["raw_hca"] = (
            season_df["foul_advantage_z"] * FOUL_WEIGHT
            + season_df["scoring_advantage_z"] * SCORING_WEIGHT
            + season_df["turnover_advantage_z"] * TURNOVER_WEIGHT
            + season_df["other_advantage_z"] * OTHER_WEIGHT
        )
        season_df.loc[threshold_mask, "raw_hca"] = np.nan

        # Percentile rank among qualified teams (0.0-1.0)
        qualified = season_df["raw_hca"].dropna()
        if len(qualified) > 1:
            season_df["hca_score"] = season_df["raw_hca"].rank(pct=True)
        else:
            season_df["hca_score"] = 0.5

        # HCA points: linear map from percentile to calibrated range
        season_df["hca_points"] = HCA_PTS_MIN + season_df["hca_score"] * (HCA_PTS_MAX - HCA_PTS_MIN)

        # Merge travel distances
        season_travel = travel_df[travel_df["season"] == target_season] if not travel_df.empty else pd.DataFrame()
        if not season_travel.empty:
            season_df = season_df.merge(
                season_travel[["school_id", "avg_opp_travel_mi"]],
                on="school_id", how="left",
            )
        else:
            season_df["avg_opp_travel_mi"] = 0.0
        season_df["avg_opp_travel_mi"] = season_df["avg_opp_travel_mi"].fillna(0.0)

        # Travel advantage in points
        season_df["travel_advantage_pts"] = (
            season_df["avg_opp_travel_mi"] / 1000.0 * TRAVEL_PTS_PER_1000MI
        )

        # Fill below-threshold teams with median defaults
        season_df.loc[threshold_mask, "hca_score"] = 0.5
        season_df.loc[threshold_mask, "hca_points"] = LEAGUE_AVG_HCA_PTS
        season_df.loc[threshold_mask, "foul_advantage"] = 0.0
        season_df.loc[threshold_mask, "scoring_advantage"] = 0.0
        season_df.loc[threshold_mask, "turnover_advantage"] = 0.0
        season_df.loc[threshold_mask, "other_advantage"] = 0.0
        season_df.loc[threshold_mask, "block_advantage"] = 0.0

        results.append(season_df)

    if not results:
        return pd.DataFrame()

    result = pd.concat(results, ignore_index=True)

    # Select output columns
    output_cols = [
        "school_id", "season",
        "hca_score", "hca_points",
        "foul_advantage", "scoring_advantage", "turnover_advantage",
        "other_advantage", "block_advantage",
        "travel_advantage_pts", "avg_opp_travel_mi",
        "home_pts_margin", "road_pts_margin", "home_foul_margin", "road_foul_margin",
        "home_games", "road_games",
        "espn_home_wins", "espn_home_losses", "espn_road_wins", "espn_road_losses",
        "seasons_blended",
    ]
    # Ensure all output columns exist
    for col in output_cols:
        if col not in result.columns:
            result[col] = 0.0

    return result[output_cols].copy()
