"""Build bracket structure: assign regions, format matchups.

Implements NCAA bracket placement rules:
1. S-curve ordering across 4 regions
2. Conference separation at seeds 1-4 (hard constraint)
3. Conference meeting avoidance at seeds 5+ (soft constraint)
4. Geographic proximity optimization (soft constraint)
5. First Four pairing
"""

import json
import math
from collections import Counter, defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

from config import REGIONS, SEEDS, TEAMS_PER_SEED, QUADRANTS, TEAM_LOCATIONS_PATH, VENUES_PATH


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


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _build_s_curve(seed_line):
    """Return region indices [0-3] for a seed line following S-curve.

    Odd seeds: forward [0,1,2,3]
    Even seeds: reversed [3,2,1,0]
    """
    if seed_line % 2 == 1:
        return [0, 1, 2, 3]
    return [3, 2, 1, 0]


# Pre-compute which seeds belong to which bracket half.
# Top half: quadrants A + B, bottom half: quadrants C + D.
_TOP_HALF_SEEDS = set()
_BOTTOM_HALF_SEEDS = set()
for _q in ("A", "B"):
    _TOP_HALF_SEEDS.update(QUADRANTS[_q])
for _q in ("C", "D"):
    _BOTTOM_HALF_SEEDS.update(QUADRANTS[_q])


def _get_half(seed):
    """Return 'top' or 'bottom' bracket half for a seed number."""
    if seed in _TOP_HALF_SEEDS:
        return "top"
    if seed in _BOTTOM_HALF_SEEDS:
        return "bottom"
    return None


# ---------------------------------------------------------------------------
# Conference separation (seeds 1-4, hard constraint)
# ---------------------------------------------------------------------------

def _enforce_conference_separation(slots):
    """Ensure no two teams from the same conference share a region at seeds 1-4.

    Processes seed lines in order (1→4). For each, tries all 24 region
    permutations and picks the one with zero conflicts and minimal S-curve
    disruption.
    """
    for seed in range(1, 5):
        seed_slots = [s for s in slots if s["seed"] == seed]
        if len(seed_slots) != 4:
            continue

        # Conferences already placed at earlier seed lines, per region
        region_confs = {r: set() for r in range(4)}
        for s in slots:
            if s["seed"] < seed:
                region_confs[s["region"]].add(s["conf"])

        def count_conflicts(region_assignment):
            n = 0
            for i, s in enumerate(seed_slots):
                if s["conf"] in region_confs[region_assignment[i]]:
                    n += 1
            return n

        current = [s["region"] for s in seed_slots]
        if count_conflicts(current) == 0:
            continue

        s_curve = _build_s_curve(seed)
        best = current
        best_c = count_conflicts(current)
        best_dist = sum(1 for a, b in zip(current, s_curve) if a != b)

        for perm in permutations(range(4)):
            perm = list(perm)
            c = count_conflicts(perm)
            dist = sum(1 for a, b in zip(perm, s_curve) if a != b)
            if c < best_c or (c == best_c and dist < best_dist):
                best = perm
                best_c = c
                best_dist = dist

        for i, s in enumerate(seed_slots):
            s["region"] = best[i]


# ---------------------------------------------------------------------------
# Conference meeting avoidance (seeds 5-16, soft constraint)
# ---------------------------------------------------------------------------

def _count_half_conflicts(slots):
    """Count same-conference teams placed in the same bracket half of the
    same region. These teams could meet before the regional final."""
    groups = defaultdict(list)
    for s in slots:
        h = _get_half(s["seed"])
        if h:
            groups[(s["region"], h)].append(s["conf"])

    conflicts = 0
    for confs in groups.values():
        for _, n in Counter(confs).items():
            if n > 1:
                conflicts += n - 1
    return conflicts


def _reduce_conference_meetings(slots):
    """For seeds 5-16, try region permutations to minimize same-conference
    matchups before the regional final (same bracket half of same region)."""
    for seed in range(5, 17):
        seed_slots = [s for s in slots if s["seed"] == seed]
        if len(seed_slots) != 4:
            continue

        s_curve = _build_s_curve(seed)
        best = [s["region"] for s in seed_slots]
        best_c = None
        best_dist = None

        for perm in permutations(range(4)):
            perm = list(perm)
            # Temporarily apply this permutation
            for i, s in enumerate(seed_slots):
                s["region"] = perm[i]

            c = _count_half_conflicts(slots)
            dist = sum(1 for a, b in zip(perm, s_curve) if a != b)

            if best_c is None or c < best_c or (c == best_c and dist < best_dist):
                best = list(perm)
                best_c = c
                best_dist = dist

        # Apply best permanently
        for i, s in enumerate(seed_slots):
            s["region"] = best[i]


# ---------------------------------------------------------------------------
# Geographic proximity (optional soft constraint)
# ---------------------------------------------------------------------------

def _optimize_geography(slots):
    """Within each seed line, swap teams between regions to reduce travel
    distance to first/second-round venues, without violating conference
    separation at seeds 1-4."""
    team_locs = _load_json(TEAM_LOCATIONS_PATH)
    venues = _load_json(VENUES_PATH)

    if not team_locs or not venues:
        return

    first_second = venues.get("first_second", [])
    if not first_second:
        return

    def venue_dist(team_name, region_idx):
        """Min distance from team to any R1/R2 venue in that region."""
        loc = team_locs.get(team_name)
        if not loc:
            return 0.0
        region_name = REGIONS[region_idx]
        min_d = float("inf")
        for v in first_second:
            if v.get("region") == region_name:
                d = _haversine(loc["lat"], loc["lon"], v["lat"], v["lon"])
                min_d = min(min_d, d)
        return min_d if min_d < float("inf") else 0.0

    def swap_preserves_conf_separation(slot_a, slot_b):
        """Return True if swapping two slots' regions keeps seeds 1-4 clean."""
        if slot_a["seed"] > 4 and slot_b["seed"] > 4:
            return True  # only seeds 1-4 have hard constraint
        region_confs = {r: set() for r in range(4)}
        for s in slots:
            if s["seed"] <= 4 and s is not slot_a and s is not slot_b:
                region_confs[s["region"]].add(s["conf"])
        new_r_a, new_r_b = slot_b["region"], slot_a["region"]
        if slot_a["seed"] <= 4 and slot_a["conf"] in region_confs[new_r_a]:
            return False
        if slot_b["seed"] <= 4 and slot_b["conf"] in region_confs[new_r_b]:
            return False
        return True

    for seed in SEEDS:
        seed_slots = [s for s in slots if s["seed"] == seed]
        if len(seed_slots) < 2:
            continue

        improved = True
        while improved:
            improved = False
            for i in range(len(seed_slots)):
                for j in range(i + 1, len(seed_slots)):
                    a, b = seed_slots[i], seed_slots[j]
                    if a["region"] == b["region"]:
                        continue

                    cur = venue_dist(a["team"], a["region"]) + venue_dist(b["team"], b["region"])
                    swp = venue_dist(a["team"], b["region"]) + venue_dist(b["team"], a["region"])

                    if swp < cur and swap_preserves_conf_separation(a, b):
                        a["region"], b["region"] = b["region"], a["region"]
                        improved = True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def assign_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign the 68 teams to 4 regions using NCAA bracket placement rules.

    Args:
        df: DataFrame with predicted_seed and raw_seed columns, plus
            conference and team columns.

    Returns:
        DataFrame with 'region' and 'first_four' columns added.
    """
    df = df.copy()
    df["region"] = ""

    sort_col = "raw_seed" if "raw_seed" in df.columns else "selection_prob"
    ascending = sort_col == "raw_seed"

    # ------------------------------------------------------------------
    # Step 1: Build S-curve slot list for the 64 direct-placement teams
    # ------------------------------------------------------------------
    slots = []
    for seed in SEEDS:
        direct = df[(df["predicted_seed"] == seed) & (~df["first_four"])]
        direct = direct.sort_values(sort_col, ascending=ascending)

        n = min(4, len(direct))
        s_curve = _build_s_curve(seed)[:n]

        for i, (idx, row) in enumerate(direct.head(n).iterrows()):
            slots.append({
                "idx": idx,
                "seed": seed,
                "conf": row.get("conference", ""),
                "region": s_curve[i],
                "team": row["team"],
            })

    # ------------------------------------------------------------------
    # Step 2: Conference separation at seeds 1-4 (hard)
    # ------------------------------------------------------------------
    _enforce_conference_separation(slots)

    # ------------------------------------------------------------------
    # Step 3: Conference meeting avoidance at seeds 5-16 (soft)
    # ------------------------------------------------------------------
    _reduce_conference_meetings(slots)

    # ------------------------------------------------------------------
    # Step 4: Geographic optimization (soft)
    # ------------------------------------------------------------------
    _optimize_geography(slots)

    # ------------------------------------------------------------------
    # Step 5: Apply region assignments for direct teams
    # ------------------------------------------------------------------
    for s in slots:
        df.loc[s["idx"], "region"] = REGIONS[s["region"]]

    # ------------------------------------------------------------------
    # Step 6: Assign First Four teams to their regions
    #   Each FF seed has 2 direct teams occupying 2 regions.
    #   The 4 FF teams form 2 games, each playing into one of the
    #   remaining 2 region slots.
    # ------------------------------------------------------------------
    for ff_seed in (11, 16):
        ff_teams = df[
            (df["predicted_seed"] == ff_seed) & (df["first_four"])
        ].sort_values(sort_col, ascending=ascending)

        direct_regions = {
            s["region"] for s in slots if s["seed"] == ff_seed
        }
        empty_regions = [r for r in range(4) if r not in direct_regions]

        for i, (idx, _) in enumerate(ff_teams.iterrows()):
            region_idx = empty_regions[i // 2] if i // 2 < len(empty_regions) else i % 4
            df.loc[idx, "region"] = REGIONS[region_idx]

    return df


# ---------------------------------------------------------------------------
# Matchup builders (unchanged)
# ---------------------------------------------------------------------------

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
            if not low.empty:
                low_name = low.iloc[0]["team"]
            else:
                # Slot filled by First Four winner — show both play-in teams
                ff_in_region = region_teams[
                    (region_teams["predicted_seed"] == low_seed) &
                    (region_teams["first_four"])
                ]
                if len(ff_in_region) >= 2:
                    names = ff_in_region["team"].tolist()
                    low_name = f"{names[0]}/{names[1]}"
                else:
                    low_name = "TBD"

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
