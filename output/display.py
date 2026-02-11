"""Display bracket predictions in console and markdown formats."""

import pandas as pd

from config import REGIONS, SEEDS, PREDICTION_SEASON
from output.bracket_builder import build_matchups, build_first_four


def display_seed_list(df: pd.DataFrame, season: int = PREDICTION_SEASON):
    """Print the 68-team field organized by seed line."""
    print(f"\n{'='*60}")
    print(f"  NCAA Tournament Predictions - {season-1}-{str(season)[2:]}")
    print(f"{'='*60}")

    for seed in SEEDS:
        seed_teams = df[df["predicted_seed"] == seed].sort_values(
            "raw_seed" if "raw_seed" in df.columns else "selection_prob"
        )
        teams_str = ", ".join(seed_teams["team"].tolist())
        ff_marker = ""
        if seed in [11, 16]:
            n_ff = seed_teams["first_four"].sum() if "first_four" in seed_teams.columns else 0
            if n_ff > 0:
                ff_marker = f" (+{int(n_ff)} First Four)"
        print(f"  {seed:2d}. {teams_str}{ff_marker}")


def display_bracket(df: pd.DataFrame, season: int = PREDICTION_SEASON):
    """Print the bracket with region assignments and matchups."""
    print(f"\n{'='*60}")
    print(f"  NCAA Tournament Bracket - {season-1}-{str(season)[2:]}")
    print(f"{'='*60}")

    # First Four
    ff_games = build_first_four(df)
    if ff_games:
        print(f"\n  --- First Four ---")
        for game in ff_games:
            print(f"  ({game['seed']}) {game['team1']} vs {game['team2']} "
                  f"[{game['region']}]")

    # Regional brackets
    matchups = build_matchups(df)
    for region in REGIONS:
        print(f"\n  --- {region} Region ---")
        region_matchups = [m for m in matchups if m["region"] == region]
        for m in region_matchups:
            print(f"  ({m['high_seed']:2d}) {m['high_team']:25s} vs "
                  f"({m['low_seed']:2d}) {m['low_team']}")


def display_selection_summary(df: pd.DataFrame):
    """Print summary statistics about the selection."""
    total = len(df)
    auto_bids = len(df[df.get("selection_method", "") == "auto_bid"]) if "selection_method" in df.columns else 0
    at_large = total - auto_bids

    print(f"\n  Selection Summary:")
    print(f"    Total teams: {total}")
    if "selection_method" in df.columns:
        print(f"    Auto-bids: {auto_bids}")
        print(f"    At-large: {at_large}")

    if "conference" in df.columns:
        conf_counts = df["conference"].value_counts()
        multi_bid = conf_counts[conf_counts > 1]
        if not multi_bid.empty:
            print(f"\n  Multi-bid conferences:")
            for conf, count in multi_bid.items():
                teams = df[df["conference"] == conf]["team"].tolist()
                print(f"    {conf} ({count}): {', '.join(teams)}")


def generate_markdown(df: pd.DataFrame, season: int = PREDICTION_SEASON) -> str:
    """Generate a markdown representation of the bracket."""
    lines = [
        f"# NCAA Tournament Predictions - {season-1}-{str(season)[2:]}",
        "",
        "## Seed List",
        "",
        "| Seed | Teams |",
        "|------|-------|",
    ]

    for seed in SEEDS:
        seed_teams = df[df["predicted_seed"] == seed].sort_values(
            "raw_seed" if "raw_seed" in df.columns else "selection_prob"
        )
        teams = []
        for _, row in seed_teams.iterrows():
            name = row["team"]
            if row.get("first_four", False):
                name += "*"
            teams.append(name)
        lines.append(f"| {seed} | {', '.join(teams)} |")

    lines.extend(["", "*First Four play-in game", ""])

    # First Four
    ff_games = build_first_four(df)
    if ff_games:
        lines.extend(["## First Four", ""])
        for game in ff_games:
            lines.append(f"- ({game['seed']}) {game['team1']} vs {game['team2']} [{game['region']}]")
        lines.append("")

    # Regional brackets
    matchups = build_matchups(df)
    for region in REGIONS:
        lines.extend([f"## {region} Region", ""])
        region_matchups = [m for m in matchups if m["region"] == region]
        lines.append("| Matchup | |")
        lines.append("|---------|---|")
        for m in region_matchups:
            lines.append(f"| ({m['high_seed']}) {m['high_team']} | vs ({m['low_seed']}) {m['low_team']} |")
        lines.append("")

    return "\n".join(lines)
