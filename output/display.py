"""Display bracket predictions in console and markdown formats."""

import pandas as pd

from config import REGIONS, SEEDS, PREDICTION_SEASON
from output.bracket_builder import build_matchups, build_first_four


def _render_half_bracket(teams, region_name, name_w):
    """Render 8 teams as a bracket half (4 R64 matchups -> E8 connector).

    Returns 15 lines of text forming:
        R64 -> R32 -> Sweet 16 -> Elite 8 connector
    """
    gap = 15  # distance between round column positions
    c1 = name_w       # R64 connector column
    c2 = c1 + gap      # R32 connector column
    c3 = c2 + gap      # E8 connector column

    suffix = f"── {region_name}"
    width = c3 + 1 + len(suffix)

    grid = [[' '] * width for _ in range(15)]

    # Team labels on even rows: 0, 2, 4, 6, 8, 10, 12, 14
    for i in range(8):
        row = i * 2
        label = teams[i]
        for j, ch in enumerate(label):
            grid[row][j] = ch
        # Space after name, then ─ padding to connector
        if len(label) < c1:
            grid[row][len(label)] = ' '
        for j in range(len(label) + 1, c1):
            grid[row][j] = '─'
        grid[row][c1] = '┐' if i % 2 == 0 else '┘'

    # R64 connectors: ├ at c1, horizontal ─ to c2
    for row in [1, 5, 9, 13]:
        grid[row][c1] = '├'
        for j in range(c1 + 1, c2):
            grid[row][j] = '─'

    # R32 top pair (connects R64 rows 1 and 5, connector at row 3)
    grid[1][c2] = '┐'
    grid[2][c2] = '│'
    grid[3][c2] = '├'
    grid[4][c2] = '│'
    grid[5][c2] = '┘'

    # R32 bottom pair (connects R64 rows 9 and 13, connector at row 11)
    grid[9][c2] = '┐'
    grid[10][c2] = '│'
    grid[11][c2] = '├'
    grid[12][c2] = '│'
    grid[13][c2] = '┘'

    # R32 horizontal lines to S16/E8 at rows 3 and 11
    for row in [3, 11]:
        for j in range(c2 + 1, c3):
            grid[row][j] = '─'

    # E8 connector at c3 (connects rows 3 and 11, connector at row 7)
    grid[3][c3] = '┐'
    for row in range(4, 7):
        grid[row][c3] = '│'
    grid[7][c3] = '├'
    for row in range(8, 11):
        grid[row][c3] = '│'
    grid[11][c3] = '┘'

    # Region label on E8 connector
    for j, ch in enumerate(suffix):
        grid[7][c3 + 1 + j] = ch

    return [''.join(row).rstrip() for row in grid]


def _render_region_bracket(region_matchups, region_name):
    """Render a full region bracket (16 teams, 8 R64 matchups) as ASCII art.

    region_matchups: list of 8 matchup dicts in bracket order
        (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
    """
    # Format team labels
    teams = []
    for m in region_matchups:
        teams.append(f"({m['high_seed']}) {m['high_team']}")
        teams.append(f"({m['low_seed']}) {m['low_team']}")

    top_teams = teams[:8]     # matchups 0-3: 1v16, 8v9, 5v12, 4v13
    bottom_teams = teams[8:]  # matchups 4-7: 6v11, 3v14, 7v10, 2v15

    # Consistent column width across both halves
    max_name_len = max(len(t) for t in teams)
    name_w = max_name_len + 3  # space + at least 2 ─ chars

    top_lines = _render_half_bracket(top_teams, region_name, name_w)
    bottom_lines = _render_half_bracket(bottom_teams, region_name, name_w)

    # Center title above bracket
    max_width = max(
        max((len(l) for l in top_lines), default=0),
        max((len(l) for l in bottom_lines), default=0),
    )
    title = f"{region_name} REGION"

    lines = [title.center(max_width).rstrip()]
    lines.append("")
    lines.extend(top_lines)
    lines.append("")
    lines.extend(bottom_lines)

    return lines


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
        region_matchups = [m for m in matchups if m["region"] == region]
        bracket_lines = _render_region_bracket(region_matchups, region)
        print()
        for line in bracket_lines:
            print(f"  {line}")


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
        lines.append("```text")
        bracket_lines = _render_region_bracket(region_matchups, region)
        lines.extend(bracket_lines)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
