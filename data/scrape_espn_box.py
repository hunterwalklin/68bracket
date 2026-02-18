"""ESPN box score scraper for KenPom-style home court advantage analysis.

Fetches game-level box scores from the ESPN API (no auth needed):
- Scoreboard: day-by-day to collect game IDs
- Summary: per-game box scores for fouls, turnovers, steals, blocks, points

Output: data/processed/espn_box_scores.parquet — one row per team per game.
"""

import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from config import (
    DATA_DIR, RAW_DIR, PROCESSED_DIR, PREDICTION_SEASON, TRAINING_SEASONS,
)

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard?dates={date}&groups=50&limit=200"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary?event={game_id}"
)
ESPN_REQUEST_DELAY = 0.1  # ESPN API is fast; minimal politeness delay

# ESPN logo mapping (school_id -> espn_id) — load once for reverse lookup
_espn_logos_path = os.path.join(DATA_DIR, "espn_logos.json")
_ESPN_TO_SCHOOL = {}  # espn_id -> school_id (reverse mapping)
if os.path.exists(_espn_logos_path):
    with open(_espn_logos_path, "r") as _f:
        _logos = json.load(_f)
        _ESPN_TO_SCHOOL = {v: k for k, v in _logos.items()}


def _espn_to_school(espn_id: str) -> str:
    """Map ESPN team ID to Sports-Reference school_id."""
    return _ESPN_TO_SCHOOL.get(str(espn_id), "")


def _cache_path(subdir: str, name: str) -> str:
    """Return a cache file path within data/raw/espn/."""
    path = os.path.join(RAW_DIR, "espn", subdir)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{name}.json")


def _is_cache_valid(path: str, max_age_hours: float = 24 * 7) -> bool:
    """Check if a cached JSON file exists and is fresh enough."""
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime < timedelta(hours=max_age_hours)


def _fetch_json(url: str, cache_file: str, max_age_hours: float = 24 * 7) -> dict | None:
    """Fetch JSON from ESPN API with file caching."""
    if _is_cache_valid(cache_file, max_age_hours):
        with open(cache_file, "r") as f:
            return json.load(f)

    time.sleep(ESPN_REQUEST_DELAY)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  ESPN API error: {e}")
        # Fall back to stale cache if available
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
        return None


def _season_date_range(season: int) -> tuple[datetime, datetime]:
    """Return (start, end) dates for an NCAA basketball season.

    Season 2026 means the 2025-26 academic year: Nov 2025 through Apr 2026.
    """
    start = datetime(season - 1, 11, 1)
    end = min(datetime(season, 4, 15), datetime.now())
    return start, end


def _fetch_scoreboard(date_str: str, max_age_hours: float = 24 * 7) -> list[dict]:
    """Fetch scoreboard for a single date. Returns list of game info dicts."""
    url = ESPN_SCOREBOARD_URL.format(date=date_str)
    cache = _cache_path("scoreboards", date_str)
    data = _fetch_json(url, cache, max_age_hours=max_age_hours)
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        game_id = event.get("id")
        status_type = event.get("status", {}).get("type", {}).get("name", "")
        if status_type != "STATUS_FINAL":
            continue
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue
        home_team = None
        away_team = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_team = c.get("team", {}).get("id")
            else:
                away_team = c.get("team", {}).get("id")
        if home_team and away_team:
            games.append({
                "game_id": game_id,
                "date": date_str,
                "home_espn_id": home_team,
                "away_espn_id": away_team,
            })
    return games


def _parse_box_score(data: dict, game_id: str, date_str: str, season: int) -> list[dict]:
    """Parse an ESPN summary response into box score rows (one per team)."""
    boxscore = data.get("boxscore", {})
    teams_data = boxscore.get("teams", [])
    if len(teams_data) != 2:
        return []

    # Determine home/away from header
    header = data.get("header", {})
    competitions = header.get("competitions", [{}])
    comp = competitions[0] if competitions else {}
    competitors = comp.get("competitors", [])

    home_id = None
    away_id = None
    scores = {}  # espn_id -> points (from header, not boxscore stats)
    for c in competitors:
        tid = str(c.get("id", ""))
        if c.get("homeAway") == "home":
            home_id = tid
        else:
            away_id = tid
        try:
            scores[tid] = int(c.get("score", 0))
        except (ValueError, TypeError):
            scores[tid] = 0

    rows = []
    parsed = {}  # espn_id -> stats dict

    for team_entry in teams_data:
        team_info = team_entry.get("team", {})
        espn_id = str(team_info.get("id", ""))
        statistics = team_entry.get("statistics", [])

        # Build stat lookup from the statistics array
        stats = {}
        for stat in statistics:
            stats[stat.get("name", "")] = stat.get("displayValue", "0")

        # Extract relevant stats
        def _int(key):
            val = stats.get(key, "0")
            try:
                return int(val.split("-")[0]) if "-" in str(val) else int(float(val))
            except (ValueError, TypeError):
                return 0

        # Points come from header competitors, not boxscore stats
        points = scores.get(espn_id, 0)

        parsed[espn_id] = {
            "espn_id": espn_id,
            "points": points,
            "fouls": _int("fouls"),
            "turnovers": _int("totalTurnovers") or _int("turnovers"),
            "steals": _int("steals"),
            "blocks": _int("blocks"),
        }

    if len(parsed) != 2:
        return []

    ids = list(parsed.keys())
    for espn_id in ids:
        opp_id = ids[1] if espn_id == ids[0] else ids[0]
        me = parsed[espn_id]
        opp = parsed[opp_id]

        is_home = str(espn_id) == str(home_id)
        home_away = "home" if is_home else "road"

        rows.append({
            "game_id": game_id,
            "date": date_str,
            "season": season,
            "school_id": _espn_to_school(espn_id),
            "espn_id": espn_id,
            "opponent_espn_id": opp_id,
            "home_away": home_away,
            "points": me["points"],
            "opp_points": opp["points"],
            "fouls": me["fouls"],
            "opp_fouls": opp["fouls"],
            "turnovers": me["turnovers"],
            "opp_turnovers": opp["turnovers"],
            "steals": me["steals"],
            "opp_steals": opp["steals"],
            "blocks": me["blocks"],
            "opp_blocks": opp["blocks"],
        })

    return rows


def _fetch_box_score(game_id: str, date_str: str, season: int,
                     max_age_hours: float = 24 * 7) -> list[dict]:
    """Fetch and parse a single game's box score."""
    url = ESPN_SUMMARY_URL.format(game_id=game_id)
    cache = _cache_path("boxscores", game_id)
    data = _fetch_json(url, cache, max_age_hours=max_age_hours)
    if not data:
        return []
    return _parse_box_score(data, game_id, date_str, season)


def scrape_season(season: int, progress: bool = True) -> pd.DataFrame:
    """Scrape all box scores for a single season.

    1. Fetch scoreboards day-by-day to collect game IDs.
    2. Fetch box score for each game.
    """
    start, end = _season_date_range(season)
    if progress:
        print(f"  Season {season}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

    # Phase 1: Collect game IDs from scoreboards
    all_games = {}  # game_id -> game info
    day = start
    day_count = 0
    while day <= end:
        date_str = day.strftime("%Y%m%d")
        # Completed games never expire; today's scoreboard refreshes hourly
        is_today = (day.date() == datetime.now().date())
        max_age = 1 if is_today else 24 * 365
        games = _fetch_scoreboard(date_str, max_age_hours=max_age)
        for g in games:
            all_games[g["game_id"]] = g
        day += timedelta(days=1)
        day_count += 1
        if progress and day_count % 30 == 0:
            print(f"    Scoreboards: {day_count} days, {len(all_games)} games so far...")

    if progress:
        print(f"    Scoreboards: {day_count} days, {len(all_games)} completed games")

    # Phase 2: Fetch box scores
    all_rows = []
    fetched = 0
    for game_id, game_info in all_games.items():
        rows = _fetch_box_score(game_id, game_info["date"], season)
        all_rows.extend(rows)
        fetched += 1
        if progress and fetched % 500 == 0:
            print(f"    Box scores: {fetched}/{len(all_games)}...")

    if progress:
        print(f"    Box scores: {fetched} games -> {len(all_rows)} team-game rows")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def scrape_all_seasons(seasons: list[int] | None = None,
                       progress: bool = True) -> pd.DataFrame:
    """Scrape box scores for multiple seasons (full historical scrape)."""
    if seasons is None:
        seasons = TRAINING_SEASONS + [PREDICTION_SEASON]

    all_dfs = []
    for season in seasons:
        if progress:
            print(f"\nESPN box scores — season {season}")
        df = scrape_season(season, progress=progress)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def scrape_recent(days: int = 2, progress: bool = True) -> pd.DataFrame:
    """Scrape only recent games (for daily pipeline refresh).

    Fetches scoreboards for the last N days, then box scores for any
    new game IDs not already in the existing parquet.
    """
    if progress:
        print(f"\nESPN box scores — last {days} days")

    # Load existing data to find known game IDs
    parquet_path = os.path.join(PROCESSED_DIR, "espn_box_scores.parquet")
    known_ids = set()
    if os.path.exists(parquet_path):
        existing = pd.read_parquet(parquet_path, columns=["game_id"])
        known_ids = set(existing["game_id"].unique())

    # Fetch recent scoreboards
    today = datetime.now()
    new_games = {}
    for offset in range(days):
        day = today - timedelta(days=offset)
        date_str = day.strftime("%Y%m%d")
        games = _fetch_scoreboard(date_str, max_age_hours=1)
        for g in games:
            if g["game_id"] not in known_ids:
                new_games[g["game_id"]] = g

    if progress:
        print(f"  Found {len(new_games)} new games")

    if not new_games:
        return pd.DataFrame()

    # Determine season from date
    now_month = today.month
    season = today.year if now_month <= 6 else today.year + 1

    # Fetch box scores for new games
    all_rows = []
    for game_id, game_info in new_games.items():
        rows = _fetch_box_score(game_id, game_info["date"], season)
        all_rows.extend(rows)

    if progress:
        print(f"  Fetched {len(new_games)} box scores -> {len(all_rows)} rows")

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def save(df: pd.DataFrame, upsert: bool = False):
    """Save box score data to parquet.

    Args:
        df: New box score rows.
        upsert: If True, merge with existing data by game_id (for daily refresh).
                If False, overwrite entirely (for full scrape).
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, "espn_box_scores.parquet")

    if upsert and os.path.exists(path):
        existing = pd.read_parquet(path)
        new_ids = set(df["game_id"].unique())
        existing = existing[~existing["game_id"].isin(new_ids)]
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(path, index=False)
        print(f"  Upserted {len(df)} rows -> {len(combined)} total in espn_box_scores.parquet")
    else:
        df.to_parquet(path, index=False)
        print(f"  Saved {len(df)} rows to espn_box_scores.parquet")
