"""Scrape the AP Top 25 poll from ESPN's Rankings API.

ESPN Rankings API returns AP Top 25 with rank, previous rank, points,
first-place votes, trend, record, and others receiving votes.
Supports fetching all historical weeks for the current season.
"""

import json
import os
import time

import requests

from config import ESPN_RANKINGS_URL, ESPN_REQUEST_DELAY, PROCESSED_DIR, DATA_DIR, PREDICTION_SEASON


# Build reverse mapping: ESPN team ID (str) -> school_id
_ESPN_LOGOS = {}
_espn_logos_path = os.path.join(DATA_DIR, "espn_logos.json")
if os.path.exists(_espn_logos_path):
    with open(_espn_logos_path, "r") as _f:
        _ESPN_LOGOS = json.load(_f)
_ESPN_ID_TO_SCHOOL = {v: k for k, v in _ESPN_LOGOS.items()}


def _parse_ap_poll(data: dict) -> dict | None:
    """Parse the AP poll from an ESPN Rankings API response.

    Returns dict with week_label, updated, ranks, others — or None if not found.
    """
    ap_poll = None
    for poll in data.get("rankings", []):
        if poll.get("type") == "ap":
            ap_poll = poll
            break

    if ap_poll is None:
        return None

    updated = ap_poll.get("date", "")[:10]
    week_label = ap_poll.get("occurrence", {}).get("displayValue", "")

    ranks = []
    for entry in ap_poll.get("ranks", []):
        team_info = entry.get("team", {})
        espn_id = team_info.get("id", "")
        school_id = _ESPN_ID_TO_SCHOOL.get(espn_id, "")

        ranks.append({
            "rank": entry.get("current", 0),
            "previous": entry.get("previous", 0),
            "team_name": team_info.get("location", ""),
            "nickname": team_info.get("nickname", ""),
            "abbreviation": team_info.get("abbreviation", ""),
            "school_id": school_id,
            "espn_id": espn_id,
            "record": entry.get("recordSummary", ""),
            "points": entry.get("points", 0),
            "first_place_votes": entry.get("firstPlaceVotes", 0),
            "trend": entry.get("trend", ""),
        })

    others = []
    for entry in ap_poll.get("others", []):
        team_info = entry.get("team", {})
        espn_id = team_info.get("id", "")
        school_id = _ESPN_ID_TO_SCHOOL.get(espn_id, "")

        others.append({
            "team_name": team_info.get("location", ""),
            "nickname": team_info.get("nickname", ""),
            "abbreviation": team_info.get("abbreviation", ""),
            "school_id": school_id,
            "espn_id": espn_id,
            "record": entry.get("recordSummary", ""),
            "points": entry.get("points", 0),
            "previous": entry.get("previous", 0),
        })

    return {
        "week_label": week_label,
        "updated": updated,
        "ranks": ranks,
        "others": others,
    }


def _fetch_week(week_num: int) -> dict:
    """Fetch a single week from the ESPN Rankings API."""
    url = f"{ESPN_RANKINGS_URL}?seasons={PREDICTION_SEASON}&weeks={week_num}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def scrape_ap_poll() -> dict:
    """Fetch and parse all AP Top 25 poll weeks for the current season.

    Returns dict with season, current_week, and weeks mapping.
    """
    # First, fetch latest to discover current week number
    print("  Fetching ESPN Rankings API (latest)...")
    resp = requests.get(ESPN_RANKINGS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Find current week from occurrence.value
    current_week = 1
    for poll in data.get("rankings", []):
        if poll.get("type") == "ap":
            current_week = int(poll.get("occurrence", {}).get("value", 1))
            break

    print(f"  Current AP Poll week: {current_week}")

    # Parse the latest week (already fetched)
    weeks = {}
    latest = _parse_ap_poll(data)
    if latest:
        weeks[str(current_week)] = latest

    # Fetch all prior weeks
    for week_num in range(1, current_week):
        print(f"  Fetching AP Poll week {week_num}/{current_week}...")
        time.sleep(ESPN_REQUEST_DELAY)
        try:
            week_data = _fetch_week(week_num)
            parsed = _parse_ap_poll(week_data)
            if parsed and parsed["ranks"]:
                weeks[str(week_num)] = parsed
        except Exception as e:
            print(f"  Warning: Failed to fetch week {week_num}: {e}")

    total_teams = sum(len(w["ranks"]) for w in weeks.values())
    print(f"  AP Poll: {len(weeks)} weeks fetched, {total_teams} total ranked entries")

    return {
        "season": PREDICTION_SEASON,
        "current_week": current_week,
        "weeks": weeks,
    }


def save(poll_data: dict) -> str:
    """Save AP poll data to processed directory."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "ap_poll.json")
    with open(out_path, "w") as f:
        json.dump(poll_data, f, indent=2)
    print(f"  Saved to {out_path}")
    return out_path


def scrape_and_save() -> dict:
    """Scrape AP poll and save to disk. Returns the poll data."""
    poll_data = scrape_ap_poll()
    if poll_data.get("weeks"):
        save(poll_data)
    return poll_data
