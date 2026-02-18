"""Daily Bluesky bot for 68bracket.

Reads pipeline output + ESPN scoreboard to post a daily summary thread.
Usage: python bot/twitter_bot.py [--dry-run]

Env vars: BLUESKY_HANDLE, BLUESKY_APP_PASSWORD
"""

import argparse
import json
import math
import os
import sys
from datetime import date, timedelta
from urllib.request import urlopen


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SITE_URL = "hunterwalklin.github.io/68bracket"
ESPN_API = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


# ── Data loading ──────────────────────────────────────────────


def load_json(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_team_data():
    """Load teams_data.json and build lookup dicts."""
    teams = load_json("teams_data.json") or []
    by_espn = {}  # espn_id (str) -> team dict
    by_name = {}  # team name -> team dict
    for t in teams:
        if t.get("espn"):
            by_espn[str(t["espn"])] = t
        if t.get("name"):
            by_name[t["name"]] = t
    return teams, by_espn, by_name


def load_seeds():
    """Load seed lookup from daily_snapshot.json."""
    snapshot = load_json("daily_snapshot.json")
    if not snapshot or "teams" not in snapshot:
        return {}
    seeds = {}
    for info in snapshot["teams"].values():
        seeds[info["team"]] = info["seed"]
    return seeds


def load_bubble():
    """Load bubble data and build team->category lookup."""
    bubble = load_json("bubble.json") or {}
    bubble_teams = {}
    for key in ("last_4_in", "first_4_out", "next_4_out", "last_4_byes"):
        for name in bubble.get(key, []):
            bubble_teams[name] = key
    return bubble, bubble_teams


# ── ESPN API ──────────────────────────────────────────────────


def fetch_espn_scoreboard(dt):
    """Fetch ESPN scoreboard for a given date. Returns list of game dicts."""
    date_str = dt.strftime("%Y%m%d")
    url = f"{ESPN_API}?dates={date_str}"
    try:
        with urlopen(url) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  Warning: ESPN API fetch failed for {date_str}: {e}")
        return []

    games = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        away, home = None, None
        for c in comp.get("competitors", []):
            if c.get("homeAway") == "away":
                away = c
            else:
                home = c
        if not away or not home:
            continue

        away_id = str((away.get("team") or {}).get("id", ""))
        home_id = str((home.get("team") or {}).get("id", ""))
        status = ((comp.get("status") or {}).get("type") or {})

        games.append({
            "away_id": away_id,
            "home_id": home_id,
            "away_name": (away.get("team") or {}).get("displayName", "TBD"),
            "home_name": (home.get("team") or {}).get("displayName", "TBD"),
            "away_score": int(away.get("score", 0) or 0),
            "home_score": int(home.get("score", 0) or 0),
            "state": status.get("state", "pre"),
            "detail": status.get("shortDetail", ""),
        })
    return games


# ── Prediction (mirrors site's JS model) ─────────────────────


def predict_game(away_team, home_team, by_espn, home_id):
    """Predict a game using the same model as the site."""
    a, b = away_team, home_team
    if not a or not b:
        return None

    pace_a = a.get("pace") or 68
    pace_b = b.get("pace") or 68
    pos_factor = (pace_a + pace_b) / 2 / 100

    pts_a = (a["oe"] + b["de"]) / 2 * pos_factor
    pts_b = (b["oe"] + a["de"]) / 2 * pos_factor

    # Home court advantage
    if home_id:
        ht = by_espn.get(str(home_id))
        if ht:
            hca_pts = ht.get("hcaPts", ht.get("hca", 0) * 6 + 1)
            if hca_pts < 0.5:
                hca_pts = 0.5
            half = hca_pts / 2
            if str(home_id) == str(a.get("espn")):
                pts_a += half
                pts_b -= half
            else:
                pts_b += half
                pts_a -= half

    spread = pts_a - pts_b
    win_a = 1 / (1 + 10 ** (-spread / (11 * pos_factor)))

    comp = max(0, 1 - abs(spread) / 20)
    avg_net = (a.get("net", 200) + b.get("net", 200)) / 2
    qual = max(0, 1 - avg_net / 200)
    watch = round((comp * 0.6 + qual * 0.4) * 100)

    return {
        "pts_a": round(pts_a),
        "pts_b": round(pts_b),
        "spread": spread,
        "win_a": win_a,
        "watch": watch,
    }


# ── Tweet composers ───────────────────────────────────────────


def compose_movers(changes, today):
    """Bracket movers tweet."""
    if not changes:
        return None

    lines = [f"\U0001f4ca Bracket Movers \u2014 {today}\n"]
    risers = [(t, c) for t, c in changes.items() if c["direction"] == "up"]
    fallers = [(t, c) for t, c in changes.items() if c["direction"] == "down"]
    risers.sort(key=lambda x: x[1]["new_seed"])
    fallers.sort(key=lambda x: x[1]["new_seed"])

    for team, c in risers:
        prev = c["prev_seed"] or "OUT"
        lines.append(f"\u2b06\ufe0f {team} \u2192 {c['new_seed']}-seed (was {prev})")
    for team, c in fallers:
        lines.append(f"\u2b07\ufe0f {team} \u2192 {c['new_seed']}-seed (was {c['prev_seed']})")

    text = "\n".join(lines)
    if len(text) > 300:
        text = _truncate_to_300(lines[0] + "\n", lines[1:])
    return text


def compose_upsets(games, by_espn, seeds, today):
    """Yesterday's upsets tweet."""
    upsets = []
    for g in games:
        if g["state"] != "post":
            continue
        away_won = g["away_score"] > g["home_score"]
        win_id = g["away_id"] if away_won else g["home_id"]
        lose_id = g["home_id"] if away_won else g["away_id"]
        winner = by_espn.get(win_id)
        loser = by_espn.get(lose_id)
        if not winner or not loser:
            continue

        # Upset = winner NET is 30+ worse than loser AND loser is in the field
        win_net = winner.get("net", 999)
        lose_net = loser.get("net", 999)
        if win_net - lose_net >= 30 and seeds.get(loser["name"]):
            win_score = g["away_score"] if away_won else g["home_score"]
            lose_score = g["home_score"] if away_won else g["away_score"]
            seed_tag = f" ({seeds[loser['name']]}-seed)" if seeds.get(loser["name"]) else ""
            upsets.append({
                "line": f"{winner['name']} {win_score}, {loser['name']}{seed_tag} {lose_score}",
                "watch": win_net - lose_net,
            })

    if not upsets:
        return None

    upsets.sort(key=lambda x: -x["watch"])
    lines = [f"\U0001f6a8 Upsets \u2014 {today}\n"]
    for u in upsets[:5]:
        lines.append(u["line"])

    text = "\n".join(lines)
    if len(text) > 300:
        text = _truncate_to_300(lines[0] + "\n", lines[1:])
    return text


def compose_bubble(bubble, today):
    """Bubble watch tweet."""
    last_4_in = bubble.get("last_4_in", [])
    first_4_out = bubble.get("first_4_out", [])
    if not last_4_in and not first_4_out:
        return None

    lines = [f"\U0001fae7 Bubble Watch \u2014 {today}\n"]
    if last_4_in:
        lines.append(f"Last Four In: {', '.join(last_4_in)}")
    if first_4_out:
        lines.append(f"First Four Out: {', '.join(first_4_out)}")
    return "\n".join(lines)


def compose_bracket_results(games, by_espn, seeds, today):
    """Around the bracket — top results from yesterday."""
    results = []
    for g in games:
        if g["state"] != "post":
            continue
        away_won = g["away_score"] > g["home_score"]
        win_id = g["away_id"] if away_won else g["home_id"]
        lose_id = g["home_id"] if away_won else g["away_id"]
        winner = by_espn.get(win_id)
        loser = by_espn.get(lose_id)
        if not winner or not loser:
            continue

        # Both teams in the field, or winner is a top seed
        win_seed = seeds.get(winner["name"])
        lose_seed = seeds.get(loser["name"])
        if not win_seed:
            continue

        win_score = g["away_score"] if away_won else g["home_score"]
        lose_score = g["home_score"] if away_won else g["away_score"]
        results.append({
            "line": f"{winner['name']} {win_score}, {loser['name']} {lose_score}",
            "seed": win_seed,
        })

    if not results:
        return None

    results.sort(key=lambda x: x["seed"])
    lines = [f"\U0001f3c0 Around the Bracket \u2014 {today}\n"]
    for r in results[:6]:
        lines.append(r["line"])

    text = "\n".join(lines)
    if len(text) > 300:
        text = _truncate_to_300(lines[0] + "\n", lines[1:])
    return text


def compose_games_to_watch(games, by_espn, seeds, bubble_teams, today):
    """Today's games to watch tweet."""
    previews = []
    for g in games:
        if g["state"] != "pre":
            continue
        away_team = by_espn.get(g["away_id"])
        home_team = by_espn.get(g["home_id"])
        if not away_team or not home_team:
            continue

        # At least one team should be in the field or on the bubble
        a_relevant = seeds.get(away_team["name"]) or bubble_teams.get(away_team["name"])
        h_relevant = seeds.get(home_team["name"]) or bubble_teams.get(home_team["name"])
        if not a_relevant and not h_relevant:
            continue

        pred = predict_game(away_team, home_team, by_espn, g["home_id"])
        if not pred:
            continue

        # Format: "Duke at UNC — Duke -4.5 (62%)"
        spread = pred["spread"]
        if abs(spread) < 1.5:
            spread_str = "Pick 'em"
        else:
            fav = away_team["name"] if spread > 0 else home_team["name"]
            spread_str = f"{fav} -{abs(spread):.1f}"

        pct = max(pred["win_a"], 1 - pred["win_a"])
        line = f"{away_team['name']} at {home_team['name']} \u2014 {spread_str} ({pct:.0%})"

        previews.append({"line": line, "watch": pred["watch"]})

    if not previews:
        return None

    previews.sort(key=lambda x: -x["watch"])
    lines = [f"\U0001f440 Games to Watch \u2014 {today}\n"]
    for p in previews[:4]:
        lines.append(p["line"])
    lines.append(f"\n{SITE_URL}")

    text = "\n".join(lines)
    if len(text) > 300:
        text = _truncate_to_300(lines[0] + "\n", lines[1:-1], f"\n{SITE_URL}")
    return text


def _truncate_to_300(header, body_lines, footer=""):
    """Fit as many body lines as possible within 300 chars (Bluesky limit)."""
    available = 300 - len(header) - len(footer)
    kept = []
    for line in body_lines:
        candidate = "\n".join(kept + [line])
        if len(candidate) <= available:
            kept.append(line)
        else:
            break
    return header + "\n".join(kept) + footer


# ── Thread assembly ───────────────────────────────────────────


def compose_thread():
    """Build the full daily summary thread."""
    today = date.today()
    today_str = today.strftime("%b %-d")
    yesterday = today - timedelta(days=1)

    # Load data
    changes = load_json("daily_changes.json") or {}
    bubble, bubble_teams = load_bubble()
    seeds = load_seeds()
    _, by_espn, by_name = load_team_data()

    # Fetch ESPN scores
    print("Fetching ESPN scoreboard...")
    yesterday_games = fetch_espn_scoreboard(yesterday)
    today_games = fetch_espn_scoreboard(today)
    print(f"  Yesterday: {len(yesterday_games)} games, Today: {len(today_games)} games")

    tweets = []

    # 1. Bracket movers
    t = compose_movers(changes, today_str)
    if t:
        tweets.append(t)

    # 2. Upsets from yesterday
    t = compose_upsets(yesterday_games, by_espn, seeds, today_str)
    if t:
        tweets.append(t)

    # 3. Bubble watch
    t = compose_bubble(bubble, today_str)
    if t:
        tweets.append(t)

    # 4. Around the bracket
    t = compose_bracket_results(yesterday_games, by_espn, seeds, today_str)
    if t:
        tweets.append(t)

    # 5. Games to watch today
    t = compose_games_to_watch(today_games, by_espn, seeds, bubble_teams, today_str)
    if t:
        tweets.append(t)

    return tweets


# ── Posting (Bluesky) ─────────────────────────────────────────


def post_thread(posts):
    from atproto import Client, models
    handle = os.environ["BLUESKY_HANDLE"]
    client = Client()
    client.login(handle, os.environ["BLUESKY_APP_PASSWORD"])

    root_ref = None
    parent_ref = None

    for i, text in enumerate(posts):
        reply_to = None
        if parent_ref and root_ref:
            reply_to = models.AppBskyFeedPost.ReplyRef(
                parent=parent_ref, root=root_ref,
            )
        response = client.send_post(text=text, reply_to=reply_to)
        ref = models.create_strong_ref(response)
        if i == 0:
            root_ref = ref
        parent_ref = ref

        label = "Thread start" if i == 0 else f"Reply {i}"
        print(f"  {label}: https://bsky.app/profile/{handle}/post/{response.uri.split('/')[-1]}")


def main():
    parser = argparse.ArgumentParser(description="68bracket Bluesky bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the composed thread without posting")
    args = parser.parse_args()

    posts = compose_thread()
    if not posts:
        print("No data available to compose posts.")
        sys.exit(1)

    print(f"\n--- Thread ({len(posts)} posts) ---")
    for i, text in enumerate(posts):
        print(f"\n[{i + 1}] ({len(text)} chars)")
        print(text)
    print("\n---")

    if args.dry_run:
        print("\n(dry run — not posted)")
        return

    post_thread(posts)
    print(f"\nPosted {len(posts)}-post thread.")


if __name__ == "__main__":
    main()
