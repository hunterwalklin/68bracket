"""Daily Twitter/X bot for 68bracket.

Reads the daily pipeline output and posts a bracket update tweet.
Usage: python bot/twitter_bot.py [--dry-run]
"""

import argparse
import json
import os
import sys
from datetime import date


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SITE_URL = "hunterwalklin.github.io/68bracket"


def load_json(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compose_seed_movers(changes, today):
    """Priority 1: teams that moved seeds after playing."""
    lines = [f"\U0001f4ca Bracket Update \u2014 {today}\n"]

    # Sort: risers first (by new seed), then fallers
    risers = [(t, c) for t, c in changes.items() if c["direction"] == "up"]
    fallers = [(t, c) for t, c in changes.items() if c["direction"] == "down"]
    risers.sort(key=lambda x: x[1]["new_seed"])
    fallers.sort(key=lambda x: x[1]["new_seed"])

    for team, c in risers:
        prev = c["prev_seed"] or "OUT"
        lines.append(f"\u2b06\ufe0f {team} \u2192 {c['new_seed']}-seed (was {prev})")
    for team, c in fallers:
        lines.append(f"\u2b07\ufe0f {team} \u2192 {c['new_seed']}-seed (was {c['prev_seed']})")

    lines.append(f"\nFull bracket: {SITE_URL}")
    return "\n".join(lines)


def compose_bubble_watch(bubble, today):
    """Priority 2: bubble teams."""
    last_4_in = bubble.get("last_4_in", [])
    first_4_out = bubble.get("first_4_out", [])

    if not last_4_in and not first_4_out:
        return None

    lines = [f"\U0001fae7 Bubble Watch \u2014 {today}\n"]
    if last_4_in:
        lines.append(f"Last Four In: {', '.join(last_4_in)}")
    if first_4_out:
        lines.append(f"First Four Out: {', '.join(first_4_out)}")
    lines.append(f"\nFull projections: {SITE_URL}")
    return "\n".join(lines)


def compose_top_seeds(snapshot, today):
    """Priority 3: top 4 seed lines (always available)."""
    if not snapshot or "teams" not in snapshot:
        return None

    teams = snapshot["teams"]
    # Group by seed
    by_seed = {}
    for info in teams.values():
        seed = info["seed"]
        if seed <= 2:
            by_seed.setdefault(seed, []).append(info["team"])

    # Sort each seed line by team name for consistency
    for seed in by_seed:
        by_seed[seed].sort()

    ones = ", ".join(by_seed.get(1, []))
    twos = ", ".join(by_seed.get(2, []))

    if not ones:
        return None

    lines = [f"\U0001f3c0 68bracket Projections \u2014 {today}\n"]
    lines.append(f"1-seeds: {ones}")
    if twos:
        lines.append(f"2-seeds: {twos}")
    lines.append(f"\nUpdated daily: {SITE_URL}")
    return "\n".join(lines)


def compose_tweet():
    """Pick the highest-priority tweet type and compose it."""
    today = date.today().strftime("%b %-d")

    # Priority 1: Seed movers
    changes = load_json("daily_changes.json")
    if changes:
        tweet = compose_seed_movers(changes, today)
        # Truncate to 280 chars if needed
        if len(tweet) > 280:
            tweet = truncate_movers(changes, today)
        return tweet

    # Priority 2: Bubble watch
    bubble = load_json("bubble.json")
    if bubble:
        tweet = compose_bubble_watch(bubble, today)
        if tweet:
            return tweet

    # Priority 3: Top seeds
    snapshot = load_json("daily_snapshot.json")
    if snapshot:
        tweet = compose_top_seeds(snapshot, today)
        if tweet:
            return tweet

    return None


def truncate_movers(changes, today):
    """If the full movers tweet exceeds 280 chars, include only the top movers."""
    risers = [(t, c) for t, c in changes.items() if c["direction"] == "up"]
    fallers = [(t, c) for t, c in changes.items() if c["direction"] == "down"]
    risers.sort(key=lambda x: x[1]["new_seed"])
    fallers.sort(key=lambda x: x[1]["new_seed"])

    all_movers = []
    for team, c in risers:
        prev = c["prev_seed"] or "OUT"
        all_movers.append(f"\u2b06\ufe0f {team} \u2192 {c['new_seed']}-seed (was {prev})")
    for team, c in fallers:
        all_movers.append(f"\u2b07\ufe0f {team} \u2192 {c['new_seed']}-seed (was {c['prev_seed']})")

    header = f"\U0001f4ca Bracket Update \u2014 {today}\n\n"
    footer = f"\n\nFull bracket: {SITE_URL}"
    available = 280 - len(header) - len(footer)

    body_lines = []
    for line in all_movers:
        candidate = "\n".join(body_lines + [line])
        if len(candidate) <= available:
            body_lines.append(line)
        else:
            break

    return header + "\n".join(body_lines) + footer


def post_tweet(text):
    """Post a tweet using tweepy and Twitter API v2."""
    import tweepy

    client = tweepy.Client(
        consumer_key=os.environ["TWITTER_API_KEY"],
        consumer_secret=os.environ["TWITTER_API_SECRET"],
        access_token=os.environ["TWITTER_ACCESS_TOKEN"],
        access_token_secret=os.environ["TWITTER_ACCESS_SECRET"],
    )

    response = client.create_tweet(text=text)
    tweet_id = response.data["id"]
    print(f"Posted tweet: https://x.com/i/web/status/{tweet_id}")
    return tweet_id


def main():
    parser = argparse.ArgumentParser(description="68bracket Twitter bot")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the composed tweet without posting",
    )
    args = parser.parse_args()

    tweet = compose_tweet()
    if tweet is None:
        print("No data available to compose a tweet.")
        sys.exit(1)

    print(f"--- Tweet ({len(tweet)} chars) ---")
    print(tweet)
    print("---")

    if args.dry_run:
        print("\n(dry run — not posted)")
        return

    post_tweet(tweet)


if __name__ == "__main__":
    main()
