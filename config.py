"""Configuration constants for 68bracket."""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved")

# Sports-Reference base URLs
SR_BASE = "https://www.sports-reference.com/cbb"
SR_SEASONS_URL = SR_BASE + "/seasons/{season}-school-stats.html"
SR_ADVANCED_URL = SR_BASE + "/seasons/{season}-advanced-school-stats.html"
SR_RATINGS_URL = SR_BASE + "/seasons/{season}-ratings.html"
SR_TOURNAMENT_URL = SR_BASE + "/postseason/{season}-ncaa.html"

# WarrenNolan URLs
WARRENNOLAN_NITTY_URL = "https://www.warrennolan.com/basketball/{year}/net-nitty"
WARRENNOLAN_COMPARE_URL = "https://www.warrennolan.com/basketball/{year}/compare-rankings"

# Scraping settings
REQUEST_DELAY = 15.0  # seconds between requests (Sports-Reference rate limit)
REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
CACHE_EXPIRY_DAYS = 7  # re-scrape after this many days

# Training seasons (NET era only, skip 2020 due to COVID cancellation)
TRAINING_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025]
PREDICTION_SEASON = 2026

# Tournament structure
TOTAL_TEAMS = 68
SEEDS = list(range(1, 17))
TEAMS_PER_SEED = {s: 4 for s in SEEDS}
TEAMS_PER_SEED[11] = 6  # First Four at-large
TEAMS_PER_SEED[16] = 6  # First Four auto-bid
REGIONS = ["East", "West", "South", "Midwest"]

# Features used in modeling
BASIC_STAT_FEATURES = [
    "wins", "losses", "win_pct",
    "conf_wins", "conf_losses", "conf_win_pct",
    "srs",
]

ADVANCED_STAT_FEATURES = [
    "drtg", "nrtg",
]

COMMITTEE_FEATURES = [
    # Results-based rankings (WarrenNolan)
    "net_ranking", "net_sos",
    "kpi", "sor",
    # Predictive rankings (WarrenNolan)
    "bpi", "pom",
    # Torvik metrics (actual values)
    "wab", "barthag", "adj_oe", "adj_de",
    # Top quadrant records (Torvik)
    "q1_wins",
]

CONFERENCE_FEATURES = [
    "conf_strength",
    "conf_tourney_winner",
]

ALL_FEATURES = (
    BASIC_STAT_FEATURES
    + ADVANCED_STAT_FEATURES
    + COMMITTEE_FEATURES
    + CONFERENCE_FEATURES
)

# Model hyperparameters
SELECTION_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

SEEDING_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}
