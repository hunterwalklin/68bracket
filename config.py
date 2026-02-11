"""Configuration constants for 68bracket."""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Sports-Reference base URLs
SR_BASE = "https://www.sports-reference.com/cbb"
SR_SEASONS_URL = SR_BASE + "/seasons/{season}-school-stats.html"
SR_ADVANCED_URL = SR_BASE + "/seasons/{season}-advanced-school-stats.html"
SR_RATINGS_URL = SR_BASE + "/seasons/{season}-ratings.html"
SR_POLLS_URL = SR_BASE + "/seasons/{season}-polls.html"
SR_TOURNAMENT_URL = SR_BASE + "/postseason/{season}-ncaa.html"
SR_SCHOOL_URL = SR_BASE + "/schools/{school}/{season}-schedule.html"

# Scraping settings
REQUEST_DELAY = 3.5  # seconds between requests (Sports-Reference rate limit)
REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = "68bracket/1.0 (NCAA Tournament Prediction Research)"
CACHE_EXPIRY_DAYS = 7  # re-scrape after this many days

# Training seasons (skip 2020 due to COVID cancellation)
TRAINING_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
PREDICTION_SEASON = 2026

# Tournament structure
TOTAL_TEAMS = 68
SEEDS = list(range(1, 17))
TEAMS_PER_SEED = {s: 4 for s in SEEDS}
TEAMS_PER_SEED[11] = 6  # First Four at-large
TEAMS_PER_SEED[16] = 6  # First Four auto-bid
REGIONS = ["East", "West", "South", "Midwest"]

# Power conferences (current era)
POWER_CONFERENCES = [
    "ACC", "Big 12", "Big East", "Big Ten", "SEC",
]

# All D1 conferences (2024-25 alignment, 32 conferences + independents)
ALL_CONFERENCES = [
    "ACC", "American", "A-10", "A-Sun", "Big 12", "Big East", "Big Sky",
    "Big South", "Big Ten", "Big West", "CAA", "CUSA", "Horizon",
    "Ivy", "MAAC", "MAC", "MEAC", "MVC", "MWC", "NEC", "OVC",
    "Pac-12", "Patriot", "SEC", "SBC", "Southern", "Southland",
    "Summit", "SWAC", "WAC", "WCC",
]

# Features used in modeling
BASIC_STAT_FEATURES = [
    "wins", "losses", "win_pct",
    "conf_wins", "conf_losses", "conf_win_pct",
    "srs", "sos",
    "pts_per_game", "opp_pts_per_game",
]

ADVANCED_STAT_FEATURES = [
    "pace", "ortg", "drtg", "nrtg",
    "efg_pct", "tov_pct", "orb_pct", "ft_rate",
    "opp_efg_pct", "opp_tov_pct", "opp_orb_pct", "opp_ft_rate",
]

RATING_FEATURES = [
    "ap_rank", "ap_weeks_ranked",
    "coaches_rank",
]

CONFERENCE_FEATURES = [
    "power_conf",
    "conf_strength",
    "conf_tourney_winner",
]

ALL_FEATURES = (
    BASIC_STAT_FEATURES
    + ADVANCED_STAT_FEATURES
    + RATING_FEATURES
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
