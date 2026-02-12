"""CLI entry point for 68bracket: NCAA Tournament Seed Prediction."""

import argparse
import os
import sys

import pandas as pd

from config import (
    TRAINING_SEASONS, PREDICTION_SEASON, PROCESSED_DIR,
    ALL_FEATURES, MODEL_DIR,
)


def cmd_scrape(args):
    """Scrape all data sources."""
    from data.scrape_team_stats import TeamStatsScraper
    from data.scrape_tournament import TournamentScraper
    from data.scrape_nitty_gritty import NittyGrittyScraper
    from data.scrape_torvik import TorvikvScraper
    from data.scrape_conferences import ConferenceScraper

    current_only = getattr(args, "current_only", False)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if current_only:
        # Fast path: only re-scrape the current season and upsert into existing data.
        # Tournament brackets and conference tourneys are historical — skip them.
        seasons = [PREDICTION_SEASON]
        scrapers = [
            ("team stats", TeamStatsScraper(), "scrape_all_seasons", "team_stats.parquet"),
            ("WarrenNolan rankings", NittyGrittyScraper(), "scrape_all_seasons", "nitty_gritty.parquet"),
            ("Torvik ratings", TorvikvScraper(), "scrape_all_seasons", "torvik.parquet"),
        ]

        print(f"Refreshing current season ({PREDICTION_SEASON}) only...")

        for name, scraper, method, filename in scrapers:
            print(f"\nScraping {name} ({PREDICTION_SEASON})...")
            try:
                scraper.force_refresh = True
                fresh_df = getattr(scraper, method)(seasons)
                if fresh_df.empty:
                    print(f"  Warning: no data collected for {name}")
                    continue

                # Upsert: load existing, drop current season rows, append fresh
                path = os.path.join(PROCESSED_DIR, filename)
                if os.path.exists(path):
                    existing_df = pd.read_parquet(path)
                    existing_df = existing_df[existing_df["season"] != PREDICTION_SEASON]
                    df = pd.concat([existing_df, fresh_df], ignore_index=True)
                else:
                    df = fresh_df

                df.to_parquet(path, index=False)
                print(f"  Saved {len(df)} rows to {filename} "
                      f"({len(fresh_df)} fresh for {PREDICTION_SEASON})")
            except Exception as e:
                print(f"  Error scraping {name}: {e}")
                existing = os.path.join(PROCESSED_DIR, filename)
                if os.path.exists(existing):
                    print(f"  Keeping previous {filename}")
    else:
        # Full scrape: all seasons, overwrites parquet files entirely.
        seasons = TRAINING_SEASONS.copy()
        if args.include_current:
            seasons.append(PREDICTION_SEASON)

        scrapers = [
            ("team stats", TeamStatsScraper(), "scrape_all_seasons", seasons, "team_stats.parquet"),
            ("tournament brackets", TournamentScraper(), "scrape_all_seasons", TRAINING_SEASONS, "tournament.parquet"),
            ("WarrenNolan rankings", NittyGrittyScraper(), "scrape_all_seasons", seasons, "nitty_gritty.parquet"),
            ("Torvik ratings", TorvikvScraper(), "scrape_all_seasons", seasons, "torvik.parquet"),
            ("conference tournaments", ConferenceScraper(), "scrape_all_seasons", TRAINING_SEASONS, "conferences.parquet"),
        ]

        for name, scraper, method, szns, filename in scrapers:
            print(f"\nScraping {name}...")
            try:
                df = getattr(scraper, method)(szns)
                if not df.empty:
                    df.to_parquet(os.path.join(PROCESSED_DIR, filename), index=False)
                    print(f"  Saved {len(df)} rows to {filename}")
                else:
                    print(f"  Warning: no data collected for {name}")
            except Exception as e:
                print(f"  Error scraping {name}: {e}")
                existing = os.path.join(PROCESSED_DIR, filename)
                if os.path.exists(existing):
                    print(f"  Keeping previous {filename}")

    print("\nScraping complete!")


def cmd_build(args):
    """Build feature matrix from scraped data."""
    from features.build_features import build_features, save_features

    print("Loading scraped data...")
    team_stats = _load_parquet("team_stats.parquet")
    tournament = _load_parquet("tournament.parquet")
    nitty_gritty = _load_parquet("nitty_gritty.parquet")
    torvik = _load_parquet("torvik.parquet")
    conferences = _load_parquet("conferences.parquet")

    print("Building features...")
    df = build_features(team_stats, tournament, nitty_gritty, torvik, conferences)
    save_features(df)

    # Print summary
    n_seasons = df["season"].nunique()
    n_teams = len(df)
    n_tourney = df["made_tournament"].sum()
    print(f"\n  Feature matrix: {n_teams} team-seasons across {n_seasons} seasons")
    print(f"  Tournament teams: {n_tourney}")
    print(f"  Features: {len(ALL_FEATURES)}")


def cmd_train(args):
    """Train both models on all training data."""
    from features.build_features import load_features, get_training_data
    from models.stage1_selection import SelectionModel
    from models.stage2_seeding import SeedingModel

    print("Loading features...")
    df = load_features()
    X, y_selection, y_seed = get_training_data(df)

    print("\nTraining Stage 1: Selection model...")
    sel_model = SelectionModel()
    sel_model.train(X, y_selection)
    sel_model.save()

    print("\nTraining Stage 2: Seeding model...")
    seed_model = SeedingModel()
    tourn_mask = y_seed.notna()
    seed_model.train(X[tourn_mask], y_seed[tourn_mask])
    seed_model.save()

    print("\nTraining complete!")


def cmd_evaluate(args):
    """Run leave-one-season-out cross-validation."""
    from features.build_features import load_features
    from models.evaluate import leave_one_season_out_cv, print_feature_importance

    print("Loading features...")
    df = load_features()

    print("\nRunning leave-one-season-out cross-validation...")
    results = leave_one_season_out_cv(df)

    if args.importance:
        print_feature_importance(df)


def cmd_predict(args):
    """Generate predictions for the current season."""
    from features.build_features import load_features, get_prediction_data
    from models.stage1_selection import SelectionModel
    from models.stage2_seeding import SeedingModel
    from output.bracket_builder import assign_regions
    from output.display import (
        display_seed_list, display_bracket,
        display_selection_summary, generate_markdown,
    )

    season = args.season or PREDICTION_SEASON

    print(f"Loading features for {season}...")
    df = load_features()
    pred_df = get_prediction_data(df, season=season)

    if pred_df.empty:
        print(f"Error: no data for season {season}. Run 'scrape --include-current' first.")
        sys.exit(1)

    print(f"  {len(pred_df)} teams found for {season}")

    # Load trained models
    sel_model = SelectionModel()
    sel_model.load()
    seed_model = SeedingModel()
    seed_model.load()

    # Stage 1: Select 68 teams
    print("\nStage 1: Selecting tournament field...")
    field = sel_model.select_field(pred_df)
    print(f"  Selected {len(field)} teams")

    # Stage 2: Assign seeds
    print("Stage 2: Assigning seeds...")
    seeded = seed_model.assign_seeds(field)

    # Assign regions
    print("Assigning regions...")
    bracket = assign_regions(seeded)

    # Display results
    display_seed_list(bracket, season=season)
    display_selection_summary(bracket)
    display_bracket(bracket, season=season)

    # Save markdown output
    md = generate_markdown(bracket, season=season)
    md_path = os.path.join(PROCESSED_DIR, f"predictions_{season}.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\n  Markdown saved to {md_path}")


def cmd_all(args):
    """Run the full pipeline: scrape, build, train, evaluate, predict."""
    print("=" * 60)
    print("  68bracket - Full Pipeline")
    print("=" * 60)

    args.include_current = True
    print("\n[1/5] Scraping data...")
    cmd_scrape(args)

    print("\n[2/5] Building features...")
    cmd_build(args)

    print("\n[3/5] Training models...")
    cmd_train(args)

    print("\n[4/5] Evaluating models...")
    args.importance = True
    cmd_evaluate(args)

    print("\n[5/5] Generating predictions...")
    args.season = PREDICTION_SEASON
    cmd_predict(args)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


def _load_parquet(filename: str) -> pd.DataFrame:
    """Load a parquet file from the processed directory."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, returning empty DataFrame")
        return pd.DataFrame()
    return pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(
        description="68bracket: NCAA Tournament Seed Prediction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  scrape     Scrape data from Sports-Reference
  build      Build feature matrix from scraped data
  train      Train selection and seeding models
  evaluate   Run leave-one-season-out cross-validation
  predict    Generate predictions for a season
  all        Run the full pipeline
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scrape
    scrape_parser = subparsers.add_parser("scrape", help="Scrape data from Sports-Reference")
    scrape_parser.add_argument("--include-current", action="store_true",
                               help="Also scrape the current/prediction season")
    scrape_parser.add_argument("--current-only", action="store_true",
                               help="Only re-scrape the current season (fast daily refresh)")

    # Build
    subparsers.add_parser("build", help="Build feature matrix")

    # Train
    subparsers.add_parser("train", help="Train models")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run cross-validation")
    eval_parser.add_argument("--importance", action="store_true",
                             help="Show feature importance analysis")

    # Predict
    pred_parser = subparsers.add_parser("predict", help="Generate predictions")
    pred_parser.add_argument("--season", type=int, default=None,
                             help=f"Season to predict (default: {PREDICTION_SEASON})")

    # All
    subparsers.add_parser("all", help="Run the full pipeline")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "scrape": cmd_scrape,
        "build": cmd_build,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "all": cmd_all,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
