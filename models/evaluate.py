"""Leave-one-season-out cross-validation for model evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import ALL_FEATURES, TRAINING_SEASONS, TOTAL_TEAMS
from models.stage1_selection import SelectionModel
from models.stage2_seeding import SeedingModel


def leave_one_season_out_cv(df: pd.DataFrame, seasons: list[int] | None = None) -> dict:
    """Run leave-one-season-out cross-validation.

    For each season:
    1. Train on all other seasons
    2. Predict on held-out season
    3. Evaluate selection accuracy and seed MAE

    Returns dict with per-season and aggregate metrics.
    """
    if seasons is None:
        seasons = [s for s in TRAINING_SEASONS if s in df["season"].unique()]

    results = []

    for holdout_season in seasons:
        print(f"\n  Evaluating holdout season: {holdout_season}")

        # Split
        train_mask = (df["season"] != holdout_season) & (df["season"].isin(TRAINING_SEASONS))
        test_mask = df["season"] == holdout_season

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if train_df.empty or test_df.empty:
            print(f"    Skipping {holdout_season}: insufficient data")
            continue

        X_train = train_df[ALL_FEATURES]
        y_train_sel = train_df["made_tournament"]
        y_train_seed = train_df["seed"]

        X_test = test_df[ALL_FEATURES]
        y_test_sel = test_df["made_tournament"]
        y_test_seed = test_df["seed"]

        # Stage 1: Selection
        sel_model = SelectionModel()
        sel_model.train(X_train, y_train_sel)

        # Get predicted field
        test_with_probs = test_df.copy()
        test_with_probs["selection_prob"] = sel_model.predict_proba(test_df)
        predicted_field = sel_model.select_field(test_with_probs)

        # Evaluate selection
        actual_teams = set(test_df[test_df["made_tournament"] == 1].index)
        predicted_teams = set(predicted_field.index) if not predicted_field.empty else set()

        # Use school_id for matching since index might differ
        if "school_id" in test_df.columns:
            actual_ids = set(test_df[test_df["made_tournament"] == 1]["school_id"])
            predicted_ids = set(predicted_field["school_id"]) if not predicted_field.empty else set()
            correct_selections = len(actual_ids & predicted_ids)
            total_actual = len(actual_ids)
        else:
            correct_selections = len(actual_teams & predicted_teams)
            total_actual = len(actual_teams)

        # Stage 2: Seeding
        seed_model = SeedingModel()
        tourn_mask = y_train_seed.notna()
        seed_model.train(X_train[tourn_mask], y_train_seed[tourn_mask])

        # Evaluate seed prediction on actual tournament teams
        actual_tourn = test_df[test_df["made_tournament"] == 1].copy()
        if not actual_tourn.empty and actual_tourn["seed"].notna().any():
            actual_tourn["predicted_raw_seed"] = seed_model.predict_raw(actual_tourn)
            seed_valid = actual_tourn["seed"].notna()
            mae = np.mean(np.abs(
                actual_tourn.loc[seed_valid, "predicted_raw_seed"] -
                actual_tourn.loc[seed_valid, "seed"]
            ))

            # Also compute constrained seed assignment
            seeded = seed_model.assign_seeds(predicted_field)
            # Match back to actual seeds
            if "school_id" in seeded.columns:
                merged = seeded.merge(
                    actual_tourn[["school_id", "seed"]].rename(columns={"seed": "actual_seed"}),
                    on="school_id", how="inner"
                )
                if not merged.empty:
                    constrained_mae = np.mean(np.abs(
                        merged["predicted_seed"] - merged["actual_seed"]
                    ))
                else:
                    constrained_mae = np.nan
            else:
                constrained_mae = np.nan
        else:
            mae = np.nan
            constrained_mae = np.nan

        season_result = {
            "season": holdout_season,
            "correct_selections": correct_selections,
            "total_actual": total_actual,
            "selection_accuracy": correct_selections / total_actual if total_actual > 0 else 0,
            "seed_mae_raw": mae,
            "seed_mae_constrained": constrained_mae,
        }
        results.append(season_result)

        print(f"    Selection: {correct_selections}/{total_actual} correct "
              f"({season_result['selection_accuracy']:.1%})")
        print(f"    Seed MAE (raw): {mae:.2f}" if not np.isnan(mae) else "    Seed MAE: N/A")
        if not np.isnan(constrained_mae):
            print(f"    Seed MAE (constrained): {constrained_mae:.2f}")

    # Aggregate results
    if not results:
        return {"per_season": [], "aggregate": {}}

    results_df = pd.DataFrame(results)
    aggregate = {
        "mean_selection_accuracy": results_df["selection_accuracy"].mean(),
        "mean_correct_selections": results_df["correct_selections"].mean(),
        "mean_seed_mae_raw": results_df["seed_mae_raw"].mean(),
        "mean_seed_mae_constrained": results_df["seed_mae_constrained"].mean(),
        "min_correct": results_df["correct_selections"].min(),
        "max_correct": results_df["correct_selections"].max(),
    }

    print(f"\n  === Aggregate Results ({len(results)} seasons) ===")
    print(f"  Mean selection accuracy: {aggregate['mean_selection_accuracy']:.1%}")
    print(f"  Mean correct selections: {aggregate['mean_correct_selections']:.1f}/{TOTAL_TEAMS}")
    print(f"  Selection range: {aggregate['min_correct']}-{aggregate['max_correct']}/{TOTAL_TEAMS}")
    print(f"  Mean seed MAE (raw): {aggregate['mean_seed_mae_raw']:.2f}")
    if not np.isnan(aggregate['mean_seed_mae_constrained']):
        print(f"  Mean seed MAE (constrained): {aggregate['mean_seed_mae_constrained']:.2f}")

    return {"per_season": results, "aggregate": aggregate}


def print_feature_importance(df: pd.DataFrame):
    """Train on all data and print feature importances for both models."""
    X = df[ALL_FEATURES]
    y_sel = df["made_tournament"]
    y_seed = df["seed"]

    print("\n  === Selection Model Feature Importance ===")
    sel_model = SelectionModel()
    sel_model.train(X, y_sel)
    sel_imp = sel_model.feature_importance()
    for _, row in sel_imp.head(15).iterrows():
        print(f"    {row['feature']:25s} {row['importance']:.4f}")

    print("\n  === Seeding Model Feature Importance ===")
    seed_model = SeedingModel()
    tourn_mask = y_seed.notna()
    seed_model.train(X[tourn_mask], y_seed[tourn_mask])
    seed_imp = seed_model.feature_importance()
    for _, row in seed_imp.head(15).iterrows():
        print(f"    {row['feature']:25s} {row['importance']:.4f}")
