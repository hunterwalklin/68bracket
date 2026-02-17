"""Stage 2: Regressor for seed prediction (RF or XGBoost)."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

from config import (
    SEEDING_RF_PARAMS, SEEDING_XGB_PARAMS,
    ALL_FEATURES, MODEL_DIR, TEAMS_PER_SEED, SEEDS,
)


class SeedingModel:
    """Predict seed (1-16) for tournament teams."""

    def __init__(self):
        self.model = RandomForestRegressor(**SEEDING_RF_PARAMS)
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the seeding regressor on tournament teams only.

        X and y should be filtered to only teams that made the tournament.
        """
        # Drop rows where seed is missing
        valid = y.notna()
        X_clean = X.loc[valid, ALL_FEATURES].fillna(0)
        y_clean = y[valid]

        self.model.fit(X_clean, y_clean)
        self.is_fitted = True

        train_preds = self.model.predict(X_clean)
        mae = np.mean(np.abs(train_preds - y_clean))
        print(f"  Seeding model trained: MAE={mae:.2f} on {len(y_clean)} tournament teams")

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw seed values (continuous)."""
        X_clean = X[ALL_FEATURES].fillna(0)
        return self.model.predict(X_clean)

    def assign_seeds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign constrained seeds to the 68-team field.

        NCAA First Four rules:
        - Seed 11 First Four = 4 weakest at-large teams
        - Seed 16 First Four = 4 weakest auto-bid teams
        - Remaining 60 teams get 4 per seed at every line (1-16)
        """
        df = df.copy()
        df["raw_seed"] = self.predict_raw(df)
        df["raw_seed"] = df["raw_seed"].clip(1, 16)
        df["first_four"] = False

        # Split by selection method
        at_large = df[df["selection_method"] == "at_large"].sort_values("raw_seed")
        auto_bid = df[df["selection_method"] == "auto_bid"].sort_values("raw_seed")

        # 4 weakest at-large → seed 11 First Four
        ff_at_large_idx = at_large.index[-4:]
        df.loc[ff_at_large_idx, "first_four"] = True
        df.loc[ff_at_large_idx, "predicted_seed"] = 11

        # 4 weakest auto-bid → seed 16 First Four
        ff_auto_bid_idx = auto_bid.index[-4:]
        df.loc[ff_auto_bid_idx, "first_four"] = True
        df.loc[ff_auto_bid_idx, "predicted_seed"] = 16

        # Seed the remaining 60 teams: 4 per seed, except 2 at seeds 11/16
        # (FF winners fill the other 2 region slots at those seed lines)
        main_field = df[~df["first_four"]].sort_values("raw_seed")
        main_indices = main_field.index.tolist()
        assigned_seeds = []
        for seed in SEEDS:
            n = 2 if seed in (11, 16) else 4
            for _ in range(n):
                if len(assigned_seeds) < len(main_indices):
                    assigned_seeds.append(seed)

        for i, orig_idx in enumerate(main_indices):
            if i < len(assigned_seeds):
                df.loc[orig_idx, "predicted_seed"] = assigned_seeds[i]

        return df

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted by importance."""
        if not self.is_fitted:
            return pd.DataFrame()

        importance = pd.DataFrame({
            "feature": ALL_FEATURES,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return importance

    def save(self, filename: str = "seeding_model.joblib"):
        """Save model to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(self.model, path)
        print(f"  Saved seeding model to {path}")

    def load(self, filename: str = "seeding_model.joblib"):
        """Load model from disk."""
        path = os.path.join(MODEL_DIR, filename)
        self.model = joblib.load(path)
        self.is_fitted = True


class XGBSeedingModel(SeedingModel):
    """XGBoost variant of the seeding regressor."""

    def __init__(self):
        self.model = XGBRegressor(**SEEDING_XGB_PARAMS)
        self.is_fitted = False

    def save(self, filename: str = "seeding_model_xgb.joblib"):
        super().save(filename)

    def load(self, filename: str = "seeding_model_xgb.joblib"):
        super().load(filename)


class EnsembleSeedingModel(SeedingModel):
    """Averages RF and XGBoost seed predictions."""

    def __init__(self):
        self.rf = SeedingModel()
        self.xgb = XGBSeedingModel()
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.rf.train(X, y)
        self.xgb.train(X, y)
        self.is_fitted = True

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return (self.rf.predict_raw(X) + self.xgb.predict_raw(X)) / 2

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame()
        rf_imp = self.rf.feature_importance().set_index("feature")
        xgb_imp = self.xgb.feature_importance().set_index("feature")
        avg = ((rf_imp["importance"] + xgb_imp["importance"]) / 2).reset_index()
        avg.columns = ["feature", "importance"]
        return avg.sort_values("importance", ascending=False)

    def save(self, filename: str | None = None):
        self.rf.save()
        self.xgb.save()

    def load(self, filename: str | None = None):
        self.rf.load()
        self.xgb.load()
        self.is_fitted = True


def get_seeding_model(model_type: str = "rf") -> SeedingModel:
    """Factory: return the right seeding model class."""
    if model_type == "xgb":
        return XGBSeedingModel()
    if model_type == "ensemble":
        return EnsembleSeedingModel()
    return SeedingModel()
