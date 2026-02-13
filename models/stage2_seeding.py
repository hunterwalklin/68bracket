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

        Ensures exactly TEAMS_PER_SEED teams at each seed line.
        (4 per seed, 6 at seeds 11 and 16 for First Four)
        """
        df = df.copy()
        df["raw_seed"] = self.predict_raw(df)

        # Clip to valid range
        df["raw_seed"] = df["raw_seed"].clip(1, 16)

        # Sort by raw seed prediction
        df = df.sort_values("raw_seed").reset_index(drop=True)

        # Assign constrained seeds
        assigned_seeds = []
        idx = 0
        for seed in SEEDS:
            n_teams = TEAMS_PER_SEED[seed]
            for _ in range(n_teams):
                if idx < len(df):
                    assigned_seeds.append(seed)
                    idx += 1

        # Pad if needed (shouldn't happen with 68 teams)
        while len(assigned_seeds) < len(df):
            assigned_seeds.append(16)

        df["predicted_seed"] = assigned_seeds[:len(df)]

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
