"""Stage 1: Classifier for tournament selection (RF or XGBoost)."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

from config import (
    SELECTION_RF_PARAMS, SELECTION_XGB_PARAMS,
    ALL_FEATURES, MODEL_DIR, TOTAL_TEAMS,
)


class SelectionModel:
    """Predict which teams make the NCAA tournament."""

    def __init__(self):
        self.model = RandomForestClassifier(**SELECTION_RF_PARAMS)
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the selection classifier."""
        X_clean = X[ALL_FEATURES].fillna(0)
        self.model.fit(X_clean, y)
        self.is_fitted = True

        train_acc = self.model.score(X_clean, y)
        n_pos = y.sum()
        print(f"  Selection model trained: accuracy={train_acc:.3f}, "
              f"tournament teams in training={n_pos}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of making the tournament."""
        X_clean = X[ALL_FEATURES].fillna(0)
        return self.model.predict_proba(X_clean)[:, 1]

    def select_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the 68-team tournament field.

        Logic:
        1. Auto-bids: 1 team per conference (conf_tourney_winner or highest prob)
        2. At-large: Fill remaining spots with highest probability teams

        Returns the input DataFrame filtered to 68 teams with a 'selection_prob' column.
        """
        df = df.copy()
        df["selection_prob"] = self.predict_proba(df)

        selected = []
        remaining = df.copy()

        # Step 1: Auto-bids (one per conference)
        if "conference" in df.columns:
            for conf in df["conference"].unique():
                conf_teams = df[df["conference"] == conf]

                # Prefer explicit conf tourney winner
                winners = conf_teams[conf_teams.get("conf_tourney_winner", 0) == 1]
                if not winners.empty:
                    best = winners.sort_values("selection_prob", ascending=False).iloc[0]
                else:
                    # Otherwise pick highest probability team from conference
                    best = conf_teams.sort_values("selection_prob", ascending=False).iloc[0]

                selected.append(best.name)  # index

        # Remove selected teams from remaining pool
        remaining = remaining.drop(index=selected, errors="ignore")

        # Step 2: Fill remaining at-large spots
        n_at_large = TOTAL_TEAMS - len(selected)
        if n_at_large > 0:
            at_large = remaining.sort_values("selection_prob", ascending=False).head(n_at_large)
            selected.extend(at_large.index.tolist())

        # If we somehow have too many (more conferences than expected), trim
        if len(selected) > TOTAL_TEAMS:
            # Keep the auto-bids that have highest probability
            all_selected = df.loc[selected].sort_values("selection_prob", ascending=False)
            selected = all_selected.head(TOTAL_TEAMS).index.tolist()

        result = df.loc[selected].copy()
        result["selection_method"] = "auto_bid"
        result.loc[result.index.isin(selected[len(selected) - n_at_large:]), "selection_method"] = "at_large"

        return result.sort_values("selection_prob", ascending=False).reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted by importance."""
        if not self.is_fitted:
            return pd.DataFrame()

        importance = pd.DataFrame({
            "feature": ALL_FEATURES,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return importance

    def save(self, filename: str = "selection_model.joblib"):
        """Save model to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(self.model, path)
        print(f"  Saved selection model to {path}")

    def load(self, filename: str = "selection_model.joblib"):
        """Load model from disk."""
        path = os.path.join(MODEL_DIR, filename)
        self.model = joblib.load(path)
        self.is_fitted = True


class XGBSelectionModel(SelectionModel):
    """XGBoost variant of the selection classifier."""

    def __init__(self):
        self.model = XGBClassifier(**SELECTION_XGB_PARAMS)
        self.is_fitted = False

    def save(self, filename: str = "selection_model_xgb.joblib"):
        super().save(filename)

    def load(self, filename: str = "selection_model_xgb.joblib"):
        super().load(filename)


class EnsembleSelectionModel(SelectionModel):
    """Averages RF and XGBoost selection probabilities."""

    def __init__(self):
        self.rf = SelectionModel()
        self.xgb = XGBSelectionModel()
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.rf.train(X, y)
        self.xgb.train(X, y)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return (self.rf.predict_proba(X) + self.xgb.predict_proba(X)) / 2

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


def get_selection_model(model_type: str = "rf") -> SelectionModel:
    """Factory: return the right selection model class."""
    if model_type == "xgb":
        return XGBSelectionModel()
    if model_type == "ensemble":
        return EnsembleSelectionModel()
    return SelectionModel()
