"""
Demand Prediction Models:
- GradientBoostingRegressor / RandomForestRegressor ML Model
- Historical Time-of-Week Heuristic Baseline
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HeuristicBaselineModel:
    """
    Baseline model that calculates historical average demand grouped by (pickup_zone_id, day_of_week, hour).
    Serves as an interpretable benchmark.
    """

    def __init__(self):
        self.lookup_table: Dict[Tuple[int, int, int], float] = {}
        self.global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "HeuristicBaselineModel":
        required_cols = {"pickup_zone_id", "day_of_week", "hour", "demand_count"}
        assert required_cols.issubset(train_df.columns), f"Missing required columns: {required_cols - set(train_df.columns)}"

        self.global_mean = float(train_df["demand_count"].mean())
        grouped = train_df.groupby(["pickup_zone_id", "day_of_week", "hour"])["demand_count"].mean()
        self.lookup_table = grouped.to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            key = (int(row["pickup_zone_id"]), int(row["day_of_week"]), int(row["hour"]))
            preds.append(self.lookup_table.get(key, self.global_mean))
        return np.array(preds)


class MLDemandModel:
    """
    Supervised Machine Learning model for forecasting ride-hailing demand across zones and time.
    """

    def __init__(self, model_type: str = "gradient_boosting", n_estimators: int = 120, max_depth: int = 6):
        self.model_type = model_type
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.08,
                random_state=42,
            )
        self.feature_names: list = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLDemandModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        # Ensure correct column ordering
        X_ordered = X[self.feature_names]
        preds = self.model.predict(X_ordered)
        # Demand counts cannot be negative
        return np.clip(preds, a_min=0, a_max=None)

    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return fi_df

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names, "type": self.model_type}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MLDemandModel":
        data = joblib.load(filepath)
        instance = cls(model_type=data["type"])
        instance.model = data["model"]
        instance.feature_names = data["features"]
        instance.is_fitted = True
        return instance


def evaluate_models(
    baseline: HeuristicBaselineModel,
    ml_model: MLDemandModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_featured_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Computes comparative evaluation metrics: MAE, RMSE, R2 for Baseline vs ML.
    """
    baseline_preds = baseline.predict(test_featured_df.loc[X_test.index])
    ml_preds = ml_model.predict(X_test)

    results = {
        "Heuristic Baseline": {
            "MAE": round(float(mean_absolute_error(y_test, baseline_preds)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, baseline_preds))), 3),
            "R2": round(float(r2_score(y_test, baseline_preds)), 3),
        },
        "ML Regressor": {
            "MAE": round(float(mean_absolute_error(y_test, ml_preds)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, ml_preds))), 3),
            "R2": round(float(r2_score(y_test, ml_preds)), 3),
        },
    }
    return results
