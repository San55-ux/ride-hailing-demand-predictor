"""
Model Evaluator and Performance Analytics.
Computes error metrics, residual distributions, and zone-level prediction accuracies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_label: str = "Model",
) -> Dict[str, Any]:
    """
    Computes standard regression error metrics and distribution analytics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (handling zeros safely)
    non_zero_mask = y_true > 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
    else:
        mape = 0.0

    residuals = y_pred - y_true

    return {
        "model_label": model_label,
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 4),
        "MAPE_pct": round(mape, 2),
        "mean_residual": round(float(np.mean(residuals)), 3),
        "std_residual": round(float(np.std(residuals)), 3),
        "max_overprediction": round(float(np.max(residuals)), 2),
        "max_underprediction": round(float(np.abs(np.min(residuals))), 2),
    }


def compute_zone_level_breakdown(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    zone_name_col: str = "zone_name",
) -> pd.DataFrame:
    """
    Calculates per-zone accuracy metrics to identify which geographic areas are hardest to predict.
    """
    records = []
    for zone_name, group in df.groupby(zone_name_col):
        y_t = group[y_true_col].values
        y_p = group[y_pred_col].values
        mae = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        avg_demand = np.mean(y_t)
        records.append({
            "zone_name": zone_name,
            "avg_actual_demand": round(float(avg_demand), 1),
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "rel_error_pct": round(float((mae / max(avg_demand, 1.0)) * 100), 1),
        })
    return pd.DataFrame(records).sort_values("avg_actual_demand", ascending=False).reset_index(drop=True)
