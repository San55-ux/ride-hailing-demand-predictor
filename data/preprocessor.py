"""
Data Preprocessing and Feature Engineering Module.
Transforms raw ride requests into spatio-temporal demand aggregations with cyclical and spatial features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from .generator import METRO_ZONES, ZONES_DF


def aggregate_spatio_temporal_demand(
    trips_df: pd.DataFrame,
    freq: str = "1h",
) -> pd.DataFrame:
    """
    Aggregates raw individual trip requests into discrete spatio-temporal grid bins.
    Guarantees that empty intervals (zero demand) are represented.

    Args:
        trips_df: DataFrame of raw ride requests with 'timestamp' and 'pickup_zone_id'.
        freq: Pandas time frequency string ('15min', '30min', '1h').

    Returns:
        DataFrame with aggregated demand per (timestamp_bin, zone_id).
    """
    df = trips_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["time_bin"] = df["timestamp"].dt.floor(freq)

    # Aggregate counts, surge, and earnings
    agg = df.groupby(["time_bin", "pickup_zone_id"]).agg(
        demand_count=("request_id", "count"),
        avg_surge=("surge_multiplier", "mean"),
        avg_fare=("total_fare", "mean"),
        avg_distance_km=("trip_distance_km", "mean"),
        total_earnings=("driver_earnings", "sum"),
        weather=("weather", lambda x: x.mode()[0] if not x.empty else "Clear"),
    ).reset_index()

    # Re-index across full Cartesian product of (time_bins x all zones) to include zero-demand bins
    all_time_bins = pd.date_range(
        start=df["time_bin"].min(),
        end=df["time_bin"].max(),
        freq=freq,
    )
    all_zones = [z["zone_id"] for z in METRO_ZONES]

    full_grid = pd.MultiIndex.from_product(
        [all_time_bins, all_zones],
        names=["time_bin", "pickup_zone_id"],
    ).to_frame().reset_index(drop=True)

    merged = pd.merge(full_grid, agg, on=["time_bin", "pickup_zone_id"], how="left")
    merged["demand_count"] = merged["demand_count"].fillna(0).astype(int)
    merged["avg_surge"] = merged["avg_surge"].fillna(1.0)
    merged["avg_fare"] = merged["avg_fare"].fillna(0.0)
    merged["avg_distance_km"] = merged["avg_distance_km"].fillna(0.0)
    merged["total_earnings"] = merged["total_earnings"].fillna(0.0)
    merged["weather"] = merged["weather"].ffill().bfill().fillna("Clear")

    # Merge static zone attributes (name, lat, lon, type)
    zone_lookup = ZONES_DF[["zone_id", "name", "lat", "lon", "type"]].rename(
        columns={"zone_id": "pickup_zone_id", "name": "zone_name"}
    )
    merged = pd.merge(merged, zone_lookup, on="pickup_zone_id", how="left")
    merged.sort_values(by=["time_bin", "pickup_zone_id"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    return merged


def engineer_features(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs rich feature matrix for ML models:
    - Cyclical hour and day-of-week transforms (sin/cos)
    - Rush hour and nightlife flags
    - Weather one-hot encoding
    - Zone spatial coordinates & category flags
    - Lagged demand features (lag-1, lag-24, rolling 3-interval mean)
    """
    df = demand_df.copy()

    # Temporal features
    df["hour"] = df["time_bin"].dt.hour
    df["day_of_week"] = df["time_bin"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Domain indicators
    df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 9) & (df["is_weekend"] == 0)).astype(int)
    df["is_evening_rush"] = ((df["hour"] >= 16) & (df["hour"] <= 19) & (df["is_weekend"] == 0)).astype(int)
    df["is_nightlife"] = (((df["hour"] >= 21) | (df["hour"] <= 2)) & (df["is_weekend"] == 1)).astype(int)

    # Weather encoding
    df["weather_rain"] = (df["weather"] == "Rain").astype(int)
    df["weather_fog"] = (df["weather"] == "Fog").astype(int)
    df["weather_cloudy"] = (df["weather"] == "Cloudy").astype(int)

    # Lag features per zone (sorted by zone and time)
    df.sort_values(by=["pickup_zone_id", "time_bin"], inplace=True)
    df["lag_1"] = df.groupby("pickup_zone_id")["demand_count"].shift(1).fillna(0)
    df["lag_2"] = df.groupby("pickup_zone_id")["demand_count"].shift(2).fillna(0)
    df["lag_24"] = df.groupby("pickup_zone_id")["demand_count"].shift(24).fillna(0)
    df["rolling_mean_3"] = (
        df.groupby("pickup_zone_id")["demand_count"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    # One-hot encode zone types
    zone_type_dummies = pd.get_dummies(df["type"], prefix="zone_type", dtype=int)
    df = pd.concat([df, zone_type_dummies], axis=1)

    df.sort_values(by=["time_bin", "pickup_zone_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


FEATURE_COLUMNS = [
    "pickup_zone_id",
    "lat",
    "lon",
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_morning_rush",
    "is_evening_rush",
    "is_nightlife",
    "weather_rain",
    "weather_fog",
    "weather_cloudy",
    "lag_1",
    "lag_2",
    "lag_24",
    "rolling_mean_3",
]


def prepare_train_test_split(
    featured_df: pd.DataFrame,
    test_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological train/test split to prevent temporal data leakage.
    """
    unique_times = sorted(featured_df["time_bin"].unique())
    split_idx = int(len(unique_times) * (1.0 - test_ratio))
    split_time = unique_times[split_idx]

    train_mask = featured_df["time_bin"] < split_time
    test_mask = featured_df["time_bin"] >= split_time

    # Find available feature columns
    avail_features = [col for col in FEATURE_COLUMNS if col in featured_df.columns]
    for col in featured_df.columns:
        if col.startswith("zone_type_") and col not in avail_features:
            avail_features.append(col)

    X_train = featured_df.loc[train_mask, avail_features]
    y_train = featured_df.loc[train_mask, "demand_count"]
    X_test = featured_df.loc[test_mask, avail_features]
    y_test = featured_df.loc[test_mask, "demand_count"]

    return X_train, X_test, y_train, y_test
