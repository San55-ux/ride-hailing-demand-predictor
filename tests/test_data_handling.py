"""
Unit Tests for Data Generation, Spatio-Temporal Aggregation, and Feature Engineering.
"""

import pytest
import numpy as np
import pandas as pd
from data.generator import METRO_ZONES, generate_synthetic_trips, haversine_distance_km
from data.preprocessor import (
    aggregate_spatio_temporal_demand,
    engineer_features,
    prepare_train_test_split,
    FEATURE_COLUMNS,
)


def test_synthetic_trip_generator():
    """Verify that generated trip records satisfy schema and domain constraints."""
    df = generate_synthetic_trips(days=2, sample_rate=0.1, random_seed=42)

    assert not df.empty
    expected_cols = [
        "request_id",
        "timestamp",
        "pickup_zone_id",
        "pickup_zone_name",
        "pickup_lat",
        "pickup_lon",
        "dropoff_zone_id",
        "dropoff_zone_name",
        "dropoff_lat",
        "dropoff_lon",
        "trip_distance_km",
        "trip_duration_min",
        "surge_multiplier",
        "base_fare",
        "total_fare",
        "driver_earnings",
        "weather",
        "hour",
        "day_of_week",
        "is_weekend",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column {col}"

    # Verify value ranges
    assert (df["surge_multiplier"] >= 1.0).all()
    assert (df["trip_distance_km"] > 0).all()
    assert (df["total_fare"] > 0).all()
    assert (df["driver_earnings"] > 0).all()
    assert (df["hour"] >= 0).all() and (df["hour"] <= 23).all()
    assert (df["day_of_week"] >= 0).all() and (df["day_of_week"] <= 6).all()


def test_haversine_distance():
    """Verify haversine formula with known coordinates."""
    # Distance from SF Downtown (~37.7891, -122.4014) to SFO Airport (~37.6213, -122.3790) is ~19 km
    dist = haversine_distance_km(37.7891, -122.4014, 37.6213, -122.3790)
    assert 17.0 <= dist <= 21.0

    # Distance to self must be 0
    assert haversine_distance_km(37.7891, -122.4014, 37.7891, -122.4014) == 0.0


def test_spatio_temporal_demand_aggregation():
    """Verify spatial grid binning and zero-demand padding."""
    trips_df = generate_synthetic_trips(days=2, sample_rate=0.1, random_seed=42)
    grid_df = aggregate_spatio_temporal_demand(trips_df, freq="1h")

    assert not grid_df.empty
    assert "demand_count" in grid_df.columns
    assert "pickup_zone_id" in grid_df.columns
    assert "time_bin" in grid_df.columns

    # Every time bin must have entries for ALL zones
    num_zones = len(METRO_ZONES)
    time_bins_count = grid_df["time_bin"].nunique()
    expected_total_rows = time_bins_count * num_zones
    assert len(grid_df) == expected_total_rows

    # Verify no null values in core fields
    assert not grid_df["demand_count"].isnull().any()
    assert (grid_df["demand_count"] >= 0).all()
    assert not grid_df["avg_surge"].isnull().any()


def test_feature_engineering():
    """Verify cyclical transformations, domain flags, and lag feature creation."""
    trips_df = generate_synthetic_trips(days=3, sample_rate=0.1, random_seed=42)
    grid_df = aggregate_spatio_temporal_demand(trips_df, freq="1h")
    feat_df = engineer_features(grid_df)

    # Check cyclical ranges [-1, 1]
    assert feat_df["hour_sin"].between(-1.0001, 1.0001).all()
    assert feat_df["hour_cos"].between(-1.0001, 1.0001).all()
    assert feat_df["dow_sin"].between(-1.0001, 1.0001).all()
    assert feat_df["dow_cos"].between(-1.0001, 1.0001).all()

    # Check binary flags
    assert set(feat_df["is_weekend"].unique()).issubset({0, 1})
    assert set(feat_df["is_morning_rush"].unique()).issubset({0, 1})
    assert set(feat_df["is_evening_rush"].unique()).issubset({0, 1})
    assert set(feat_df["weather_rain"].unique()).issubset({0, 1})

    # Check lags exist and are non-null
    assert "lag_1" in feat_df.columns
    assert not feat_df["lag_1"].isnull().any()
    assert "rolling_mean_3" in feat_df.columns
    assert not feat_df["rolling_mean_3"].isnull().any()


def test_train_test_split_temporal_integrity():
    """Ensure train/test split has no temporal leakage (train timestamps < test timestamps)."""
    trips_df = generate_synthetic_trips(days=4, sample_rate=0.1, random_seed=42)
    grid_df = aggregate_spatio_temporal_demand(trips_df, freq="1h")
    feat_df = engineer_features(grid_df)

    X_train, X_test, y_train, y_test = prepare_train_test_split(feat_df, test_ratio=0.25)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(feat_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    train_max_time = feat_df.loc[X_train.index, "time_bin"].max()
    test_min_time = feat_df.loc[X_test.index, "time_bin"].min()
    assert train_max_time < test_min_time
