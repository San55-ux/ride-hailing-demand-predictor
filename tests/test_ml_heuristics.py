"""
Unit Tests for Machine Learning Models, Heuristic Baselines, and Recommender.
"""

import pytest
import os
import tempfile
import numpy as np
import pandas as pd
from data.generator import generate_synthetic_trips, METRO_ZONES
from data.preprocessor import aggregate_spatio_temporal_demand, engineer_features, prepare_train_test_split
from models.demand_model import HeuristicBaselineModel, MLDemandModel, evaluate_models
from models.recommender import HotspotRecommender, calculate_bearing
from models.evaluator import compute_comprehensive_metrics, compute_zone_level_breakdown


@pytest.fixture(scope="module")
def prepared_data():
    raw = generate_synthetic_trips(days=4, sample_rate=0.15, random_seed=42)
    grid = aggregate_spatio_temporal_demand(raw, freq="1h")
    feat = engineer_features(grid)
    X_train, X_test, y_train, y_test = prepare_train_test_split(feat, test_ratio=0.20)
    return feat, X_train, X_test, y_train, y_test


def test_heuristic_baseline_model(prepared_data):
    """Test that baseline model trains and generates non-negative predictions."""
    feat, X_train, X_test, y_train, y_test = prepared_data
    baseline = HeuristicBaselineModel()
    baseline.fit(feat.loc[X_train.index])

    preds = baseline.predict(feat.loc[X_test.index])
    assert len(preds) == len(X_test)
    assert (preds >= 0).all()
    assert not np.isnan(preds).any()


def test_ml_demand_model_training_and_prediction(prepared_data):
    """Test ML training, non-negative clipped predictions, and feature importance."""
    feat, X_train, X_test, y_train, y_test = prepared_data
    model = MLDemandModel(model_type="gradient_boosting", n_estimators=20, max_depth=3)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert (preds >= 0).all()

    # Feature importances
    fi_df = model.get_feature_importances()
    assert not fi_df.empty
    assert set(fi_df.columns) == {"feature", "importance"}
    assert np.isclose(fi_df["importance"].sum(), 1.0, atol=0.01)


def test_model_serialization(prepared_data):
    """Verify joblib model save and load round-trip consistency."""
    feat, X_train, X_test, y_train, y_test = prepared_data
    model = MLDemandModel(model_type="gradient_boosting", n_estimators=10, max_depth=3)
    model.fit(X_train, y_train)
    orig_preds = model.predict(X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.joblib")
        model.save(save_path)
        assert os.path.exists(save_path)

        loaded_model = MLDemandModel.load(save_path)
        loaded_preds = loaded_model.predict(X_test)
        np.testing.assert_allclose(orig_preds, loaded_preds)


def test_evaluate_models(prepared_data):
    """Verify model comparison metric generation."""
    feat, X_train, X_test, y_train, y_test = prepared_data
    baseline = HeuristicBaselineModel().fit(feat.loc[X_train.index])
    ml = MLDemandModel(model_type="gradient_boosting", n_estimators=15, max_depth=3).fit(X_train, y_train)

    results = evaluate_models(baseline, ml, X_test, y_test, feat)
    assert "Heuristic Baseline" in results
    assert "ML Regressor" in results
    for key in ["MAE", "RMSE", "R2"]:
        assert key in results["Heuristic Baseline"]
        assert key in results["ML Regressor"]
        assert results["ML Regressor"][key] is not None


def test_hotspot_recommender():
    """Verify driver repositioning optimizer ranking and constraints."""
    recommender = HotspotRecommender()

    # Driver in Sunset Suburbs (Zone 8: 37.7530, -122.4860)
    driver_lat, driver_lon = 37.7530, -122.4860
    # Simulate high demand in Downtown (Zone 1) and Airport (Zone 3)
    demands = {z["zone_id"]: 10.0 for z in METRO_ZONES}
    demands[1] = 120.0  # Downtown
    demands[3] = 95.0   # Airport
    surges = {z["zone_id"]: 1.0 for z in METRO_ZONES}
    surges[1] = 1.8

    recs = recommender.recommend(
        driver_lat=driver_lat,
        driver_lon=driver_lon,
        predicted_demands=demands,
        predicted_surges=surges,
        top_k=3,
        max_reposition_km=25.0,
    )

    assert len(recs) == 3
    # Check that rank 1 has highest utility score
    assert recs[0]["rank"] == 1
    assert recs[0]["utility_score"] >= recs[1]["utility_score"] >= recs[2]["utility_score"]

    # Verify calculated fields
    for rec in recs:
        assert rec["distance_km"] >= 0
        assert rec["travel_time_min"] >= 0
        assert rec["deadhead_cost"] >= 0
        assert rec["cardinal_direction"] in [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
        ]
        assert len(rec["rationale"]) > 5


def test_bearing_calculation():
    """Test bearing and compass orientation."""
    # North movement: (37.0, -122.0) -> (38.0, -122.0)
    deg, cardinal = calculate_bearing(37.0, -122.0, 38.0, -122.0)
    assert 350 <= deg or deg <= 10
    assert cardinal == "N"

    # East movement: (37.0, -122.0) -> (37.0, -121.0)
    deg, cardinal = calculate_bearing(37.0, -122.0, 37.0, -121.0)
    assert 80 <= deg <= 100
    assert cardinal == "E"
