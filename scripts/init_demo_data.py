"""
Initialization script to generate demo datasets and pre-trained model weights.
Enables instant Streamlit startup.
"""

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import joblib
from data.generator import generate_synthetic_trips
from data.preprocessor import aggregate_spatio_temporal_demand, engineer_features, prepare_train_test_split
from models.demand_model import HeuristicBaselineModel, MLDemandModel, evaluate_models


def initialize():
    print("Generating demo dataset (7 days)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "trips_sample.csv")
    raw_trips = generate_synthetic_trips(days=7, sample_rate=0.20, random_seed=42)
    raw_trips.to_csv(csv_path, index=False)
    print(f"Saved {len(raw_trips)} trips to {csv_path}")

    print("Preprocessing grid and features...")
    demand_grid = aggregate_spatio_temporal_demand(raw_trips, freq="1h")
    featured_df = engineer_features(demand_grid)

    print("Training ML model...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(featured_df, test_ratio=0.20)
    ml_model = MLDemandModel(model_type="gradient_boosting", n_estimators=60, max_depth=4)
    ml_model.fit(X_train, y_train)

    model_path = os.path.join(models_dir, "demand_model.joblib")
    ml_model.save(model_path)
    print(f"Saved pre-trained model to {model_path}")
    print("Initialization complete!")


if __name__ == "__main__":
    initialize()
