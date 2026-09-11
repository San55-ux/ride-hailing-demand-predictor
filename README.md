# 🚕 Ride-Hailing Demand Heatmap Predictor & Driver Repositioning Navigator

An end-to-end intelligent spatio-temporal forecasting and driver repositioning recommendation system for ride-hailing and taxi drivers.

Live Link : https://ride-hailing-demand-predictor.onrender.com/
---

## 📌 Problem Statement
Ride-hailing drivers spend up to 40% of their shift driving empty (deadheading) or idling in low-demand areas while adjacent neighborhoods experience unfulfilled demand and surge pricing. 

This project solves that problem by:
1. Predicting demand hotspots across urban zones and future time windows using historical & simulated trip request data.
2. Rendering interactive 2D/3D map-based heatmaps and hotspot clusters.
3. Recommending where a driver should reposition next by calculating net utility (expected surge earnings minus deadhead travel costs and wait times).
4. **Stretch Goal 1**: Simulating realistic multi-hour shifts to compare AI-guided positioning vs. random cruising and static hub waiting.
5. **Stretch Goal 2**: Coordinating multi-driver fleet load balancing to prevent herd behavior and hotspot oversupply.

---

## 🏗️ Architecture & Modules

```
ride_hailing_demand_predictor/
├── data/
│   ├── generator.py            # TLC-style spatio-temporal ride request synthesizer & Poisson generator
│   ├── preprocessor.py         # Complete spatial-temporal grid builder & cyclical feature engineering
│   └── __init__.py
├── models/
│   ├── demand_model.py         # Supervised ML (Gradient Boosting) + Heuristic Baseline Benchmark
│   ├── recommender.py          # Hotspot repositioning optimizer & bearing calculator
│   ├── evaluator.py            # Error metrics (MAE, RMSE, R²), residual diagnostics, zone breakdowns
│   └── __init__.py
├── simulation/
│   ├── earnings_simulator.py   # Stochastic shift engine comparing AI vs Random vs Static strategies
│   ├── load_balancer.py        # Fleet-wide multi-driver coordination and equilibrium dispatcher
│   └── __init__.py
├── ui/
│   ├── map_visualizer.py       # PyDeck interactive heatmaps, 3D columns, driver vectors
│   ├── components.py           # Plotly charts, strategy comparisons, supply-demand balances
│   └── __init__.py
├── tests/
│   ├── test_data_handling.py   # Unit tests for data generation, spatial grid, feature engineering
│   ├── test_ml_heuristics.py   # Unit tests for ML models, baseline, and recommender ranking
│   ├── test_simulation.py      # Unit tests for shift simulator and fleet load balancer
│   └── test_ui_presentation.py # Unit tests for map layers, color gradients, and UI payloads
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Run Automated Test Suite
Ensure all components pass tests:
```powershell
cd C:\Users\sanja\.gemini\antigravity\scratch\ride_hailing_demand_predictor
pytest -v tests/
```

### 2. Launch Streamlit Interactive Application
Run the web application dashboard:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌟 Key Features

### 1. Live Heatmap & Driver Navigator
- Continuous Gaussian density heatmap layer powered by PyDeck.
- 3D zone elevation columns proportional to predicted trip request intensity.
- Real-time repositioning arrow pointing from the driver's current coordinates to the highest-utility hotspot.
- Top-3 recommendation cards with distance, travel time, compass heading, deadhead expense, and rationale.

### 2. Predictive ML Hub & Heuristics
- Dual-model comparison: Supervised Gradient Boosted Regressor vs. Historical Time-of-Week Heuristic.
- Feature importance visualization (lagged demand, hour sin/cos, weather, zone type).
- Zone-level accuracy and residual breakdown.

### 3. Driver Shift & Earnings Simulator (Stretch Goal 1)
- Simulates realistic 4-to-12 hour shifts with stochastic passenger requests.
- Compares:
  - **AI Demand-Aware Positioning**: Proactively repositions to high-surge, high-demand zones.
  - **Random Roaming**: Continuous cruising without demand guidance.
  - **Static Waiting**: Parks at a single hub (e.g. Downtown).
- Displays gross revenue, surge bonuses, fuel expense, net profit, hourly rate, and full telemetry logs.

### 4. Fleet Multi-Driver Load Balancing (Stretch Goal 2)
- Simulates fleets from 50 to 400 drivers.
- Resolves the "herd behavior" dilemma where drivers flock to the same hotspot and cannibalize surge.
- Allocates drivers proportionally to zone demand while penalizing deadhead distance.
- Displays before-and-after supply/demand equilibrium charts.
