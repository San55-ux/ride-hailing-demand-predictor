"""
Ride-Hailing Demand Heatmap Predictor & Driver Repositioning Navigator.
Main Streamlit Application.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import project modules
from data.generator import METRO_ZONES, ZONES_DF, generate_synthetic_trips, haversine_distance_km
from data.preprocessor import aggregate_spatio_temporal_demand, engineer_features, prepare_train_test_split
from models.demand_model import HeuristicBaselineModel, MLDemandModel, evaluate_models
from models.recommender import HotspotRecommender
from models.evaluator import compute_comprehensive_metrics, compute_zone_level_breakdown
from simulation.earnings_simulator import ShiftSimulationEngine
from simulation.load_balancer import FleetLoadBalancer
from ui.map_visualizer import build_demand_heatmap_deck, build_fleet_dispatch_deck
from ui.components import (
    create_strategy_comparison_chart,
    create_hourly_demand_curve,
    create_feature_importance_chart,
    create_load_balance_comparison_chart,
)

st.set_page_config(
    page_title="Ride-Hailing Demand Heatmap Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFB319 0%, #00C897 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #A0AEC0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #718096;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .rec-card {
        background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
        border-left: 5px solid #FFB319;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .badge-surge {
        background-color: #FF5252;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-demand {
        background-color: #00C897;
        color: #0F172A;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data & Model Pipeline
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading trip requests dataset...")
def load_and_preprocess_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    sample_csv = os.path.join(base_path, "data", "trips_sample.csv")
    if os.path.exists(sample_csv):
        raw_trips = pd.read_csv(sample_csv, parse_dates=["timestamp"])
    else:
        raw_trips = generate_synthetic_trips(days=7, sample_rate=0.20, random_seed=42)
        raw_trips.to_csv(sample_csv, index=False)

    demand_grid = aggregate_spatio_temporal_demand(raw_trips, freq="1h")
    featured_df = engineer_features(demand_grid)
    return raw_trips, demand_grid, featured_df


@st.cache_resource(show_spinner="Loading ML Demand Forecasting Engine...")
def train_or_load_models(_featured_df):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "models", "demand_model.joblib")

    X_train, X_test, y_train, y_test = prepare_train_test_split(_featured_df, test_ratio=0.20)
    baseline = HeuristicBaselineModel().fit(_featured_df.loc[X_train.index])

    ml_model = None
    if os.path.exists(model_path):
        try:
            ml_model = MLDemandModel.load(model_path)
        except Exception:
            ml_model = None

    if ml_model is None:
        ml_model = MLDemandModel(model_type="gradient_boosting", n_estimators=60, max_depth=4).fit(X_train, y_train)
        try:
            ml_model.save(model_path)
        except Exception:
            pass

    eval_results = evaluate_models(baseline, ml_model, X_test, y_test, _featured_df)
    return baseline, ml_model, X_train, X_test, y_train, y_test, eval_results


# Load Data & Models
with st.spinner("Initializing Ride-Hailing Demand Intelligence Engine..."):
    raw_trips, demand_grid, featured_df = load_and_preprocess_data()
    baseline_model, ml_model, X_train, X_test, y_train, y_test, eval_results = train_or_load_models(featured_df)

recommender = HotspotRecommender()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Dispatch Controls")

# Day & Hour Selector
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day_idx = st.sidebar.selectbox("📅 Day of Week", range(7), format_func=lambda i: day_names[i], index=4)  # Friday default
selected_hour = st.sidebar.slider("⏰ Target Time Window (Hour)", 0, 23, 18, format="%02d:00")

# Quick Time Presets
st.sidebar.markdown("**Quick Time Presets:**")
cols_preset = st.sidebar.columns(3)
if cols_preset[0].button("Morning 8a"):
    selected_hour = 8
if cols_preset[1].button("Rush 5p"):
    selected_hour = 17
if cols_preset[2].button("Night 11p"):
    selected_hour = 23

weather = st.sidebar.selectbox("🌦️ Weather Conditions", ["Clear", "Cloudy", "Rain", "Fog"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Driver Location")

zone_options = {z["zone_id"]: f"{z['name']} (Zone {z['zone_id']})" for z in METRO_ZONES}
selected_zone_id = st.sidebar.selectbox(
    "Current Position",
    options=list(zone_options.keys()),
    format_func=lambda z_id: zone_options[z_id],
    index=7,  # Default to Sunset Residential Suburb
)
curr_zone = next(z for z in METRO_ZONES if z["zone_id"] == selected_zone_id)
driver_lat = curr_zone["lat"]
driver_lon = curr_zone["lon"]

st.sidebar.markdown("---")
view_3d = st.sidebar.checkbox("Show 3D Elevation Columns", value=True)
reposition_radius = st.sidebar.slider("Max Search Radius (km)", 3.0, 25.0, 15.0, 1.0)


# ---------------------------------------------------------
# Demand Inference for Target Time Window
# ---------------------------------------------------------
# Synthesize feature inputs for all zones at the chosen (day_of_week, hour, weather)
current_time_sample = datetime(2026, 9, 1 + selected_day_idx, selected_hour, 0)
is_weekend = int(selected_day_idx >= 5)

zone_preds = []
pred_demands_dict = {}
pred_surges_dict = {}

for z in METRO_ZONES:
    z_id = z["zone_id"]
    # Look up recent historical reference for lag features
    hist_subset = featured_df[
        (featured_df["pickup_zone_id"] == z_id) &
        (featured_df["hour"] == selected_hour) &
        (featured_df["day_of_week"] == selected_day_idx)
    ]
    avg_hist_demand = float(hist_subset["demand_count"].mean()) if not hist_subset.empty else 20.0
    avg_hist_surge = float(hist_subset["avg_surge"].mean()) if not hist_subset.empty else 1.0

    # Build feature record matching model features
    row_feat = {
        "pickup_zone_id": z_id,
        "lat": z["lat"],
        "lon": z["lon"],
        "hour": selected_hour,
        "day_of_week": selected_day_idx,
        "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * selected_hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * selected_hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * selected_day_idx / 7.0),
        "dow_cos": np.cos(2 * np.pi * selected_day_idx / 7.0),
        "is_morning_rush": int(7 <= selected_hour <= 9 and not is_weekend),
        "is_evening_rush": int(16 <= selected_hour <= 19 and not is_weekend),
        "is_nightlife": int((selected_hour >= 21 or selected_hour <= 2) and is_weekend),
        "weather_rain": int(weather == "Rain"),
        "weather_fog": int(weather == "Fog"),
        "weather_cloudy": int(weather == "Cloudy"),
        "lag_1": avg_hist_demand * 0.95,
        "lag_2": avg_hist_demand * 0.90,
        "lag_24": avg_hist_demand,
        "rolling_mean_3": avg_hist_demand * 0.97,
    }

    # Add zone type one-hot
    for zt in ["commercial", "transit", "entertainment", "nightlife", "education", "residential", "residential_upscale", "events"]:
        row_feat[f"zone_type_{zt}"] = int(z["type"] == zt)

    feat_df = pd.DataFrame([row_feat])
    # Keep only columns known to model
    feat_df_model = feat_df[[c for c in ml_model.feature_names if c in feat_df.columns]]
    for missing_col in set(ml_model.feature_names) - set(feat_df_model.columns):
        feat_df_model[missing_col] = 0
    feat_df_model = feat_df_model[ml_model.feature_names]

    ml_predicted_demand = float(ml_model.predict(feat_df_model)[0])

    # Dynamic weather & demand surge adjustment
    surge = avg_hist_surge
    if weather == "Rain":
        ml_predicted_demand *= 1.35
        surge = max(surge, 1.4)
    elif weather == "Fog":
        ml_predicted_demand *= 1.15

    pred_demands_dict[z_id] = round(ml_predicted_demand, 1)
    pred_surges_dict[z_id] = round(surge, 2)

    zone_preds.append({
        "zone_id": z_id,
        "zone_name": z["name"],
        "lat": z["lat"],
        "lon": z["lon"],
        "type": z["type"],
        "predicted_demand": round(ml_predicted_demand, 1),
        "predicted_surge": round(surge, 2),
    })

zone_preds_df = pd.DataFrame(zone_preds)

# Run Hotspot Recommender for Driver
recommendations = recommender.recommend(
    driver_lat=driver_lat,
    driver_lon=driver_lon,
    predicted_demands=pred_demands_dict,
    predicted_surges=pred_surges_dict,
    top_k=3,
    max_reposition_km=reposition_radius,
)
top_rec = recommendations[0] if recommendations else None


# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🚕 Ride-Hailing Demand Heatmap Predictor</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Real-Time Spatio-Temporal Demand Forecasting & AI Driver Repositioning Navigator • '
    f'Target: <b>{day_names[selected_day_idx]} {selected_hour:02d}:00</b> ({weather} Weather)</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_map, tab_models, tab_sim, tab_fleet = st.tabs([
    "🗺️ Demand Heatmap & Navigator",
    "🧠 Predictive ML Hub",
    "💰 Driver Earnings Simulator (Stretch 1)",
    "⚖️ Fleet Load Balancing (Stretch 2)",
])

# =========================================================
# TAB 1: Live Demand Heatmap & Hotspot Navigator
# =========================================================
with tab_map:
    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        top_zone_name = top_rec['zone_name'] if top_rec else 'None'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎯 Optimal Hotspot</div>
            <div class="metric-value">{top_zone_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        top_dem = f"{top_rec['predicted_demand']:.0f} rides/hr" if top_rec else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔥 Predicted Demand</div>
            <div class="metric-value">{top_dem}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        top_srg = f"{top_rec['predicted_surge']:.2f}x" if top_rec else "1.00x"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ Surge Rate</div>
            <div class="metric-value">{top_srg}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        top_earn = f"${top_rec['expected_gross_hourly']:.0f}/hr" if top_rec else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💵 Potential Gross</div>
            <div class="metric-value">{top_earn}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Layout: Map on Left (7 cols), Recommendations on Right (5 cols)
    col_map, col_recs = st.columns([7, 5])

    with col_map:
        st.markdown("#### 📍 Real-Time Demand Density & Repositioning Vector")
        st.caption("🔵 Cyan dot: Your position • 🟡 Golden line: Optimal navigation path • 3D Columns: Demand volume")
        
        # Render PyDeck Deck
        deck = build_demand_heatmap_deck(
            zone_demands_df=zone_preds_df,
            driver_lat=driver_lat,
            driver_lon=driver_lon,
            top_recommendation=top_rec,
            view_elevation_3d=view_3d,
        )
        st.pydeck_chart(deck, use_container_width=True)

    with col_recs:
        st.markdown("#### 🧭 Recommended Actions for Driver")
        if recommendations:
            for rec in recommendations:
                rank = rec["rank"]
                medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: white;">{medal} {rec['zone_name']}</h4>
                        <div>
                            <span class="badge-demand">{rec['predicted_demand']:.0f} rides/hr</span>
                            <span class="badge-surge">{rec['predicted_surge']:.1f}x</span>
                        </div>
                    </div>
                    <p style="margin: 8px 0 4px 0; font-size: 0.92rem; color: #E2E8F0;">
                        <b>{rec['rationale']}</b>
                    </p>
                    <div style="display: flex; gap: 14px; font-size: 0.85rem; color: #CBD5E0; margin-top: 6px;">
                        <span>🚗 <b>{rec['distance_km']:.1f} km</b> ({rec['travel_time_min']:.0f} min)</span>
                        <span>🧭 Heading: <b>{rec['cardinal_direction']}</b> ({rec['bearing_deg']:.0f}°)</span>
                        <span>⏱️ Est. Wait: <b>{rec['expected_wait_min']:.1f} min</b></span>
                    </div>
                    <div style="font-size: 0.85rem; color: #A0AEC0; margin-top: 4px;">
                        ⛽ Deadhead Expense: <b>${rec['deadhead_cost']:.2f}</b> • Net Utility Index: <b>+{rec['utility_score']:.1f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No zones within selected search radius. Try increasing the search radius in sidebar.")

    # All Zones Table
    with st.expander("📋 View All City Zones & Predicted Demand Grid"):
        st.dataframe(
            zone_preds_df[["zone_id", "zone_name", "type", "predicted_demand", "predicted_surge", "lat", "lon"]],
            use_container_width=True,
        )


# =========================================================
# TAB 2: Machine Learning Hub & Model Evaluation
# =========================================================
with tab_models:
    st.markdown("### 🧠 Predictive Model Architecture & Validation")
    st.markdown("""
    The demand predictor uses **Supervised Gradient Boosted Decision Trees** trained on spatio-temporal features,
    benchmarked directly against a historical time-of-week moving-average **Heuristic Baseline**.
    """)

    col_m1, col_m2 = st.columns([6, 6])

    with col_m1:
        st.markdown("#### 📊 Model Performance Benchmarks (Test Set)")
        eval_df = pd.DataFrame(eval_results).T
        eval_df["MAE"] = eval_df["MAE"].apply(lambda v: f"{v:.3f} rides/hr")
        eval_df["RMSE"] = eval_df["RMSE"].apply(lambda v: f"{v:.3f} rides/hr")
        eval_df["R2 Score"] = eval_df["R2"].apply(lambda v: f"{v:.3f}")
        st.table(eval_df[["MAE", "RMSE", "R2 Score"]])

        mae_improvement = (
            (eval_results["Heuristic Baseline"]["MAE"] - eval_results["ML Regressor"]["MAE"])
            / eval_results["Heuristic Baseline"]["MAE"]
        ) * 100.0
        st.success(f"✨ **ML Model achieves {mae_improvement:.1f}% lower MAE** compared to heuristic baseline.")

        # Zone level errors
        st.markdown("#### 🎯 Per-Zone Accuracy Breakdown")
        test_featured = featured_df.loc[X_test.index].copy()
        test_featured["ml_pred"] = ml_model.predict(X_test)
        zone_acc_df = compute_zone_level_breakdown(test_featured, "demand_count", "ml_pred")
        st.dataframe(zone_acc_df, use_container_width=True)

    with col_m2:
        st.markdown("#### 🔍 Feature Importances")
        fi_df = ml_model.get_feature_importances()
        fi_fig = create_feature_importance_chart(fi_df, top_n=10)
        st.plotly_chart(fi_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 24-Hour Spatio-Temporal Demand Waves")
    hourly_agg = featured_df.groupby(["hour", "zone_name"])["demand_count"].mean().reset_index()
    sample_zones = ["Downtown Financial", "Midtown Tech Hub", "Mission Nightlife & Arts", "Metro Int'l Airport", "Sunset Residential District"]
    demand_fig = create_hourly_demand_curve(hourly_agg, sample_zones)
    st.plotly_chart(demand_fig, use_container_width=True)


# =========================================================
# TAB 3: Driver Earnings Simulation (Stretch Goal 1)
# =========================================================
with tab_sim:
    st.markdown("### 💰 Driver Shift & Earnings Simulation (Stretch Goal)")
    st.markdown("""
    Evaluate how much additional income a driver makes by following the **Demand-Aware Positioning Engine**
    compared to traditional **Random Cruising** and **Static Hub Waiting** over a full shift.
    """)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sim_shift_hours = st.slider("Shift Duration (Hours)", 4.0, 12.0, 8.0, 1.0)
    with col_s2:
        sim_start_hour = st.slider("Shift Start Time", 0, 23, 14, format="%02d:00")
    with col_s3:
        sim_fuel_cost = st.slider("Vehicle Operating Cost ($/km)", 0.20, 0.80, 0.40, 0.05)

    if st.button("🚀 Run Multi-Strategy Shift Simulation", type="primary"):
        with st.spinner("Simulating realistic 8-hour shifts with stochastic trip dispatches..."):
            engine = ShiftSimulationEngine(
                shift_hours=sim_shift_hours,
                fuel_cost_per_km=sim_fuel_cost,
                random_seed=42,
            )

            # Build hourly demand profiles for simulation
            hourly_dem_map = {}
            hourly_srg_map = {}
            for h in range(24):
                h_df = featured_df[featured_df["hour"] == h]
                hourly_dem_map[h] = h_df.groupby("pickup_zone_id")["demand_count"].mean().to_dict()
                hourly_srg_map[h] = h_df.groupby("pickup_zone_id")["avg_surge"].mean().to_dict()

            comparison_df = engine.compare_strategies(
                predicted_demands_by_hour=hourly_dem_map,
                predicted_surges_by_hour=hourly_srg_map,
                start_hour=sim_start_hour,
                n_trials=4,
            )

            # Store in session state for display
            st.session_state["comparison_df"] = comparison_df

            # Run a single detailed shift for AI strategy to get telemetry
            ai_detail = engine.run_shift("ai_demand_aware", hourly_dem_map, hourly_srg_map, start_hour=sim_start_hour)
            st.session_state["ai_detail"] = ai_detail

    if "comparison_df" in st.session_state:
        comp_df = st.session_state["comparison_df"]

        st.markdown("#### 🏆 Strategy Benchmark Results")
        st.dataframe(comp_df, use_container_width=True)

        chart_fig = create_strategy_comparison_chart(comp_df)
        st.plotly_chart(chart_fig, use_container_width=True)

        # Highlight earnings delta
        ai_net = comp_df.loc[comp_df["Strategy"].str.contains("AI"), "Net Earnings ($)"].values[0]
        random_net = comp_df.loc[comp_df["Strategy"].str.contains("Random"), "Net Earnings ($)"].values[0]
        earnings_boost = ai_net - random_net
        pct_boost = (earnings_boost / random_net) * 100.0

        st.success(
            f"💡 **AI Demand-Aware Positioning yields +${earnings_boost:.2f} extra profit (+{pct_boost:.1f}%)** "
            f"over random roaming by eliminating deadhead idle time and targeting surge bonuses!"
        )

        if "ai_detail" in st.session_state:
            with st.expander("📜 View Simulated AI Driver Shift Telemetry Log"):
                trip_df = pd.DataFrame(st.session_state["ai_detail"]["trip_log"])
                st.dataframe(trip_df, use_container_width=True)


# =========================================================
# TAB 4: Multi-Driver Fleet Load Balancing (Stretch Goal 2)
# =========================================================
with tab_fleet:
    st.markdown("### ⚖️ Multi-Driver Fleet Load Balancing & Coordination (Stretch Goal)")
    st.markdown("""
    When hundreds of drivers are on the road, naive greedy repositioning causes **hotspot overcrowding**
    (e.g., 200 drivers all racing to Downtown, causing surge collapse and massive wait times).
    
    Our **Fleet Coordinator** solves this with supply-demand equilibrium matching.
    """)

    col_f1, col_f2 = st.columns([4, 8])

    with col_f1:
        fleet_size = st.slider("Active Fleet Size (Drivers)", 50, 400, 150, 25)
        run_lb_btn = st.button("🔄 Simulate Fleet Rebalancing", type="primary")

    balancer = FleetLoadBalancer(random_seed=42)
    initial_fleet = balancer.generate_initial_fleet(fleet_size=fleet_size)

    # Run dispatch comparison
    greedy_res = balancer.simulate_dispatch(initial_fleet, pred_demands_dict, strategy="naive_greedy")
    balanced_res = balancer.simulate_dispatch(initial_fleet, pred_demands_dict, strategy="load_balanced")

    # Metrics comparison row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Fulfillment Rate",
            f"{balanced_res['fulfillment_pct']:.1f}%",
            delta=f"+{balanced_res['fulfillment_pct'] - greedy_res['fulfillment_pct']:.1f}%",
        )
    with m2:
        st.metric(
            "Overcrowded Zones",
            f"{balanced_res['overcrowded_zones']}",
            delta=f"{balanced_res['overcrowded_zones'] - greedy_res['overcrowded_zones']} zones",
            delta_color="inverse",
        )
    with m3:
        st.metric(
            "Unserved Zones (0 Drivers)",
            f"{balanced_res['unserved_zones']}",
            delta=f"{balanced_res['unserved_zones'] - greedy_res['unserved_zones']} zones",
            delta_color="inverse",
        )
    with m4:
        st.metric(
            "Imbalance Index (StdDev)",
            f"{balanced_res['imbalance_index']:.2f}",
            delta=f"-{greedy_res['imbalance_index'] - balanced_res['imbalance_index']:.2f}",
            delta_color="inverse",
        )

    st.markdown("---")
    st.markdown("#### 🗺️ Fleet Dispatch Distribution Map (Coordinated Load Balancing)")
    st.caption("Vectors represent coordinated driver assignments preventing overcrowding.")

    fleet_deck = build_fleet_dispatch_deck(balanced_res["dispatches"], pred_demands_dict)
    st.pydeck_chart(fleet_deck, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚖️ Supply-to-Demand Ratio Comparison")
    lb_fig = create_load_balance_comparison_chart(greedy_res["zone_report"], balanced_res["zone_report"])
    st.plotly_chart(lb_fig, use_container_width=True)

    with st.expander("📊 Inspect Detailed Zone Supply/Demand Allocation Tables"):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Naive Greedy Allocation**")
            st.dataframe(greedy_res["zone_report"], use_container_width=True)
        with col_t2:
            st.markdown("**Coordinated Balanced Allocation**")
            st.dataframe(balanced_res["zone_report"], use_container_width=True)
