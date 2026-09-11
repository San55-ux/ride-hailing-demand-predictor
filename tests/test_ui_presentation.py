"""
Unit Tests for Map Presentation Payloads, PyDeck Deck Construction, and Plotly Components.
"""

import pytest
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from data.generator import METRO_ZONES
from ui.map_visualizer import get_demand_color, build_demand_heatmap_deck, build_fleet_dispatch_deck
from ui.components import (
    create_strategy_comparison_chart,
    create_hourly_demand_curve,
    create_feature_importance_chart,
    create_load_balance_comparison_chart,
)


def test_get_demand_color():
    """Verify color generator returns valid 4-element RGBA lists with 0-255 bounds."""
    for dem in [0.0, 25.0, 75.0, 150.0, 300.0]:
        rgba = get_demand_color(dem, max_demand=200.0)
        assert len(rgba) == 4
        assert all(0 <= c <= 255 for c in rgba)
        # Alpha should be non-zero
        assert rgba[3] > 0


def test_build_demand_heatmap_deck():
    """Verify PyDeck map object generation with layers, view state, and driver vector."""
    zone_df = pd.DataFrame([
        {
            "zone_id": z["zone_id"],
            "zone_name": z["name"],
            "lat": z["lat"],
            "lon": z["lon"],
            "type": z["type"],
            "predicted_demand": 50.0 + i * 5,
            "predicted_surge": 1.2,
        }
        for i, z in enumerate(METRO_ZONES)
    ])

    top_rec = {
        "zone_id": 1,
        "zone_name": "Downtown Financial",
        "zone_lat": 37.7891,
        "zone_lon": -122.4014,
        "distance_km": 4.5,
    }

    deck = build_demand_heatmap_deck(
        zone_demands_df=zone_df,
        driver_lat=37.7530,
        driver_lon=-122.4860,
        top_recommendation=top_rec,
        view_elevation_3d=True,
    )

    assert isinstance(deck, pdk.Deck)
    assert len(deck.layers) >= 3  # Heatmap, Column/Scatter, Driver, Line
    layer_types = [l.type for l in deck.layers]
    assert "HeatmapLayer" in layer_types
    assert "ColumnLayer" in layer_types
    assert "ScatterplotLayer" in layer_types
    assert "LineLayer" in layer_types


def test_build_fleet_dispatch_deck():
    """Verify fleet vector deck layer generation."""
    sample_dispatches = [
        {
            "driver_id": "DRV_001",
            "origin_zone": 8,
            "target_zone": 1,
            "target_zone_name": "Downtown Financial",
            "target_lat": 37.7891,
            "target_lon": -122.4014,
            "travel_km": 4.2,
        },
        {
            "driver_id": "DRV_002",
            "origin_zone": 7,
            "target_zone": 3,
            "target_zone_name": "Metro Int'l Airport",
            "target_lat": 37.6213,
            "target_lon": -122.3790,
            "travel_km": 18.0,
        },
    ]

    deck = build_fleet_dispatch_deck(sample_dispatches, zone_demands={1: 100.0, 3: 80.0})
    assert isinstance(deck, pdk.Deck)
    assert len(deck.layers) == 2
    layer_types = [l.type for l in deck.layers]
    assert "ScatterplotLayer" in layer_types
    assert "LineLayer" in layer_types


def test_ui_plotly_charts():
    """Verify all Plotly figure builders create valid figures with traces."""
    # 1. Strategy Comparison Chart
    comp_df = pd.DataFrame([
        {"Strategy": "AI Demand-Aware", "Net Earnings ($)": 180.0, "Surge Bonus ($)": 35.0, "Fuel Expense ($)": 12.0},
        {"Strategy": "Random Roaming", "Net Earnings ($)": 120.0, "Surge Bonus ($)": 10.0, "Fuel Expense ($)": 28.0},
    ])
    fig1 = create_strategy_comparison_chart(comp_df)
    assert isinstance(fig1, go.Figure)
    assert len(fig1.data) == 3

    # 2. Hourly Demand Curve
    hourly_df = pd.DataFrame([
        {"hour": h, "zone_name": "Downtown Financial", "demand_count": 50 + h}
        for h in range(24)
    ])
    fig2 = create_hourly_demand_curve(hourly_df, ["Downtown Financial"])
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) == 1

    # 3. Feature Importance Chart
    fi_df = pd.DataFrame({
        "feature": ["lag_1", "hour", "is_rush_hour"],
        "importance": [0.45, 0.35, 0.20],
    })
    fig3 = create_feature_importance_chart(fi_df, top_n=3)
    assert isinstance(fig3, go.Figure)
    assert len(fig3.data) == 1

    # 4. Load Balance Chart
    report_greedy = pd.DataFrame([
        {"zone_name": "Downtown", "supply_demand_ratio": 3.2},
        {"zone_name": "Airport", "supply_demand_ratio": 0.3},
    ])
    report_balanced = pd.DataFrame([
        {"zone_name": "Downtown", "supply_demand_ratio": 1.1},
        {"zone_name": "Airport", "supply_demand_ratio": 0.95},
    ])
    fig4 = create_load_balance_comparison_chart(report_greedy, report_balanced)
    assert isinstance(fig4, go.Figure)
    assert len(fig4.data) == 2
