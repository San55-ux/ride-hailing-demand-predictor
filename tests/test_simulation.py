"""
Unit Tests for Driver Shift Simulator and Multi-Driver Fleet Load Balancer.
"""

import pytest
import numpy as np
import pandas as pd
from data.generator import METRO_ZONES
from simulation.earnings_simulator import ShiftSimulationEngine
from simulation.load_balancer import FleetLoadBalancer


@pytest.fixture
def hourly_maps():
    dem_map = {h: {z["zone_id"]: float(z["base_volume"] * 0.5) for z in METRO_ZONES} for h in range(24)}
    srg_map = {h: {z["zone_id"]: 1.2 for z in METRO_ZONES} for h in range(24)}
    return dem_map, srg_map


def test_shift_simulation_engine_single_run(hourly_maps):
    """Test that an individual shift run adheres to financial and temporal conservation laws."""
    dem_map, srg_map = hourly_maps
    engine = ShiftSimulationEngine(shift_hours=6.0, fuel_cost_per_km=0.40, random_seed=42)

    for strategy in ["ai_demand_aware", "random_roaming", "static_hub"]:
        res = engine.run_shift(strategy, dem_map, srg_map, start_hour=8)

        assert res["strategy"] == strategy
        assert res["trips_completed"] >= 0
        assert res["gross_fare"] >= 0
        assert res["operating_expense"] >= 0
        # Financial equation: net = total_gross - expense
        assert np.isclose(res["net_earnings"], res["total_gross"] - res["operating_expense"], atol=0.05)
        # Net hourly rate
        assert np.isclose(res["net_hourly_rate"], res["net_earnings"] / 6.0, atol=0.05)
        # Time budget: shift duration is 6 hours (360 min)
        total_accounted_time = res["time_on_trip_min"] + res["time_repositioning_min"] + res["time_idle_min"]
        assert total_accounted_time <= 360.5
        assert 0.0 <= res["utilization_pct"] <= 100.0


def test_strategy_comparison(hourly_maps):
    """Test multi-trial strategy comparison outputs formatted summary."""
    dem_map, srg_map = hourly_maps
    engine = ShiftSimulationEngine(shift_hours=4.0, random_seed=42)
    comp_df = engine.compare_strategies(dem_map, srg_map, start_hour=10, n_trials=2)

    assert not comp_df.empty
    assert len(comp_df) == 3
    assert "Strategy" in comp_df.columns
    assert "Net Earnings ($)" in comp_df.columns
    assert "Net Hourly Rate ($/hr)" in comp_df.columns
    assert "Trips Completed" in comp_df.columns


def test_fleet_load_balancer():
    """Verify fleet generation and coordinated vs naive dispatch."""
    balancer = FleetLoadBalancer(random_seed=42)
    fleet = balancer.generate_initial_fleet(fleet_size=100)

    assert len(fleet) == 100
    for drv in fleet:
        assert drv["driver_id"].startswith("DRV_")
        assert 37.0 <= drv["lat"] <= 38.5
        assert -123.0 <= drv["lon"] <= -121.5

    demands = {z["zone_id"]: float(z["base_volume"]) for z in METRO_ZONES}

    # Run greedy vs load balanced
    greedy_res = balancer.simulate_dispatch(fleet, demands, strategy="naive_greedy")
    balanced_res = balancer.simulate_dispatch(fleet, demands, strategy="load_balanced")

    assert greedy_res["fleet_size"] == 100
    assert balanced_res["fleet_size"] == 100
    assert len(greedy_res["dispatches"]) == 100
    assert len(balanced_res["dispatches"]) == 100

    # Coordinated load balancing must achieve better demand fulfillment
    assert balanced_res["fulfillment_pct"] >= greedy_res["fulfillment_pct"]
    # Greedy dispatch leaves 8+ zones completely unserved (0 drivers)
    assert greedy_res["unserved_zones"] > balanced_res["unserved_zones"]
    assert balanced_res["unserved_zones"] == 0
    # Coordinated allocation has significantly lower spatial imbalance
    assert balanced_res["imbalance_index"] < greedy_res["imbalance_index"]
