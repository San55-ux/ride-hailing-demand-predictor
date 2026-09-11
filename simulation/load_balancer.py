"""
Multi-Driver Fleet Load Balancing and Spatial Coordination.
Prevents herd behavior and hotspot overcrowding by dynamically balancing driver fleet allocation
across demand zones proportional to customer requests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from data.generator import METRO_ZONES, haversine_distance_km


class FleetLoadBalancer:
    """
    Simulates and coordinates fleet-wide positioning for hundreds of ride-hailing drivers.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def generate_initial_fleet(self, fleet_size: int = 150) -> List[Dict[str, Any]]:
        """
        Generates random initial positions for N drivers across the metropolitan area.
        """
        rng = np.random.RandomState(self.random_seed)
        drivers = []
        for i in range(fleet_size):
            assigned_zone = rng.choice(METRO_ZONES)
            # Scatter within zone
            dlat = rng.normal(0, 0.015)
            dlon = rng.normal(0, 0.015)
            drivers.append({
                "driver_id": f"DRV_{i+1:03d}",
                "lat": round(assigned_zone["lat"] + dlat, 5),
                "lon": round(assigned_zone["lon"] + dlon, 5),
                "current_zone_id": assigned_zone["zone_id"],
                "status": "idle",
            })
        return drivers

    def simulate_dispatch(
        self,
        fleet: List[Dict[str, Any]],
        predicted_demands: Dict[int, float],
        strategy: str = "load_balanced",
    ) -> Dict[str, Any]:
        """
        Runs fleet repositioning under 'naive_greedy' vs 'load_balanced' strategy.

        Args:
            fleet: List of driver dicts with current positions.
            predicted_demands: Dict mapping zone_id -> predicted rides/hr.
            strategy: 'naive_greedy' (all chase #1 hotspot) or 'load_balanced' (coordinated allocation).

        Returns:
            Dict containing driver dispatch assignments, zone supply/demand stats, and fleet metrics.
        """
        rng = np.random.RandomState(self.random_seed + 99)
        fleet_size = len(fleet)
        zone_ids = [z["zone_id"] for z in METRO_ZONES]
        total_demand = sum(predicted_demands.values()) or 1.0

        # Target quota of drivers per zone based on demand share
        target_driver_quota = {
            z_id: max(1, int(round((predicted_demands.get(z_id, 0) / total_demand) * fleet_size)))
            for z_id in zone_ids
        }

        # Track allocated drivers per zone
        zone_allocations: Dict[int, int] = {z_id: 0 for z_id in zone_ids}
        dispatches: List[Dict[str, Any]] = []

        if strategy == "naive_greedy":
            # Find the top 2 city-wide demand hotspots
            sorted_zones = sorted(zone_ids, key=lambda z: predicted_demands.get(z, 0), reverse=True)
            top_zone_id = sorted_zones[0]
            second_zone_id = sorted_zones[1] if len(sorted_zones) > 1 else top_zone_id

            for driver in fleet:
                # 85% flock to the absolute #1 hotspot, 15% to #2
                target_id = top_zone_id if rng.rand() < 0.85 else second_zone_id
                target_zone = next(z for z in METRO_ZONES if z["zone_id"] == target_id)
                travel_km = haversine_distance_km(driver["lat"], driver["lon"], target_zone["lat"], target_zone["lon"])
                zone_allocations[target_id] += 1

                dispatches.append({
                    "driver_id": driver["driver_id"],
                    "origin_zone": driver["current_zone_id"],
                    "target_zone": target_id,
                    "target_zone_name": target_zone["name"],
                    "target_lat": target_zone["lat"],
                    "target_lon": target_zone["lon"],
                    "travel_km": round(travel_km, 2),
                })

        else:  # 'load_balanced'
            # Multi-driver entropy-regularized allocation:
            # Assign drivers such that zones receive supply proportional to demand
            # while minimizing excessive deadhead travel distance
            capacity_remaining = target_driver_quota.copy()

            # Ensure sum of quotas matches fleet size
            quota_sum = sum(capacity_remaining.values())
            diff = fleet_size - quota_sum
            if diff != 0:
                highest_dem_zone = max(zone_ids, key=lambda z: predicted_demands.get(z, 0))
                capacity_remaining[highest_dem_zone] = max(1, capacity_remaining[highest_dem_zone] + diff)

            for driver in fleet:
                # Candidate scores: utility balances travel distance and remaining quota
                candidate_scores = []
                for z in METRO_ZONES:
                    z_id = z["zone_id"]
                    dist = haversine_distance_km(driver["lat"], driver["lon"], z["lat"], z["lon"])
                    remaining = capacity_remaining[z_id]
                    # Score combines remaining unfilled demand quota with distance penalty
                    score = (remaining * 2.5) - (dist * 0.8)
                    candidate_scores.append((score, z_id, z, dist))

                # Sort candidates by score descending
                candidate_scores.sort(key=lambda x: x[0], reverse=True)
                chosen = candidate_scores[0]

                # Update capacity and allocations
                chosen_z_id = chosen[1]
                chosen_zone = chosen[2]
                chosen_dist = chosen[3]

                capacity_remaining[chosen_z_id] = max(0, capacity_remaining[chosen_z_id] - 1)
                zone_allocations[chosen_z_id] += 1

                dispatches.append({
                    "driver_id": driver["driver_id"],
                    "origin_zone": driver["current_zone_id"],
                    "target_zone": chosen_z_id,
                    "target_zone_name": chosen_zone["name"],
                    "target_lat": chosen_zone["lat"],
                    "target_lon": chosen_zone["lon"],
                    "travel_km": round(chosen_dist, 2),
                })

        # Calculate fleet-wide equilibrium and performance metrics
        zone_report = []
        overcrowded_zones = 0
        undersupplied_zones = 0
        total_fulfilled_demand = 0.0

        for z in METRO_ZONES:
            z_id = z["zone_id"]
            dem = predicted_demands.get(z_id, 0.0)
            sup = zone_allocations.get(z_id, 0)
            # Supply-to-demand ratio (ideal is ~0.8 to 1.2)
            # Assuming 1 driver can handle approx 2.2 rides/hr
            capacity = sup * 2.2
            coverage_ratio = round(capacity / max(dem, 1.0), 2)
            fulfilled = min(dem, capacity)
            total_fulfilled_demand += fulfilled

            if coverage_ratio > 1.4:
                status = "Severely Overcrowded"
                overcrowded_zones += 1
            elif coverage_ratio < 0.65:
                status = "Critically Undersupplied"
                undersupplied_zones += 1
            else:
                status = "Balanced Equilibrium"

            zone_report.append({
                "zone_id": z_id,
                "zone_name": z["name"],
                "predicted_demand": round(dem, 1),
                "allocated_drivers": sup,
                "demand_capacity": round(capacity, 1),
                "supply_demand_ratio": coverage_ratio,
                "status": status,
            })

        unserved_zones = sum(1 for sup in zone_allocations.values() if sup == 0)
        avg_travel_km = round(float(np.mean([d["travel_km"] for d in dispatches])), 2)
        fulfillment_pct = round((total_fulfilled_demand / max(total_demand, 1.0)) * 100.0, 1)

        # Gini / stddev of driver allocation vs demand to measure imbalance
        ratios = [r["supply_demand_ratio"] for r in zone_report]
        imbalance_score = round(float(np.std(ratios)), 2)

        return {
            "strategy": strategy,
            "fleet_size": fleet_size,
            "dispatches": dispatches,
            "zone_report": pd.DataFrame(zone_report),
            "avg_travel_km": avg_travel_km,
            "fulfillment_pct": min(100.0, fulfillment_pct),
            "overcrowded_zones": overcrowded_zones,
            "undersupplied_zones": undersupplied_zones,
            "unserved_zones": unserved_zones,
            "imbalance_index": imbalance_score,
        }
