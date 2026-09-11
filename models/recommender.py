"""
Driver Hotspot Repositioning Optimizer and Recommendation Engine.
Calculates optimal next positioning for a driver considering predicted demand,
surge multipliers, deadhead travel penalties, and local supply competition.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from data.generator import METRO_ZONES, haversine_distance_km


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, str]:
    """Calculates bearing angle (degrees) and cardinal direction from origin to destination."""
    dlon = np.radians(lon2 - lon1)
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)

    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - (np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon))
    initial_bearing = np.degrees(np.arctan2(x, y))
    compass_bearing = (initial_bearing + 360) % 360

    cardinals = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    cardinal_idx = int(np.round(compass_bearing / 22.5)) % 16
    return round(float(compass_bearing), 1), cardinals[cardinal_idx]


class HotspotRecommender:
    """
    Evaluates candidate zones and recommends the top positioning hotspots for a driver.
    """

    def __init__(
        self,
        cost_per_km: float = 0.45,
        driver_speed_kmh: float = 24.0,
        opportunity_cost_per_hour: float = 25.0,
    ):
        self.cost_per_km = cost_per_km
        self.driver_speed_kmh = driver_speed_kmh
        self.opportunity_cost_per_hour = opportunity_cost_per_hour

    def recommend(
        self,
        driver_lat: float,
        driver_lon: float,
        predicted_demands: Dict[int, float],
        predicted_surges: Optional[Dict[int, float]] = None,
        zone_supplies: Optional[Dict[int, int]] = None,
        top_k: int = 3,
        max_reposition_km: float = 12.0,
    ) -> List[Dict[str, Any]]:
        """
        Calculates repositioning utility across all zones and returns the top-k recommendations.

        Args:
            driver_lat: Current latitude of the driver.
            driver_lon: Current longitude of the driver.
            predicted_demands: Dict mapping zone_id -> predicted demand (rides/hour).
            predicted_surges: Dict mapping zone_id -> predicted surge multiplier.
            zone_supplies: Dict mapping zone_id -> number of active idle drivers.
            top_k: Number of recommendations to return.
            max_reposition_km: Maximum deadhead travel radius considered.

        Returns:
            List of ranked recommendation dicts.
        """
        if predicted_surges is None:
            predicted_surges = {z["zone_id"]: 1.0 for z in METRO_ZONES}
        if zone_supplies is None:
            zone_supplies = {z["zone_id"]: 5 for z in METRO_ZONES}

        max_demand = max(max(predicted_demands.values(), default=1.0), 1.0)
        evaluations: List[Dict[str, Any]] = []

        for zone in METRO_ZONES:
            z_id = zone["zone_id"]
            z_lat = zone["lat"]
            z_lon = zone["lon"]
            z_name = zone["name"]

            dist_km = haversine_distance_km(driver_lat, driver_lon, z_lat, z_lon)
            if dist_km > max_reposition_km and dist_km > 0.5:
                continue

            travel_time_min = (dist_km / self.driver_speed_kmh) * 60.0
            deadhead_cost = (dist_km * self.cost_per_km) + (
                (travel_time_min / 60.0) * (self.opportunity_cost_per_hour * 0.5)
            )

            demand = predicted_demands.get(z_id, 0.0)
            surge = predicted_surges.get(z_id, 1.0)
            supply = max(1, zone_supplies.get(z_id, 5))

            # Expected waiting time in minutes to get a ping once arriving
            # High demand + low supply => very quick ping (1-3 min)
            wait_time_min = min(25.0, max(1.5, 30.0 / (demand / supply + 0.1)))

            # Estimated revenue per trip: $16 base * surge
            est_trip_fare = 16.50 * surge
            # Hourly potential adjusted for turnaround
            trips_per_hour = min(3.0, 60.0 / (wait_time_min + 18.0))
            expected_gross_hourly = est_trip_fare * trips_per_hour

            # Net utility score = expected hourly revenue - deadhead repositioning cost penalty
            utility_score = expected_gross_hourly - (deadhead_cost * 1.5)

            bearing_deg, cardinal = calculate_bearing(driver_lat, driver_lon, z_lat, z_lon)

            # Rationale generation
            if dist_km < 0.8:
                rationale = f"Already in {z_name}. Solid demand ({demand:.0f} rides/hr). Stay positioned here."
            elif surge > 1.3:
                rationale = f"High surge ({surge:.1f}x) at {z_name}! Travel {dist_km:.1f} km ({travel_time_min:.0f} min) for high-fare trips."
            elif demand > max_demand * 0.75:
                rationale = f"Peak demand hotspot ({demand:.0f} rides/hr). Fast dispatch in ~{wait_time_min:.0f} min."
            else:
                rationale = f"Steady demand ({demand:.0f} rides/hr) within {dist_km:.1f} km."

            evaluations.append({
                "rank": 0,
                "zone_id": z_id,
                "zone_name": z_name,
                "zone_lat": z_lat,
                "zone_lon": z_lon,
                "distance_km": round(dist_km, 2),
                "travel_time_min": round(travel_time_min, 1),
                "deadhead_cost": round(deadhead_cost, 2),
                "bearing_deg": bearing_deg,
                "cardinal_direction": cardinal,
                "predicted_demand": round(demand, 1),
                "predicted_surge": round(surge, 2),
                "expected_wait_min": round(wait_time_min, 1),
                "expected_gross_hourly": round(expected_gross_hourly, 2),
                "utility_score": round(utility_score, 2),
                "rationale": rationale,
            })

        # Sort by utility_score descending
        evaluations.sort(key=lambda x: x["utility_score"], reverse=True)
        for i, item in enumerate(evaluations):
            item["rank"] = i + 1

        return evaluations[:top_k]
