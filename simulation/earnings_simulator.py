"""
Driver Shift and Earnings Simulator.
Simulates multi-hour driver shifts comparing:
1. Demand-Aware Positioning (AI Recommender)
2. Random Roaming (Cruising)
3. Static Waiting (Anchored at single hub)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from data.generator import METRO_ZONES, haversine_distance_km
from models.recommender import HotspotRecommender


class ShiftSimulationEngine:
    """
    Simulates operational shifts for a single driver using different dispatch & repositioning strategies.
    """

    def __init__(
        self,
        shift_hours: float = 8.0,
        fuel_cost_per_km: float = 0.40,
        avg_speed_kmh: float = 25.0,
        random_seed: int = 42,
    ):
        self.shift_hours = shift_hours
        self.shift_minutes = shift_hours * 60.0
        self.fuel_cost_per_km = fuel_cost_per_km
        self.avg_speed_kmh = avg_speed_kmh
        self.random_seed = random_seed
        self.recommender = HotspotRecommender(cost_per_km=fuel_cost_per_km, driver_speed_kmh=avg_speed_kmh)

    def run_shift(
        self,
        strategy: str,
        predicted_demands_by_hour: Dict[int, Dict[int, float]],
        predicted_surges_by_hour: Dict[int, Dict[int, float]],
        start_hour: int = 8,
        initial_zone_id: int = 8,  # Start in residential suburb
    ) -> Dict[str, Any]:
        """
        Simulates a shift for a given strategy.
        """
        rng = np.random.RandomState(self.random_seed + hash(strategy) % 10000)

        current_min = 0.0
        current_zone_id = initial_zone_id
        curr_zone = next(z for z in METRO_ZONES if z["zone_id"] == current_zone_id)
        current_lat = curr_zone["lat"]
        current_lon = curr_zone["lon"]

        trips_completed = 0
        gross_fare = 0.0
        surge_bonus = 0.0
        tips = 0.0
        deadhead_km = 0.0
        time_on_trip_min = 0.0
        time_repositioning_min = 0.0
        time_idle_min = 0.0

        trip_log: List[Dict[str, Any]] = []

        while current_min < self.shift_minutes:
            current_hour = (start_hour + int(current_min // 60)) % 24
            hourly_demand = predicted_demands_by_hour.get(current_hour, {})
            hourly_surge = predicted_surges_by_hour.get(current_hour, {})

            # Strategy decision on where to be
            if strategy == "ai_demand_aware":
                recs = self.recommender.recommend(
                    current_lat, current_lon, hourly_demand, hourly_surge, top_k=1
                )
                if recs and recs[0]["zone_id"] != current_zone_id:
                    target_zone = recs[0]
                    travel_km = target_zone["distance_km"]
                    travel_time = (travel_km / self.avg_speed_kmh) * 60.0

                    # Only reposition if within shift time and travel is reasonable
                    if current_min + travel_time <= self.shift_minutes and travel_km <= 10.0:
                        deadhead_km += travel_km
                        time_repositioning_min += travel_time
                        current_min += travel_time
                        current_zone_id = target_zone["zone_id"]
                        current_lat = target_zone["zone_lat"]
                        current_lon = target_zone["zone_lon"]

            elif strategy == "random_roaming":
                # Random cruising between nearby zones
                if rng.rand() > 0.45:
                    candidate = rng.choice(METRO_ZONES)
                    travel_km = haversine_distance_km(current_lat, current_lon, candidate["lat"], candidate["lon"])
                    travel_km = min(travel_km, 6.0)
                    travel_time = (travel_km / self.avg_speed_kmh) * 60.0
                    if current_min + travel_time <= self.shift_minutes:
                        deadhead_km += travel_km
                        time_repositioning_min += travel_time
                        current_min += travel_time
                        current_zone_id = candidate["zone_id"]
                        current_lat = candidate["lat"]
                        current_lon = candidate["lon"]

            elif strategy == "static_hub":
                # Remains anchored to Downtown (zone 1) or initial hub
                anchor_zone_id = 1
                if current_zone_id != anchor_zone_id:
                    anchor_z = next(z for z in METRO_ZONES if z["zone_id"] == anchor_zone_id)
                    travel_km = haversine_distance_km(current_lat, current_lon, anchor_z["lat"], anchor_z["lon"])
                    travel_time = (travel_km / self.avg_speed_kmh) * 60.0
                    if current_min + travel_time <= self.shift_minutes:
                        deadhead_km += travel_km
                        time_repositioning_min += travel_time
                        current_min += travel_time
                        current_zone_id = anchor_zone_id
                        current_lat = anchor_z["lat"]
                        current_lon = anchor_z["lon"]

            if current_min >= self.shift_minutes:
                break

            # In the current zone: Wait for a ping
            zone_demand = hourly_demand.get(current_zone_id, 20.0)
            zone_surge = hourly_surge.get(current_zone_id, 1.0)

            # Expected wait time in minutes (exponential distribution)
            # High demand zone (100 rides/hr) -> avg wait ~ 2.5 min. Low demand (10 rides/hr) -> avg wait ~ 12 min
            expected_wait = max(1.5, 60.0 / (zone_demand * 0.25 + 2.0))
            actual_wait = rng.exponential(scale=expected_wait)

            if current_min + actual_wait > self.shift_minutes:
                time_idle_min += (self.shift_minutes - current_min)
                break

            time_idle_min += actual_wait
            current_min += actual_wait

            # Execute passenger trip
            # Sample destination zone
            dest_zone = rng.choice(METRO_ZONES)
            trip_km = max(1.5, haversine_distance_km(current_lat, current_lon, dest_zone["lat"], dest_zone["lon"]))
            trip_time = max(5.0, (trip_km / self.avg_speed_kmh) * 60.0 + rng.normal(3.0, 1.0))

            if current_min + trip_time > self.shift_minutes:
                # Partial/final trip before shift end
                trip_time = self.shift_minutes - current_min
                current_min = self.shift_minutes
                break

            current_min += trip_time
            time_on_trip_min += trip_time

            # Financial calculation
            base_trip_fare = 3.50 + (trip_km * 1.85) + (trip_time * 0.35)
            surge_addition = base_trip_fare * (zone_surge - 1.0)
            trip_total = base_trip_fare * zone_surge
            trip_driver_share = trip_total * 0.75
            trip_tip = round(trip_driver_share * (0.16 if rng.rand() > 0.4 else 0.0), 2)

            gross_fare += trip_driver_share
            surge_bonus += (surge_addition * 0.75)
            tips += trip_tip
            trips_completed += 1

            trip_log.append({
                "trip_number": trips_completed,
                "pickup_time_min": round(current_min - trip_time, 1),
                "pickup_zone": current_zone_id,
                "dropoff_zone": dest_zone["zone_id"],
                "distance_km": round(trip_km, 2),
                "duration_min": round(trip_time, 1),
                "surge": round(zone_surge, 2),
                "driver_revenue": round(trip_driver_share + trip_tip, 2),
            })

            # Update driver location to dropoff destination
            current_zone_id = dest_zone["zone_id"]
            current_lat = dest_zone["lat"]
            current_lon = dest_zone["lon"]

        operating_expense = deadhead_km * self.fuel_cost_per_km
        total_gross = gross_fare + tips
        net_earnings = total_gross - operating_expense
        net_hourly_rate = net_earnings / self.shift_hours
        utilization_rate = (time_on_trip_min / self.shift_minutes) * 100.0

        return {
            "strategy": strategy,
            "trips_completed": trips_completed,
            "gross_fare": round(gross_fare, 2),
            "surge_bonus": round(surge_bonus, 2),
            "tips": round(tips, 2),
            "total_gross": round(total_gross, 2),
            "operating_expense": round(operating_expense, 2),
            "net_earnings": round(net_earnings, 2),
            "net_hourly_rate": round(net_hourly_rate, 2),
            "deadhead_km": round(deadhead_km, 1),
            "time_on_trip_min": round(time_on_trip_min, 1),
            "time_repositioning_min": round(time_repositioning_min, 1),
            "time_idle_min": round(time_idle_min, 1),
            "utilization_pct": round(utilization_rate, 1),
            "trip_log": trip_log,
        }

    def compare_strategies(
        self,
        predicted_demands_by_hour: Dict[int, Dict[int, float]],
        predicted_surges_by_hour: Dict[int, Dict[int, float]],
        start_hour: int = 8,
        n_trials: int = 5,
    ) -> pd.DataFrame:
        """
        Runs multiple Monte Carlo shift simulations across all 3 strategies and averages the results.
        """
        strategies = [
            ("ai_demand_aware", "AI Demand-Aware (Ours)"),
            ("random_roaming", "Random Roaming"),
            ("static_hub", "Static Hub (Downtown)"),
        ]

        summary_rows = []
        for strat_key, strat_label in strategies:
            trial_results = []
            for trial in range(n_trials):
                self.random_seed += trial * 13
                res = self.run_shift(strat_key, predicted_demands_by_hour, predicted_surges_by_hour, start_hour)
                trial_results.append(res)

            avg_trips = np.mean([r["trips_completed"] for r in trial_results])
            avg_gross = np.mean([r["total_gross"] for r in trial_results])
            avg_surge = np.mean([r["surge_bonus"] for r in trial_results])
            avg_expense = np.mean([r["operating_expense"] for r in trial_results])
            avg_net = np.mean([r["net_earnings"] for r in trial_results])
            avg_hourly = np.mean([r["net_hourly_rate"] for r in trial_results])
            avg_deadhead = np.mean([r["deadhead_km"] for r in trial_results])
            avg_util = np.mean([r["utilization_pct"] for r in trial_results])
            avg_idle = np.mean([r["time_idle_min"] for r in trial_results])

            summary_rows.append({
                "Strategy": strat_label,
                "Trips Completed": round(float(avg_trips), 1),
                "Gross Revenue ($)": round(float(avg_gross), 2),
                "Surge Bonus ($)": round(float(avg_surge), 2),
                "Fuel/Expense ($)": round(float(avg_expense), 2),
                "Net Earnings ($)": round(float(avg_net), 2),
                "Net Hourly Rate ($/hr)": round(float(avg_hourly), 2),
                "Deadhead (km)": round(float(avg_deadhead), 1),
                "Utilization (%)": round(float(avg_util), 1),
                "Idle Wait (min)": round(float(avg_idle), 1),
            })

        return pd.DataFrame(summary_rows)
