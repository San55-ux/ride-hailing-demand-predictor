"""
Data Generator for Ride-Hailing Spatio-Temporal Trip Requests.
Simulates realistic urban ride-hailing demand across zones, hours, days, and weather conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Predefined urban zones with realistic centers, characteristics, and base hourly weights
METRO_ZONES = [
    {
        "zone_id": 1,
        "name": "Downtown Financial",
        "lat": 37.7891,
        "lon": -122.4014,
        "radius_km": 1.2,
        "type": "commercial",
        "base_volume": 120,
    },
    {
        "zone_id": 2,
        "name": "Midtown Tech Hub",
        "lat": 37.7765,
        "lon": -122.4172,
        "radius_km": 1.5,
        "type": "commercial",
        "base_volume": 110,
    },
    {
        "zone_id": 3,
        "name": "Metro Int'l Airport",
        "lat": 37.6213,
        "lon": -122.3790,
        "radius_km": 2.2,
        "type": "transit",
        "base_volume": 140,
    },
    {
        "zone_id": 4,
        "name": "Waterfront & Pier Dining",
        "lat": 37.8080,
        "lon": -122.4177,
        "radius_km": 1.0,
        "type": "entertainment",
        "base_volume": 90,
    },
    {
        "zone_id": 5,
        "name": "Mission Nightlife & Arts",
        "lat": 37.7599,
        "lon": -122.4148,
        "radius_km": 1.3,
        "type": "nightlife",
        "base_volume": 100,
    },
    {
        "zone_id": 6,
        "name": "University Campus & Med",
        "lat": 37.7635,
        "lon": -122.4578,
        "radius_km": 1.4,
        "type": "education",
        "base_volume": 75,
    },
    {
        "zone_id": 7,
        "name": "Marina & Golden Gate View",
        "lat": 37.8010,
        "lon": -122.4370,
        "radius_km": 1.2,
        "type": "residential_upscale",
        "base_volume": 65,
    },
    {
        "zone_id": 8,
        "name": "Sunset Residential District",
        "lat": 37.7530,
        "lon": -122.4860,
        "radius_km": 2.0,
        "type": "residential",
        "base_volume": 55,
    },
    {
        "zone_id": 9,
        "name": "Central Railway Station",
        "lat": 37.7770,
        "lon": -122.3950,
        "radius_km": 0.9,
        "type": "transit",
        "base_volume": 85,
    },
    {
        "zone_id": 10,
        "name": "Convention Center Arena",
        "lat": 37.7840,
        "lon": -122.4010,
        "radius_km": 1.1,
        "type": "events",
        "base_volume": 80,
    },
]

ZONES_DF = pd.DataFrame(METRO_ZONES)


def get_hourly_profile(zone_type: str, hour: int, is_weekend: bool) -> float:
    """Calculates demand multiplier based on zone category, hour of day, and weekend flag."""
    if zone_type == "commercial":
        # Peaks morning commute (7-9) and evening commute (17-19) on weekdays
        if is_weekend:
            return 0.35 + 0.3 * np.sin((hour - 12) / 12 * np.pi)
        if 7 <= hour <= 9:
            return 2.2 + 0.4 * np.sin((hour - 7) / 2 * np.pi)
        elif 16 <= hour <= 19:
            return 2.5 + 0.5 * np.sin((hour - 16) / 3 * np.pi)
        elif 10 <= hour <= 15:
            return 1.1
        else:
            return 0.25

    elif zone_type == "nightlife":
        # Low during day, surges Friday/Saturday nights (21:00 - 03:00)
        if is_weekend:
            if hour >= 20 or hour <= 3:
                return 3.0 + (0.6 if hour in [22, 23, 0, 1] else 0.0)
            elif 12 <= hour < 20:
                return 1.4
            else:
                return 0.3
        else:
            if 20 <= hour <= 23:
                return 1.6
            elif 12 <= hour < 20:
                return 0.9
            else:
                return 0.2

    elif zone_type == "transit":
        # Airport and train station: steady, peaks at flight arrival banks (11am-2pm, 7pm-10pm)
        if 6 <= hour <= 23:
            base = 1.3 + 0.5 * np.sin((hour - 6) / 17 * np.pi * 2)
            return max(0.6, base)
        return 0.4

    elif zone_type == "entertainment":
        # Waterfront & piers peak afternoon and evening weekends
        if is_weekend:
            if 12 <= hour <= 21:
                return 2.4
            return 0.7
        else:
            if 17 <= hour <= 21:
                return 1.5
            return 0.6

    elif zone_type == "residential" or zone_type == "residential_upscale":
        # Morning departures (7-9am), evening arrivals
        if is_weekend:
            return 0.9 + 0.4 * np.sin((hour - 10) / 14 * np.pi)
        if 7 <= hour <= 9:
            return 2.1
        elif 18 <= hour <= 21:
            return 1.2
        return 0.5

    elif zone_type == "events":
        # Evening spikes (19:00 - 23:00)
        if 18 <= hour <= 22:
            return 2.6
        return 0.5

    else:
        return 1.0


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates geodesic distance between two points in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def generate_synthetic_trips(
    days: int = 14,
    start_date: Optional[datetime] = None,
    random_seed: int = 42,
    sample_rate: float = 0.25,
) -> pd.DataFrame:
    """
    Generates realistic synthetic ride-hailing trip requests over a multi-day span.
    
    Args:
        days: Number of days to simulate.
        start_date: Starting datetime. Defaults to 14 days before today.
        random_seed: Seed for reproducible data generation.
        sample_rate: Downsampling fraction to balance realism with performance (0.1 to 1.0).
    
    Returns:
        DataFrame containing trip requests with timestamps, coords, fares, weather, surge, etc.
    """
    np.random.seed(random_seed)
    if start_date is None:
        start_date = datetime(2026, 9, 1, 0, 0, 0)

    weather_types = ["Clear", "Cloudy", "Rain", "Fog"]
    weather_multipliers = {"Clear": 1.0, "Cloudy": 1.05, "Rain": 1.45, "Fog": 1.15}

    records: List[Dict] = []
    total_hours = days * 24

    for h in range(total_hours):
        current_time = start_date + timedelta(hours=h)
        hour = current_time.hour
        day_of_week = current_time.weekday()
        is_weekend = day_of_week >= 5

        # Hourly weather simulation (Markov-like consistency or daily blocks)
        weather_seed = int((h // 6) * 17 + random_seed) % 100
        if weather_seed < 60:
            weather = "Clear"
        elif weather_seed < 80:
            weather = "Cloudy"
        elif weather_seed < 95:
            weather = "Rain"
        else:
            weather = "Fog"

        w_mult = weather_multipliers[weather]

        # Generate requests for each zone
        for zone in METRO_ZONES:
            z_id = zone["zone_id"]
            z_name = zone["name"]
            z_lat = zone["lat"]
            z_lon = zone["lon"]
            z_type = zone["type"]
            base_vol = zone["base_volume"]

            profile = get_hourly_profile(z_type, hour, is_weekend)
            # Expected trips in this 1-hour window
            expected_requests = base_vol * profile * w_mult * sample_rate
            # Add stochastic Poisson-like variation
            num_requests = int(np.random.poisson(max(1, expected_requests)))

            if num_requests == 0:
                continue

            # Surge multiplier calculation: supply constraint heuristic
            # If demand is unusually high for the zone, surge activates
            demand_ratio = profile * w_mult
            if demand_ratio > 2.0:
                surge = round(1.0 + (demand_ratio - 2.0) * 0.5 + np.random.uniform(0.0, 0.3), 2)
            elif demand_ratio > 1.4:
                surge = round(1.0 + (demand_ratio - 1.4) * 0.25, 2)
            else:
                surge = 1.0
            surge = min(surge, 3.2)

            for i in range(num_requests):
                # Request timestamp spread across the hour
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)
                req_time = current_time + timedelta(minutes=int(minute), seconds=int(second))

                # Pickup coords: Gaussian scatter around zone centroid (1 deg lat ~ 111 km)
                sigma_lat = (zone["radius_km"] * 0.5) / 111.0
                sigma_lon = (zone["radius_km"] * 0.5) / (111.0 * np.cos(np.radians(z_lat)))
                pickup_lat = round(float(np.random.normal(z_lat, sigma_lat)), 5)
                pickup_lon = round(float(np.random.normal(z_lon, sigma_lon)), 5)

                # Destination selection: pick another zone
                dest_zone_idx = np.random.choice([idx for idx, z in enumerate(METRO_ZONES) if z["zone_id"] != z_id])
                dest_zone = METRO_ZONES[dest_zone_idx]
                dest_lat = round(float(np.random.normal(dest_zone["lat"], 0.015)), 5)
                dest_lon = round(float(np.random.normal(dest_zone["lon"], 0.015)), 5)

                trip_dist_km = max(1.2, haversine_distance_km(pickup_lat, pickup_lon, dest_lat, dest_lon))
                # Avg speed 24 km/h in city traffic, duration in minutes
                trip_duration_min = round(max(4.0, (trip_dist_km / 24.0) * 60.0 + np.random.normal(2.0, 1.0)), 1)

                # Fare calculation: $3.50 base + $1.80/km + $0.35/min * surge
                base_fare = 3.50 + (trip_dist_km * 1.80) + (trip_duration_min * 0.35)
                total_fare = round(base_fare * surge, 2)
                tip = round(total_fare * (0.15 if np.random.rand() > 0.4 else 0.0), 2)

                passenger_count = int(np.random.choice([1, 2, 3, 4], p=[0.68, 0.20, 0.08, 0.04]))

                records.append({
                    "request_id": f"REQ_{req_time.strftime('%Y%m%d%H%M')}_{z_id}_{i:03d}",
                    "timestamp": req_time,
                    "pickup_zone_id": z_id,
                    "pickup_zone_name": z_name,
                    "pickup_lat": pickup_lat,
                    "pickup_lon": pickup_lon,
                    "dropoff_zone_id": dest_zone["zone_id"],
                    "dropoff_zone_name": dest_zone["name"],
                    "dropoff_lat": dest_lat,
                    "dropoff_lon": dest_lon,
                    "trip_distance_km": round(trip_dist_km, 2),
                    "trip_duration_min": trip_duration_min,
                    "surge_multiplier": surge,
                    "base_fare": round(base_fare, 2),
                    "total_fare": total_fare,
                    "tip": tip,
                    "driver_earnings": round((total_fare * 0.75) + tip, 2),  # 75% driver take rate
                    "passenger_count": passenger_count,
                    "weather": weather,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": int(is_weekend),
                })

    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    print("Generating synthetic ride-hailing sample dataset...")
    df_sample = generate_synthetic_trips(days=7, sample_rate=0.20)
    print(f"Generated {len(df_sample)} trip records.")
    print("Sample:\n", df_sample.head(3))
