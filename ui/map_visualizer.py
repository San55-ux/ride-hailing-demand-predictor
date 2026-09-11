"""
Interactive Map Visualization Module.
Renders PyDeck continuous heatmaps, 3D zone demand columns, driver position vectors,
and fleet dispatch distributions.
"""

import numpy as np
import pandas as pd
import pydeck as pdk
from typing import Dict, Any, List, Optional
from data.generator import METRO_ZONES


def get_demand_color(demand: float, max_demand: float) -> List[int]:
    """Returns an RGBA color gradient from cool cyan/green to fiery red based on demand intensity."""
    ratio = min(1.0, max(0.0, demand / max(max_demand, 1.0)))
    if ratio < 0.25:
        # Cyan to lime
        r = int(20 + ratio * 4 * 100)
        g = int(180 + ratio * 4 * 50)
        b = int(220 - ratio * 4 * 100)
    elif ratio < 0.6:
        # Lime to yellow/orange
        r = int(120 + (ratio - 0.25) * 2.8 * 135)
        g = int(230 - (ratio - 0.25) * 2.8 * 60)
        b = 20
    else:
        # Orange to intense magenta/red
        r = int(255)
        g = int(170 - (ratio - 0.6) * 2.5 * 150)
        b = int(20 + (ratio - 0.6) * 2.5 * 80)
    return [r, g, b, 200]


def build_demand_heatmap_deck(
    zone_demands_df: pd.DataFrame,
    driver_lat: Optional[float] = None,
    driver_lon: Optional[float] = None,
    top_recommendation: Optional[Dict[str, Any]] = None,
    map_style: str = "mapbox://styles/mapbox/dark-v11",
    view_elevation_3d: bool = True,
) -> pdk.Deck:
    """
    Constructs a PyDeck map containing:
    1. Continuous smooth HeatmapLayer representing customer pickup demand density
    2. Interactive ColumnLayer or ScatterplotLayer for zone centers and tooltips
    3. Marker for current driver position
    4. Repositioning vector line pointing to optimal hotspot
    """
    df = zone_demands_df.copy()
    max_dem = df["predicted_demand"].max() if not df.empty and "predicted_demand" in df.columns else 100.0

    # Format colors and elevations
    df["color"] = df["predicted_demand"].apply(lambda d: get_demand_color(d, max_dem))
    df["elevation"] = df["predicted_demand"].apply(lambda d: float(d) * 35.0)

    # Center map on Downtown or average coords
    center_lat = float(df["lat"].mean()) if not df.empty else 37.7749
    center_lon = float(df["lon"].mean()) if not df.empty else -122.4194

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11.5,
        pitch=45 if view_elevation_3d else 0,
        bearing=10,
    )

    layers: List[pdk.Layer] = []

    # 1. Smooth Heatmap Density Layer
    # Synthesize small point cloud around zone centers proportional to demand for smooth Gaussian heatmap
    heat_points = []
    for _, row in df.iterrows():
        n_pts = max(3, int(row["predicted_demand"] / 3.0))
        z_lat, z_lon = row["lat"], row["lon"]
        for _ in range(n_pts):
            heat_points.append({
                "lat": z_lat + np.random.normal(0, 0.006),
                "lon": z_lon + np.random.normal(0, 0.006),
                "weight": float(row["predicted_demand"]),
            })
    heat_df = pd.DataFrame(heat_points)

    if not heat_df.empty:
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=heat_df,
            get_position=["lon", "lat"],
            get_weight="weight",
            radius_pixels=65,
            intensity=1.2,
            threshold=0.08,
        )
        layers.append(heatmap_layer)

    # 2. Zone 3D Columns or High-Tech Rings
    if view_elevation_3d:
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=df,
            get_position=["lon", "lat"],
            get_elevation="elevation",
            elevation_scale=1,
            radius=450,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        layers.append(column_layer)
    else:
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["lon", "lat"],
            get_radius="elevation",
            radius_scale=8,
            radius_min_pixels=12,
            radius_max_pixels=40,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        layers.append(scatter_layer)

    # 3. Driver Marker & Repositioning Vector
    if driver_lat is not None and driver_lon is not None:
        driver_df = pd.DataFrame([{
            "lat": driver_lat,
            "lon": driver_lon,
            "label": "Your Current Location",
            "color": [0, 240, 255, 255],  # Neon Cyan
        }])
        driver_layer = pdk.Layer(
            "ScatterplotLayer",
            data=driver_df,
            get_position=["lon", "lat"],
            get_radius=300,
            radius_min_pixels=10,
            get_fill_color="color",
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=3,
            pickable=True,
        )
        layers.append(driver_layer)

        # Repositioning vector line
        if top_recommendation and top_recommendation.get("distance_km", 0) > 0.4:
            dest_lat = top_recommendation["zone_lat"]
            dest_lon = top_recommendation["zone_lon"]
            line_df = pd.DataFrame([{
                "source": [driver_lon, driver_lat],
                "target": [dest_lon, dest_lat],
                "color": [255, 215, 0, 220],  # Glowing Gold
            }])
            vector_layer = pdk.Layer(
                "LineLayer",
                data=line_df,
                get_source_position="source",
                get_target_position="target",
                get_color="color",
                get_width=5,
                width_min_pixels=3,
                pickable=False,
            )
            layers.append(vector_layer)

    tooltip = {
        "html": "<b>{zone_name}</b><br/>"
                "🔥 Predicted Demand: <b>{predicted_demand} rides/hr</b><br/>"
                "⚡ Surge Multiplier: <b>{predicted_surge}x</b><br/>"
                "ℹ️ Zone Type: <i>{type}</i>",
        "style": {
            "backgroundColor": "#181a20",
            "color": "#ffffff",
            "fontSize": "13px",
            "borderRadius": "8px",
            "padding": "10px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.5)",
        },
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=tooltip,
    )


def build_fleet_dispatch_deck(
    fleet_dispatches: List[Dict[str, Any]],
    zone_demands: Dict[int, float],
) -> pdk.Deck:
    """
    Renders multi-driver fleet vectors showing rebalancing movements across the city.
    """
    line_records = []
    driver_points = []
    zone_colors = {
        1: [255, 99, 132],
        2: [54, 162, 235],
        3: [255, 206, 86],
        4: [75, 192, 192],
        5: [153, 102, 255],
        6: [255, 159, 64],
        7: [201, 203, 207],
        8: [46, 204, 113],
        9: [231, 76, 60],
        10: [52, 152, 219],
    }

    for d in fleet_dispatches:
        target_z = d["target_zone"]
        color = zone_colors.get(target_z, [255, 255, 255]) + [200]
        # Target zone coordinates
        dest_zone = next(z for z in METRO_ZONES if z["zone_id"] == target_z)
        orig_zone = next(z for z in METRO_ZONES if z["zone_id"] == d["origin_zone"])

        # Origin point with slight noise
        orig_lat = orig_zone["lat"] + np.random.normal(0, 0.008)
        orig_lon = orig_zone["lon"] + np.random.normal(0, 0.008)

        driver_points.append({
            "lat": orig_lat,
            "lon": orig_lon,
            "driver_id": d["driver_id"],
            "color": color,
        })

        line_records.append({
            "source": [orig_lon, orig_lat],
            "target": [d["target_lon"], d["target_lat"]],
            "color": color,
            "driver_id": d["driver_id"],
            "target_name": d["target_zone_name"],
        })

    lines_df = pd.DataFrame(line_records)
    pts_df = pd.DataFrame(driver_points)

    view_state = pdk.ViewState(
        latitude=37.7749,
        longitude=-122.4194,
        zoom=11.2,
        pitch=30,
    )

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=pts_df,
            get_position=["lon", "lat"],
            get_radius=180,
            radius_min_pixels=4,
            get_fill_color="color",
            pickable=True,
        ),
        pdk.Layer(
            "LineLayer",
            data=lines_df,
            get_source_position="source",
            get_target_position="target",
            get_color="color",
            get_width=2,
            width_min_pixels=1.5,
            pickable=True,
        ),
    ]

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
    )
