"""
UI Component Builders: Plotly charts, metrics cards, and evaluation displays.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List


def create_strategy_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Creates a grouped bar chart comparing key metrics across the 3 driver strategies.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Net Earnings ($)",
        x=comparison_df["Strategy"],
        y=comparison_df["Net Earnings ($)"],
        marker_color="#00C897",
        text=[f"${v:.0f}" for v in comparison_df["Net Earnings ($)"]],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name="Surge Bonus ($)",
        x=comparison_df["Strategy"],
        y=comparison_df["Surge Bonus ($)"],
        marker_color="#FFB319",
        text=[f"${v:.0f}" for v in comparison_df["Surge Bonus ($)"]],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name="Fuel Expense ($)",
        x=comparison_df["Strategy"],
        y=comparison_df["Fuel Expense ($)"],
        marker_color="#FF4C61",
        text=[f"-${v:.0f}" for v in comparison_df["Fuel Expense ($)"]],
        textposition="auto",
    ))

    fig.update_layout(
        title="<b>8-Hour Shift Financial Comparison by Strategy</b>",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Segoe UI, Roboto, sans-serif"),
    )
    return fig


def create_hourly_demand_curve(
    hourly_df: pd.DataFrame,
    selected_zones: List[str],
) -> go.Figure:
    """
    Creates a line chart showing 24-hour demand evolution across selected urban zones.
    """
    fig = go.Figure()
    palette = px.colors.qualitative.Safe

    for i, zone_name in enumerate(selected_zones):
        z_data = hourly_df[hourly_df["zone_name"] == zone_name].sort_values("hour")
        if z_data.empty:
            continue
        fig.add_trace(go.Scatter(
            x=z_data["hour"],
            y=z_data["demand_count"],
            mode="lines+markers",
            name=zone_name,
            line=dict(width=3, color=palette[i % len(palette)]),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="<b>24-Hour Spatio-Temporal Demand Waves by Zone</b>",
        xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Trip Requests / Hour"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_feature_importance_chart(fi_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Creates a horizontal bar chart showing top predictive features.
    """
    top_fi = fi_df.head(top_n).sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=top_fi["importance"],
        y=top_fi["feature"],
        orientation="h",
        marker=dict(
            color=top_fi["importance"],
            colorscale="Viridis",
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in top_fi["importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"<b>Top {top_n} Demand Predictor Feature Importances</b>",
        xaxis=dict(title="Relative Importance"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=40, t=40, b=20),
    )
    return fig


def create_load_balance_comparison_chart(
    greedy_report: pd.DataFrame,
    balanced_report: pd.DataFrame,
) -> go.Figure:
    """
    Compares supply/demand ratio across zones for Naive Greedy vs Coordinated Load Balancing.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Naive Greedy Dispatch",
        x=greedy_report["zone_name"],
        y=greedy_report["supply_demand_ratio"],
        marker_color="#FF5252",
    ))

    fig.add_trace(go.Bar(
        name="Coordinated Load Balanced",
        x=balanced_report["zone_name"],
        y=balanced_report["supply_demand_ratio"],
        marker_color="#00E676",
    ))

    # Add ideal balance reference line (ratio = 1.0)
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(greedy_report) - 0.5,
        y0=1.0,
        y1=1.0,
        line=dict(color="white", width=2, dash="dash"),
    )

    fig.update_layout(
        title="<b>Supply-to-Demand Coverage Ratio by Zone (Ideal = 1.0)</b>",
        yaxis=dict(title="Supply / Demand Ratio (1.0 = Equilibrium)"),
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
