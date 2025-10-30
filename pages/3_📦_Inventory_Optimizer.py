"""
Inventory Optimizer Demo - Streamlit Page

Pipeline:
1. Load pre-generated inventory snapshot (50 SKUs)
2. Analyze stock health (low stock, overstock, fast movers)
3. Forecast 30-day demand with Prophet per SKU
4. Generate reorder plan using LangChain + Gemini
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.inventory_agent import InventoryAgent

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Inventory Optimizer",
    page_icon="📦",
    layout="wide",
)

st.title("📦 AI Inventory Optimizer")
st.markdown(
    """
Get an end-to-end look at how AI can manage inventory:

1. **Load & inspect** the current stock levels  
2. **Forecast demand** for every SKU with Prophet  
3. **Generate reorder recommendations** using Gemini via LangChain  
4. **See the full prompt + response** for transparency
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "your_api_key_here":
    st.error("⚠️ Google Gemini API key not configured!")
    st.info(
        """
        **Setup instructions**
        1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create an API key
        3. Add it to `.env` as `GOOGLE_API_KEY=your_key`
        4. Restart the Streamlit app
        """
    )
    st.stop()

data_path = Path(__file__).parent.parent / "data" / "inventory.csv"
if not data_path.exists():
    st.error(f"Inventory data not found at `{data_path}`.")
    st.stop()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
state_defaults = {
    "inventory_agent": None,
    "inventory_ready": False,
    "inventory_summary": None,
    "inventory_analysis": None,
    "inventory_forecast": None,
    "inventory_recommendations": None,
    "inventory_steps": [],
    "inventory_error": None,
    "inventory_last_run": None,
}
for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def log_step(label: str, status: str, details: Optional[Dict] = None) -> None:
    """Append a step to the behind-the-scenes log."""
    st.session_state.inventory_steps.append(
        {
            "label": label,
            "status": status,
            "details": details or {},
            "timestamp": time.strftime("%H:%M:%S"),
        }
    )


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------
col_action, col_last_run = st.columns([2, 1])
with col_action:
    run_analysis = st.button("🚚 Analyze Current Inventory", type="primary")

with col_last_run:
    if st.session_state.inventory_last_run:
        elapsed = int(time.time() - st.session_state.inventory_last_run)
        st.caption(f"Last run completed {elapsed} seconds ago.")


if run_analysis:
    progress_box = st.empty()
    st.session_state.inventory_steps = []
    st.session_state.inventory_error = None

    try:
        progress_box.info("Initializing inventory agent...")
        agent = InventoryAgent(api_key=api_key, data_path=str(data_path))
        st.session_state.inventory_agent = agent

        progress_box.info("Step 1/4 • Loading inventory snapshot...")
        summary = agent.load_inventory()
        st.session_state.inventory_summary = summary
        log_step("Load inventory data", "success", summary)

        progress_box.info("Step 2/4 • Evaluating stock health...")
        analysis = agent.analyze_inventory()
        st.session_state.inventory_analysis = analysis
        log_step(
            "Analyze stock signals",
            "success",
            {
                "low_stock": analysis["summary"]["low_stock_count"],
                "overstock": analysis["summary"]["overstock_count"],
                "fast_movers": analysis["summary"]["fast_mover_count"],
            },
        )

        progress_box.info("Step 3/4 • Forecasting 30-day demand (Prophet)...")
        forecast_df = agent.forecast_demand(horizon_days=30)
        st.session_state.inventory_forecast = forecast_df
        log_step("Forecast demand", "success", {"models_built": len(forecast_df)})

        progress_box.info("Step 4/4 • Generating Gemini recommendations...")
        recommendations = agent.generate_recommendations()
        st.session_state.inventory_recommendations = recommendations
        log_step(
            "Generate reorder plan",
            "success",
            {
                "recommendations": len(recommendations.get("recommendations", [])),
                "estimated_cost": recommendations["token_usage"]["estimated_cost"],
            },
        )

        st.session_state.inventory_ready = True
        st.session_state.inventory_last_run = time.time()
        progress_box.success("Inventory analysis complete!")
    except Exception as exc:
        st.session_state.inventory_error = str(exc)
        st.session_state.inventory_ready = False
        progress_box.error("Unable to complete the analysis.")
        st.error(f"Run failed: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.inventory_ready:
    if st.session_state.inventory_error:
        st.warning("Check the error above, update your configuration, and try again.")
    else:
        st.info(
            "Click **Analyze Current Inventory** to see the AI-driven pipeline in action. "
            "You'll watch each step run, and the final reorder plan will appear here."
        )
    st.stop()

# ---------------------------------------------------------------------------
# Helper functions for display
# ---------------------------------------------------------------------------
def derive_status(row: pd.Series) -> str:
    if row["current_stock"] < row["reorder_point"]:
        return "Low Stock"
    if row["current_stock"] > row["reorder_point"] * 1.6 and row["last_30_days_sales"] < 10:
        return "Overstock"
    if row["last_30_days_sales"] > 50:
        return "Fast Mover"
    return "Healthy"


def style_rows(row: pd.Series) -> List[str]:
    status = row["status"]
    color_map = {
        "Low Stock": "#fee2e2",
        "Overstock": "#fef3c7",
        "Fast Mover": "#d1fae5",
        "Healthy": "#f8fafc",
    }
    color = color_map.get(status, "#ffffff")
    return [f"background-color: {color}"] * len(row)


agent: InventoryAgent = st.session_state.inventory_agent
inventory_df = agent.df.copy()
inventory_df["status"] = inventory_df.apply(derive_status, axis=1)
inventory_df["inventory_value"] = inventory_df["current_stock"] * inventory_df["cost_per_unit"]

forecast_df = st.session_state.inventory_forecast.copy()
recommendations = st.session_state.inventory_recommendations
analysis_summary = st.session_state.inventory_analysis["summary"]
summary = st.session_state.inventory_summary


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
metrics_col1.metric("Total Inventory Value", f"${summary['total_inventory_value']:,.2f}")
metrics_col2.metric("Average Lead Time", f"{summary['average_lead_time']:.1f} days")
metrics_col3.metric("Low Stock SKUs", analysis_summary["low_stock_count"])
metrics_col4.metric("Fast Movers", analysis_summary["fast_mover_count"])

st.caption(
    "Transparent cost tracking: "
    f"estimated Gemini cost for this run **${recommendations['token_usage']['estimated_cost']:.4f}** "
    f"({recommendations['token_usage']['total_tokens']} tokens)"
)

st.divider()

# ---------------------------------------------------------------------------
# Behind the scenes
# ---------------------------------------------------------------------------
with st.expander("🔍 Behind the Scenes (prompts, steps, raw output)", expanded=True):
    for step in st.session_state.inventory_steps:
        badge = "✅" if step["status"] == "success" else "⚠️"
        st.markdown(f"**{badge} {step['label']}** — {step['timestamp']}")
        if step["details"]:
            st.json(step["details"])
        st.markdown("---")

    st.markdown("**Prompt (truncated preview)**")
    st.code(recommendations["prompt_preview"][:2000], language="json")

    st.markdown("**Gemini Raw Response**")
    st.code(recommendations["raw_response"], language="json")

st.divider()

# ---------------------------------------------------------------------------
# Inventory snapshot
# ---------------------------------------------------------------------------
st.subheader("Current Inventory Snapshot")

styled_inventory = (
    inventory_df[
        [
            "sku",
            "product_name",
            "category",
            "status",
            "current_stock",
            "reorder_point",
            "lead_time_days",
            "last_30_days_sales",
            "inventory_value",
        ]
    ]
    .sort_values("status")
    .reset_index(drop=True)
    .style.apply(style_rows, axis=1)
    .format(
        {
            "inventory_value": "${:,.2f}",
            "current_stock": "{:,.0f}",
            "reorder_point": "{:,.0f}",
            "lead_time_days": "{:,.0f}",
            "last_30_days_sales": "{:,.0f}",
        }
    )
)

st.dataframe(styled_inventory, use_container_width=True, height=480)

# ---------------------------------------------------------------------------
# Reorder recommendations
# ---------------------------------------------------------------------------
st.subheader("🤖 Gemini Reorder Recommendations")

recommendations_df = pd.DataFrame(recommendations.get("recommendations", []))
if recommendations_df.empty:
    st.warning("Gemini did not flag any items for action. Review the raw output above.")
else:
    if "estimated_cost" in recommendations_df.columns:
        recommendations_df["estimated_cost"] = recommendations_df["estimated_cost"].astype(float)

    display_cols = [
        col
        for col in [
            "sku",
            "product_name",
            "priority",
            "recommended_order_qty",
            "estimated_cost",
            "reason",
        ]
        if col in recommendations_df.columns
    ]
    st.dataframe(
        recommendations_df[display_cols],
        use_container_width=True,
        height=400,
    )

if recommendations.get("overall_notes"):
    st.info(f"**Planner Notes:** {recommendations['overall_notes']}")

st.divider()

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
st.subheader("📊 Visual Insights")

status_colors = {
    "Low Stock": "#ef4444",
    "Overstock": "#f59e0b",
    "Fast Mover": "#10b981",
    "Healthy": "#2563eb",
}

inventory_sorted = inventory_df.sort_values("status")

fig_inventory_levels = px.bar(
    inventory_sorted,
    x="product_name",
    y="current_stock",
    color="status",
    color_discrete_map=status_colors,
    labels={"current_stock": "Units on Hand", "product_name": "Product"},
)
fig_inventory_levels.add_scatter(
    x=inventory_sorted["product_name"],
    y=inventory_sorted["reorder_point"],
    mode="markers",
    marker=dict(color="#111827", symbol="diamond", size=8),
    name="Reorder Point",
)
fig_inventory_levels.update_layout(
    height=420,
    margin=dict(l=40, r=40, t=40, b=160),
    xaxis_tickangle=60,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fastest = (
    forecast_df.dropna(subset=["coverage_days"])
    .sort_values("coverage_days")
    .head(3)
)

if fastest.empty:
    st.info("Prophet coverage data unavailable for visualization this run.")
else:
    forecast_tabs = st.tabs([f"{row['product_name']} ({row['sku']})" for _, row in fastest.iterrows()])
    for (_, row), tab in zip(fastest.iterrows(), forecast_tabs):
        with tab:
            timeline = agent.forecast_timelines.get(row["sku"])
            if timeline is None or timeline.empty:
                st.warning("No forecast series available.")
                continue

            fig_forecast = go.Figure()
            fig_forecast.add_trace(
                go.Scatter(
                    x=timeline["ds"],
                    y=timeline["yhat"],
                    name="Predicted Demand",
                    line=dict(color="#2563eb"),
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=timeline["ds"],
                    y=timeline["yhat_upper"],
                    name="Upper Bound",
                    line=dict(color="#93c5fd", dash="dot"),
                    showlegend=False,
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=timeline["ds"],
                    y=timeline["yhat_lower"],
                    name="Lower Bound",
                    fill="tonexty",
                    line=dict(color="#bfdbfe", dash="dot"),
                    fillcolor="rgba(37,99,235,0.1)",
                    showlegend=False,
                )
            )
            fig_forecast.add_hline(
                y=row["avg_daily_demand"],
                line=dict(color="#f97316", dash="dash"),
                annotation_text="Avg Daily Demand",
            )
            fig_forecast.update_layout(
                height=360,
                margin=dict(l=40, r=40, t=40, b=40),
                yaxis_title="Units per Day",
                xaxis_title="Date",
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    st.plotly_chart(fig_inventory_levels, use_container_width=True)

with col_chart2:
    if recommendations_df.empty:
        st.info("Run with an active API key to visualize recommendation priorities.")
    else:
        if "priority" not in recommendations_df.columns:
            st.info("Gemini response did not include priority fields this run.")
        else:
            priority_counts = recommendations_df["priority"].value_counts().reset_index()
            priority_counts.columns = ["priority", "count"]
            fig_priority = px.bar(
                priority_counts,
                x="priority",
                y="count",
                color="priority",
                color_discrete_map={
                    "HIGH": "#ef4444",
                    "MEDIUM": "#f59e0b",
                    "LOW": "#10b981",
                },
                title="Recommendation Priority Mix",
                labels={"priority": "Priority", "count": "SKUs"},
            )
            fig_priority.update_layout(height=320, margin=dict(l=40, r=30, t=60, b=60))
            st.plotly_chart(fig_priority, use_container_width=True)

st.caption(
    "Tip: Rerun the demo to see fresh Gemini responses — prompts and reasoning stay visible for transparency."
)
