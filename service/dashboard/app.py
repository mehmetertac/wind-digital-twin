"""Streamlit dashboard — thermal twin timeline, anomaly score, and RUL band."""

from __future__ import annotations

import altair as alt
import httpx
import pandas as pd
import streamlit as st
from client import (
    api_url,
    fetch_attributions,
    fetch_health,
    fetch_series,
    fetch_turbines,
    filter_series_to_date,
)


st.set_page_config(page_title="Wind twin dashboard", layout="wide")
st.title("Wind gearbox digital twin")
st.caption(f"API: `{api_url()}`")

try:
    health = fetch_health()
except httpx.HTTPError as exc:
    st.error(f"Cannot reach API at {api_url()}: {exc}")
    st.stop()

if health.get("status") != "ok":
    st.warning(f"API unhealthy: {health.get('detail', 'unknown error')}")
    st.stop()

with st.sidebar:
    st.subheader("Controls")
    try:
        turbines = fetch_turbines()
    except httpx.HTTPError as exc:
        st.error(f"Failed to load turbines: {exc}")
        st.stop()

    if not turbines:
        st.warning("No turbines in the timeline cache.")
        st.stop()

    turbine_id = st.selectbox("Turbine", turbines)

try:
    payload = fetch_series(turbine_id=turbine_id)
except httpx.HTTPError as exc:
    st.error(f"Series request failed: {exc}")
    st.stop()

full_df = pd.DataFrame(payload["rows"])
if full_df.empty:
    st.warning("No timeline rows for this turbine.")
    st.stop()

full_df["date"] = pd.to_datetime(full_df["date"], utc=True)
dates = full_df["date"].dt.strftime("%Y-%m-%d").tolist()
selected_label = st.sidebar.select_slider("Day in failure window", options=dates, value=dates[-1])
chart_df = filter_series_to_date(full_df, f"{selected_label}T00:00:00Z")
threshold = float(payload.get("anomaly_threshold", 0.0))

metrics = st.columns(4)
metrics[0].metric("Model", health.get("model_version", "—"))
metrics[1].metric("Turbine", turbine_id)
metrics[2].metric("Anomaly threshold", f"{threshold:.3f}")
if not chart_df.empty and "rul_p50" in chart_df.columns:
    metrics[3].metric("RUL P50 (selected day)", f"{float(chart_df['rul_p50'].iloc[-1]):.1f} d")
else:
    metrics[3].metric("RUL P50 (selected day)", "—")

st.subheader("Panel 1 — Oil temperature vs physics expectation")
temp_chart = alt.Chart(chart_df).encode(x=alt.X("date:T", title="Day (UTC)"))
actual_line = temp_chart.mark_line(color="black").encode(y=alt.Y("oil_temp_actual:Q", title="°C"))
physics_line = temp_chart.mark_line(color="#E45756").encode(y="oil_temp_physics:Q")
residual_line = temp_chart.mark_line(color="#72B7B2", strokeDash=[4, 2]).encode(
    y=alt.Y("oil_residual:Q", title="Hybrid residual (°C)")
)
st.altair_chart(
    alt.layer(actual_line, physics_line, residual_line).resolve_scale(y="independent").properties(height=280),
    use_container_width=True,
)

st.subheader("Panel 2 — Hybrid anomaly score")
score_chart = (
    alt.Chart(chart_df)
    .mark_line(color="#4C78A8")
    .encode(
        x=alt.X("date:T", title="Day (UTC)"),
        y=alt.Y("anomaly_score:Q", title="Score"),
    )
    .properties(height=220)
)
rule = (
    alt.Chart(pd.DataFrame({"threshold": [threshold]}))
    .mark_rule(color="#F58518", strokeDash=[6, 4])
    .encode(y="threshold:Q")
)
st.altair_chart(score_chart + rule, use_container_width=True)

st.subheader("Panel 3 — RUL P50 with conformal band")
if {"rul_p50", "cqr80_lo", "cqr80_hi"}.issubset(chart_df.columns):
    band = (
        alt.Chart(chart_df)
        .mark_area(opacity=0.25, color="#4C78A8")
        .encode(
            x=alt.X("date:T", title="Day (UTC)"),
            y=alt.Y("cqr80_lo:Q", title="Days"),
            y2="cqr80_hi:Q",
        )
    )
    p50 = (
        alt.Chart(chart_df)
        .mark_line(color="#4C78A8", strokeWidth=2)
        .encode(x="date:T", y="rul_p50:Q")
    )
    st.altair_chart(
        alt.layer(band, p50).resolve_scale(y="shared").properties(height=260),
        use_container_width=True,
    )
else:
    st.info("RUL columns missing from timeline export.")

st.subheader("Sensor attributions (global P10 mean |SHAP|)")
try:
    attrs = pd.DataFrame(fetch_attributions())
    bar = (
        alt.Chart(attrs)
        .mark_bar(color="#72B7B2")
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
            y=alt.Y("sensor:N", sort="-x", title=""),
        )
        .properties(height=240)
    )
    st.altair_chart(bar, use_container_width=True)
except httpx.HTTPError as exc:
    st.warning(f"Could not load attributions: {exc}")

with st.expander("Raw timeline rows (scrubbed)"):
    st.dataframe(chart_df, use_container_width=True)
