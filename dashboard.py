"""Streamlit dashboard for interactive GDELT forecasting."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gdelt_timeseries import (
    MODEL_LABELS,
    SMOOTH_WINDOWS,
    build_forecast_figure,
    fetch_timeline_vol,
    generate_forecast_artifacts,
    timeline_to_daily_arrays,
)

st.set_page_config(page_title="GDELT Forecast Explorer", layout="wide", page_icon="📈")

st.title("GDELT Forecast Explorer")
st.write(
    "Use this tool to pull daily media coverage counts from GDELT, score models on a recent backtest window, "
    "and generate multi-model forecasts into the future. Provide a search expression, choose a timespan, then "
    "compare TimesFM, Prophet, ARIMA, and Exponential Smoothing forecasts with consistent evaluation metrics."
)

DEFAULT_QUERY = '"artificial intelligence"'
MODEL_OPTIONS = {label: key for key, label in MODEL_LABELS.items()}


def _load_timeline(
    query: str,
    *,
    mode: str = "TimelineVol",
    timespan: str = "52weeks",
    extra_params: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Dict[str, object]:
  payload = fetch_timeline_vol(
      query=query,
      mode=mode,
      timespan=timespan,
      timeline_smooth=None,
      extra_params=extra_params,
      timeout=timeout,
  )
  arrays, timestamps = timeline_to_daily_arrays(payload)
  if not arrays:
    raise SystemExit("No timeline data returned for the given query.")
  return {"arrays": arrays, "timestamps": timestamps}


def _timeline_figure(times: List, values: np.ndarray, *, query: str) -> go.Figure:
  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=times,
          y=values,
          mode="lines",
          name="timeline volume",
          line=dict(color="#1f77b4", width=2.0),
      )
  )
  fig.update_layout(
      template="simple_white",
      margin=dict(l=40, r=20, t=40, b=40),
      hovermode="x unified",
      title=dict(text=f"GDELT coverage volume — {query}", x=0.5, xanchor="center"),
      yaxis=dict(title="Volume intensity", showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
      xaxis=dict(showgrid=False),
  )
  return fig


if "timeline_state" not in st.session_state:
  st.session_state.timeline_state = None

st.markdown("### Step 1: Generate time series data")
st.write(
    "Enter a GDELT search expression and timespan to retrieve an aggregated daily coverage timeline. "
    "For best results: wrap multi-word phrases in double quotes (e.g. `\"climate change\"`) and place OR clauses inside parentheses (e.g. `(\"Tesla\" OR \"SpaceX\")`). "
    "AND expressions can be written as `\"climate change\" AND \"policy\"`. Single-word terms do not require quotes. Timespans accept values such as `52weeks`, `24months`, or `365` (days)."
)

row_query, row_timespan = st.columns([0.75, 0.25])
with row_query:
  query_input = st.text_input(
      "Search expression",
      value=DEFAULT_QUERY,
      help="Any valid GDELT query string. Wrap multi-word terms in quotes (e.g. \"climate change\") and wrap OR clauses in parentheses, such as (\"Tesla\" OR \"SpaceX\").",
  )
with row_timespan:
  timespan_input = st.text_input(
      "Timespan",
      value="52weeks",
      help="Relative lookback window (e.g. `52weeks`, `104weeks`, `24months`, or `365` for days).",
  )

load_col, reset_col = st.columns([1, 1])
with load_col:
  load_clicked = st.button("Load time series", type="primary", use_container_width=True)
with reset_col:
  reset_clicked = st.button("Reset", type="secondary", use_container_width=True)

if reset_clicked:
  st.session_state.timeline_state = None
  st.experimental_rerun()

if load_clicked:
  try:
    with st.spinner("Loading GDELT timeline..."):
      timeline_data = _load_timeline(
          query_input,
          timespan=timespan_input,
      )
  except SystemExit as exc:
    st.error(str(exc))
  except Exception as exc:  # pragma: no cover
    st.error(f"Failed to load timeline: {exc}")
  else:
    st.session_state.timeline_state = {
        "query": query_input,
        "timespan": timespan_input,
        **timeline_data,
    }

timeline_state = st.session_state.timeline_state

if timeline_state is not None:
  raw_values = timeline_state["arrays"][0]
  raw_times = timeline_state["timestamps"][0]
  if len(timeline_state["arrays"]) > 1:
    st.warning("Multiple timelines detected; defaulting to the first series.")
  st.plotly_chart(
      _timeline_figure(raw_times, raw_values, query=timeline_state["query"]),
      use_container_width=True,
  )
else:
  st.info("Load a timeline to view the coverage series before forecasting.")

st.markdown("---")

st.markdown("### Step 2: Forecast into the future")
st.write(
    "Choose the models to run, adjust the forecast horizon (capped at 18 weeks), and optionally limit the "
    "historical context. All forecasts share a 35-day backtest window so MAPEs remain comparable."
)

col_models, col_horizon, col_context, col_smooth = st.columns([2.4, 1.1, 1.1, 1.1])
with col_models:
  models_selected = st.multiselect(
      "Models",
      options=list(MODEL_OPTIONS.keys()),
      default=[MODEL_LABELS["timesfm"], MODEL_LABELS["prophet"]],
      help="Select one or more forecasting backends to compare (TimesFM, Prophet, ARIMA, Exponential Smoothing).",
  )
with col_horizon:
  horizon_input = st.text_input(
      "Forecast horizon",
      value="18weeks",
      help="How far ahead to forecast. Examples: `18weeks`, `126` (days). Forecasts are capped at 18 weeks to keep results reliable.",
  )
with col_context:
  max_context_input = st.text_input(
      "Max context",
      value="",
      help="Optional limit on how much history the models see (e.g. `52weeks`). Leave blank to use the entire timespan you downloaded.",
  )
with col_smooth:
  smooth_choice = st.selectbox(
      "Smoothing",
      options=["None"] + sorted(SMOOTH_WINDOWS),
      index=0,
      help="Apply a trailing moving average before forecasting (e.g. `weekly` = 7-day smooth). This can reduce noise before modeling.",
  )

run_forecast = st.button("Run forecast", type="primary", use_container_width=True)

if run_forecast:
  if timeline_state is None:
    st.warning("Load a timeline before running the forecast.")
  elif not models_selected:
    st.warning("Select at least one forecasting model.")
  else:
    model_keys = [MODEL_OPTIONS[label] for label in models_selected]
    smoothing_option = None if smooth_choice == "None" else smooth_choice
    try:
      auto_max_context = max_context_input or timespan_input
      with st.spinner("Running forecasts..."):
        ts_context, bundles, extra = generate_forecast_artifacts(
            query=timeline_state["query"],
            timespan=timeline_state["timespan"],
            timeline_smooth=None,
            series_index=0,
            models=model_keys,
            max_context=auto_max_context,
            eval_window="35days",
            horizon=horizon_input or None,
            smooth=smoothing_option,
            max_horizon_days=18 * 7,
            verbose=False,
        )
    except SystemExit as exc:
      st.error(str(exc))
    except Exception as exc:  # pragma: no cover
      st.error(f"Forecast failed: {exc}")
    else:
      fig = build_forecast_figure(
          ts_context,
          bundles,
          raw_times=extra["raw_times"],
          raw_values=extra["raw_values"],
          smoothed=extra.get("smoothed", False),
          search_query=timeline_state["query"],
      )
      st.plotly_chart(fig, use_container_width=True)

with st.expander("Implementation details & methodology"):
  st.markdown(
      """
      **Data sourcing** – For every query the app downloads the GDELT `TimelineVol` signal at daily cadence, then groups
      intraday points into a single value per day. You can optionally limit the lookback window via *Timespan* or cap the
      usable history with *Max context*.

      **Two-step evaluation** – Before forecasting, we reserve the most recent 35 days as a backtest window. Each model is
      fit on the remaining history (*context*) and we compute MAPE on that 35-day holdout. The information shown on the
      chart (green line + MAPE badge) always reflects this consistent evaluation window.

      **Context vs. training data** – [TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
      is an AI foundation model for time series forecasting. It uses "context" to describe the historical chunk provided at
      inference time. You can think of it as the training window; Prophet, ARIMA, and Exponential Smoothing all operate on that
      same context so comparisons stay apples-to-apples.

      **Forecast horizon** – We cap forecasts at 18 weeks (126 days) to stay within the range where TimesFM and the
      classical models remain stable. Any smoother you choose (weekly or monthly) is applied *before* evaluation and
      forecasting, so the holdout and future projections share the same transformed series.

      **Query normalization** – The app auto-quotes multi-word phrases and uppercases boolean operators, and will wrap OR
      clauses in parentheses when needed. You can type expressions such as `Tesla OR SpaceX` or
      `("climate change" AND policy)` directly and the request sent to GDELT will be valid.
      """
  )

st.caption("Built with TimesFM, Prophet, ARIMA, and Exponential Smoothing wrappers.")
