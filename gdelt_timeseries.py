"""Utilities for pulling GDELT timeline data and formatting it for TimesFM."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

from forecasting import (
    ForecastBundle,
    ForecastResult,
    TimeSeriesContext,
    forecast_timesfm,
)

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DATE_FMT = "%Y%m%dT%H%M%SZ"
SMOOTH_WINDOWS = {"weekly": 7, "monthly": 30}
MAX_HORIZON_DAYS = 5 * 365
DEFAULT_EVAL_DAYS = 35


def _resolve_units(smooth: Optional[str]) -> Tuple[str, int]:
  """Returns the display unit label and conversion to days."""

  if smooth == "weekly":
    return "weeks", 7
  if smooth == "monthly":
    return "months", 30
  return "days", 1


_UNIT_SCALES = {"days": 1, "weeks": 7, "months": 30}
_UNIT_ALIASES = {
    "d": "days",
    "day": "days",
    "days": "days",
    "w": "weeks",
    "week": "weeks",
    "weeks": "weeks",
    "m": "months",
    "month": "months",
    "months": "months",
}


@dataclass(frozen=True)
class UnitQuantity:
  magnitude: int
  unit_label: str
  days: int


ForecastRunner = Callable[[TimeSeriesContext, argparse.Namespace], ForecastResult]


def _forecast_with_timesfm(context: TimeSeriesContext, args: argparse.Namespace) -> ForecastResult:
  max_context = context.metadata.get("max_context") if context.metadata else None
  return forecast_timesfm(
      context.context_values,
      context.forecast_horizon,
      max_context=max_context,
      timesfm_root=args.timesfm_root,
  )


def _forecast_with_prophet(context: TimeSeriesContext, args: argparse.Namespace) -> ForecastResult:
  from forecasting import forecast_prophet

  del args  # Currently unused
  return forecast_prophet(
      context.context_values,
      context.context_timestamps,
      context.forecast_horizon,
      max_context=context.metadata.get("max_context") if context.metadata else None,
  )


FORECAST_MODEL_REGISTRY: Dict[str, ForecastRunner] = {
    "timesfm": _forecast_with_timesfm,
    "prophet": _forecast_with_prophet,
}

MODEL_LABELS = {
    "timesfm": "TimesFM",
    "prophet": "Prophet",
}


def _normalize_unit(token: str) -> Optional[str]:
  normalized = _UNIT_ALIASES.get(token.lower(), token.lower())
  return normalized if normalized in _UNIT_SCALES else None


def _parse_quantity_arg(
    raw_value: Optional[str],
    *,
    default_magnitude: Optional[int],
    default_unit: str,
    param_name: str,
) -> Optional[UnitQuantity]:
  """Parses values like '26weeks' into a normalized quantity in days."""

  if raw_value is None:
    if default_magnitude is None:
      return None
    scale = _UNIT_SCALES[default_unit]
    return UnitQuantity(default_magnitude, default_unit, default_magnitude * scale)

  raw_text = str(raw_value).strip().lower()
  match = re.fullmatch(r"(\d+)\s*(?:([a-z]+))?", raw_text)
  if not match:
    raise SystemExit(
        f"{param_name} must be an integer optionally followed by a unit (days/weeks/months), e.g. '26weeks'."
    )

  magnitude = int(match.group(1))
  if magnitude <= 0:
    raise SystemExit(f"{param_name} must be positive.")

  unit_token = match.group(2)
  unit_label = _normalize_unit(unit_token) if unit_token else default_unit
  if unit_label is None:
    raise SystemExit(
        f"Unrecognized unit '{unit_token}' for {param_name}; expected days, weeks, or months."
    )

  scale = _UNIT_SCALES[unit_label]
  return UnitQuantity(magnitude, unit_label, magnitude * scale)


def fetch_timeline_vol(
    query: str,
    *,
    mode: str = "TimelineVol",
    timespan: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    timeline_smooth: Optional[int] = None,
    format_: str = "json",
    extra_params: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Dict:
  """Fetches a TimelineVol-style response from the GDELT DOC 2.0 API."""
  params: Dict[str, str] = {"query": query, "mode": mode, "format": format_}
  if timespan:
    params["timespan"] = timespan
  if start_datetime:
    params["startdatetime"] = start_datetime
  if end_datetime:
    params["enddatetime"] = end_datetime
  if timeline_smooth is not None:
    params["timelinesmooth"] = str(timeline_smooth)
  if extra_params:
    params.update(extra_params)

  url = f"{BASE_URL}?{urlencode(params)}"
  try:
    with urlopen(url, timeout=timeout) as resp:
      payload = resp.read()
  except HTTPError as exc:
    raise RuntimeError(f"GDELT request failed with HTTP {exc.code}: {exc.reason}") from exc
  except URLError as exc:
    raise RuntimeError(f"Unable to reach GDELT API: {exc.reason}") from exc

  if format_.lower() != "json":
    raise ValueError("This helper expects format json for easier downstream parsing.")

  try:
    return json.loads(payload.decode("utf-8"))
  except json.JSONDecodeError as exc:
    raise RuntimeError("Failed to decode JSON payload from GDELT.") from exc


def timeline_to_daily_arrays(timeline_payload: Dict) -> Tuple[List[np.ndarray], List[List[datetime]]]:
  """Extracts per-day values from a GDELT timeline payload."""

  arrays: List[np.ndarray] = []
  timestamps: List[List[datetime]] = []
  series_list = timeline_payload.get("timeline", [])
  if not series_list:
    return arrays, timestamps

  for series in series_list:
    day_buckets: Dict[datetime, List[float]] = {}
    for entry in series.get("data", []):
      try:
        value = float(entry["value"])
        ts = datetime.strptime(entry["date"], DATE_FMT)
      except (KeyError, TypeError, ValueError):
        continue
      day = datetime(ts.year, ts.month, ts.day)
      day_buckets.setdefault(day, []).append(value)

    if not day_buckets:
      continue

    ordered_days = sorted(day_buckets)
    day_values = [float(np.mean(day_buckets[d])) for d in ordered_days]
    arrays.append(np.asarray(day_values, dtype=np.float32))
    timestamps.append(ordered_days)

  return arrays, timestamps


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
  """Simple trailing moving average with partial windows at the start."""

  if window <= 1:
    return values.astype(np.float32, copy=True)
  result = np.empty_like(values, dtype=np.float32)
  cumsum = np.cumsum(values, dtype=np.float64)
  for idx in range(len(values)):
    start = max(0, idx - window + 1)
    total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
    count = idx - start + 1
    result[idx] = float(total / count)
  return result


def render_forecast_panels(
    ts_context: TimeSeriesContext,
    bundles: Sequence[ForecastBundle],
    chart_path: str,
    *,
    raw_times: Optional[List[datetime]] = None,
    raw_values: Optional[np.ndarray] = None,
    smoothed: bool = False,
    search_query: Optional[str] = None,
) -> None:
  if not bundles:
    raise ValueError("At least one forecast bundle is required for rendering.")

  try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
  except ImportError as exc:  # pragma: no cover
    raise SystemExit("plotly is required for plotting; install with `pip install plotly`."
                     ) from exc

  num_rows = len(bundles)
  subplot_titles = [bundle.model_label for bundle in bundles]

  history_times = list(ts_context.context_timestamps)
  history_values = ts_context.context_values
  holdout_times = list(ts_context.holdout_timestamps)
  holdout_actual = ts_context.holdout_values
  observed_times = list(ts_context.model_timestamps)
  delta = ts_context.timestamp_cadence

  fig = make_subplots(
      rows=num_rows,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.09 if num_rows > 1 else 0.04,
      subplot_titles=subplot_titles,
  )

  for row_index, bundle in enumerate(bundles, start=1):
    showlegend = row_index == 1

    if not history_times:
      raise ValueError("History times must be non-empty for plotting.")
    if delta <= timedelta(0):
      raise ValueError("Non-positive timestep detected in timeline.")

    future_times: List[datetime] = []
    last_time = observed_times[-1]
    for _ in range(len(bundle.point_forecast)):
      last_time = last_time + delta
      future_times.append(last_time)

    if raw_times is not None and raw_values is not None:
      fig.add_trace(
          go.Scatter(
              x=raw_times,
              y=raw_values,
              mode="lines",
              name="daily (raw)",
              line=dict(color="rgba(128,128,128,0.6)" if smoothed else "rgba(128,128,128,0.75)", width=1.0),
              hoverinfo="x+y",
              legendgroup="raw",
              showlegend=showlegend,
          ),
          row=row_index,
          col=1,
      )

    context_label = "context (smoothed)" if smoothed else "context"
    fig.add_trace(
        go.Scatter(
            x=history_times,
            y=history_values,
            mode="lines",
            name=context_label,
            line=dict(color="#1f77b4", width=2.0),
            legendgroup="context",
            showlegend=showlegend,
        ),
        row=row_index,
        col=1,
    )

    if holdout_times:
      holdout_label = "holdout actual (smoothed)" if smoothed else "holdout actual"
      fig.add_trace(
          go.Scatter(
              x=holdout_times,
              y=holdout_actual,
              mode="lines+markers",
              name=holdout_label,
              line=dict(color="#2ca02c", width=1.5),
              marker=dict(color="#2ca02c", size=6, line=dict(color="white", width=0.5)),
              legendgroup="holdout",
              showlegend=showlegend,
          ),
          row=row_index,
          col=1,
      )

      if bundle.backtest_forecast.size and len(bundle.backtest_forecast) == len(holdout_times):
        fig.add_trace(
            go.Scatter(
                x=holdout_times,
                y=bundle.backtest_forecast,
                mode="lines+markers",
                name="backtest forecast",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                marker=dict(color="#ff7f0e", size=6, line=dict(color="white", width=0.5)),
                legendgroup="backtest",
                showlegend=showlegend,
            ),
            row=row_index,
            col=1,
        )

    if bundle.quantile_forecast is not None and bundle.quantile_forecast.shape[0] >= len(future_times):
      try:
        lower_10 = bundle.quantile_forecast[: len(future_times), 1]
        upper_90 = bundle.quantile_forecast[: len(future_times), 9]
      except IndexError:
        lower_10 = upper_90 = None
      if lower_10 is not None and upper_90 is not None:
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=upper_90,
                mode="lines",
                line=dict(color="rgba(255,127,14,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row_index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=lower_10,
                mode="lines",
                line=dict(color="rgba(255,127,14,0)"),
                fill="tonexty",
                fillcolor="rgba(255,127,14,0.18)",
                name="10%-90% interval",
                hoverinfo="skip",
                legendgroup="interval",
                showlegend=showlegend,
            ),
            row=row_index,
            col=1,
        )

    if bundle.point_forecast.size:
      fig.add_trace(
          go.Scatter(
              x=future_times,
              y=bundle.point_forecast,
              mode="lines",
              name="forecast",
              line=dict(color="#ff7f0e", width=2.0),
              legendgroup="forecast",
              showlegend=showlegend,
          ),
          row=row_index,
          col=1,
      )

    mape = bundle.metadata.get("mape") if bundle.metadata else None
    if mape is not None and not math.isnan(mape):
      xref = "x domain" if row_index == 1 else f"x{row_index} domain"
      yref = "y domain" if row_index == 1 else f"y{row_index} domain"
      fig.add_annotation(
          text=f"Backtest MAPE: {mape:.2f}%",
          xref=xref,
          yref=yref,
          x=0.98,
          y=0.98,
          showarrow=False,
          font=dict(size=12),
          align="right",
          bgcolor="rgba(255,255,255,0.85)",
          bordercolor="rgba(0,0,0,0.15)",
          borderwidth=1,
          borderpad=4,
      )

  fig.update_layout(
      template="simple_white",
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=-0.22,
          x=0.5,
          xanchor="center",
          font=dict(size=11),
          bgcolor="rgba(255,255,255,0.9)",
          bordercolor="rgba(0,0,0,0.15)",
          borderwidth=1,
      ),
      margin=dict(l=50, r=20, t=90, b=170),
      hovermode="x unified",
      font=dict(family="Helvetica, Arial, sans-serif", size=12, color="#222222"),
      plot_bgcolor="white",
      paper_bgcolor="white",
      height=max(360, 360 * num_rows),
  )

  fig.update_yaxes(
      title_text="Volume intensity",
      showgrid=True,
      gridcolor="rgba(0,0,0,0.08)",
      zeroline=False,
  )

  subtitle_parts = ["GDELT coverage volume forecasts"]
  if search_query:
    subtitle_parts.append(f'query: "{search_query}"')
  subtitle_text = " — ".join(subtitle_parts)
  fig.add_annotation(
      text=subtitle_text,
      xref="paper",
      yref="paper",
      x=0.5,
      y=1.08,
      yanchor="bottom",
      showarrow=False,
      font=dict(size=16, color="#222222", family="Helvetica, Arial, sans-serif"),
      align="center",
      bgcolor="rgba(255,255,255,0.95)",
      bordercolor="rgba(0,0,0,0.15)",
      borderwidth=1,
      borderpad=6,
  )

  output_path = Path(chart_path)
  suffix = output_path.suffix.lower()
  try:
    if suffix in {".html", ".htm"}:
      fig.write_html(str(output_path), include_plotlyjs="cdn")
    else:
      fig.write_image(str(output_path), scale=2)
  except (ValueError, ImportError) as exc:
    fallback = output_path.with_suffix(output_path.suffix + ".html" if suffix else ".html")
    fig.write_html(str(fallback), include_plotlyjs="cdn")
    print(
        f"Plotly static export failed ({exc}). Saved interactive HTML to {fallback}",
        file=sys.stderr,
    )
  else:
    print(f"Saved chart to {chart_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = argparse.ArgumentParser(description="Pull GDELT TimelineVol data ready for TimesFM.")
  parser.add_argument("query", help="Search expression to send to the GDELT DOC 2.0 API.")
  parser.add_argument("--mode", default="TimelineVol", help="GDELT mode, defaults to TimelineVol.")
  parser.add_argument(
      "--timespan",
      default="52weeks",
      help="Relative time window like 52weeks, 104weeks, 24months. Mutually exclusive with start/end datetimes.",
  )
  parser.add_argument("--start-datetime", dest="start_datetime", help="Explicit start datetime in YYYYMMDDHHMMSS format.")
  parser.add_argument("--end-datetime", dest="end_datetime", help="Explicit end datetime in YYYYMMDDHHMMSS format.")
  parser.add_argument(
      "--timelinesmooth",
      type=int,
      help="Moving window smoothing span (1-30) per GDELT TIMELINESMOOTH parameter.",
  )
  parser.add_argument(
      "--timeout",
      type=float,
      default=30.0,
      help="HTTP timeout in seconds (default: 30).",
  )
  parser.add_argument(
      "--extra",
      nargs="*",
      help="Optional key=value pairs forwarded to the API for advanced configuration.",
  )
  parser.add_argument("--series-index", type=int, default=0, help="Which timeline series to use (default: 0).")
  parser.add_argument(
      "--models",
      nargs="+",
      default=["timesfm"],
      choices=sorted(FORECAST_MODEL_REGISTRY),
      help="Forecasting backends to run (choose timesfm, prophet; default: timesfm).",
  )
  parser.add_argument(
      "--max-context",
      type=str,
      metavar="N[unit]",
      help="Maximum context length (e.g. 180, 52weeks, 6months). Defaults to all available history.",
  )
  parser.add_argument(
      "--eval-window",
      type=str,
      default=None,
      metavar="N[unit]",
      help="Length of the recent backtest window used for MAPE (default: 35days).",
  )
  parser.add_argument(
      "--horizon",
      type=str,
      default=None,
      metavar="N[unit]",
      help=(
          "Forecast horizon (e.g. 182, 26weeks, 6months). Bare numbers respect the active cadence "
          "(days unless smoothing switches it). Defaults to 182 in that cadence."
      ),
  )
  parser.add_argument("--chart-path", default="gdelt_timesfm_forecast.png", help="Output image path for the plot.")
  parser.add_argument(
      "--skip-forecast",
      action="store_true",
      help="Only display fetched series without running TimesFM or plotting.",
  )
  parser.add_argument(
      "--timesfm-root",
      help="Optional path to a TimesFM source tree if not installed.",
  )
  parser.add_argument(
      "--smooth",
      choices=sorted(SMOOTH_WINDOWS),
      help="Apply a trailing moving average before forecasting (choose weekly=7 days or monthly=30 days).",
  )

  args = parser.parse_args(argv)

  unit_label, _ = _resolve_units(args.smooth)
  horizon_spec = _parse_quantity_arg(
      args.horizon,
      default_magnitude=182,
      default_unit=unit_label,
      param_name="horizon",
  )
  assert horizon_spec is not None  # for type checkers
  horizon_days = horizon_spec.days
  horizon_unit_scale = _UNIT_SCALES[horizon_spec.unit_label]
  if horizon_days < 20:
    min_units = max(1, math.ceil(20 / horizon_unit_scale))
    raise SystemExit(
        f"horizon must be at least {min_units} {horizon_spec.unit_label} to satisfy a 4-point holdout (35% rule)."
    )
  if horizon_days > MAX_HORIZON_DAYS:
    raise SystemExit(
        f"horizon is capped at {MAX_HORIZON_DAYS} days (~5 years); requested {horizon_spec.magnitude} {horizon_spec.unit_label}."
    )

  eval_spec = _parse_quantity_arg(
      args.eval_window,
      default_magnitude=DEFAULT_EVAL_DAYS,
      default_unit="days",
      param_name="eval-window",
  )

  context_spec = _parse_quantity_arg(
      args.max_context,
      default_magnitude=None,
      default_unit=unit_label,
      param_name="max-context",
  )

  extra_params: Dict[str, str] = {}
  if args.extra:
    for kv in args.extra:
      if "=" not in kv:
        parser.error(f"Invalid extra parameter '{kv}'. Expected key=value.")
      key, value = kv.split("=", 1)
      extra_params[key] = value

  payload = fetch_timeline_vol(
      query=args.query,
      mode=args.mode,
      timespan=args.timespan,
      start_datetime=args.start_datetime,
      end_datetime=args.end_datetime,
      timeline_smooth=args.timelinesmooth,
      extra_params=extra_params if extra_params else None,
      timeout=args.timeout,
  )

  arrays, timestamps = timeline_to_daily_arrays(payload)
  if not arrays:
    print("No timeline data returned for the given query.", file=sys.stderr)
    return 1

  print("Prepared TimesFM inputs (daily cadence):")
  for idx, arr in enumerate(arrays):
    print(f"Series {idx}: shape={arr.shape}, dtype={arr.dtype}")
  print(arrays)

  print("\nCorresponding timestamps:")
  for idx, times in enumerate(timestamps):
    start = times[0]
    end = times[-1]
    print(f"Series {idx}: {len(times)} points from {start.isoformat()} to {end.isoformat()}")

  if args.skip_forecast:
    return 0

  if args.series_index >= len(arrays) or args.series_index < 0:
    raise SystemExit("series index out of range")

  raw_values_all = arrays[args.series_index].astype(np.float32, copy=True)
  raw_times_all = list(timestamps[args.series_index])

  selected_values = raw_values_all
  selected_times = raw_times_all

  context_limit: Optional[int] = None
  if context_spec is not None:
    context_limit = context_spec.days
    if len(selected_values) > context_limit:
      print(
          f"Trimming context to last {context_limit} days (~{context_spec.magnitude} {context_spec.unit_label}).",
          file=sys.stderr,
      )
      selected_values = selected_values[-context_limit :]
      selected_times = selected_times[-context_limit :]

  full_raw_values = raw_values_all
  full_raw_times = raw_times_all

  smoothing_window = SMOOTH_WINDOWS.get(args.smooth) if args.smooth else None
  if smoothing_window:
    if len(selected_values) < smoothing_window:
      print(
          f"Warning: smoothing window ({smoothing_window} days) exceeds available series length ({len(selected_values)}); partial averages will be used.",
          file=sys.stderr,
      )
    selected_values = moving_average(selected_values, smoothing_window)
    print(f"\nApplied {args.smooth} smoothing (window {smoothing_window} days) before forecasting.")

  num_points = len(selected_values)
  if num_points < 5:
    raise SystemExit(
        "Need at least 5 data points after preprocessing to run the evaluation backtest and future forecast."
    )

  max_holdout = max(0, num_points - 4)
  if max_holdout <= 0:
    raise SystemExit(
        "Not enough data to reserve an evaluation window while preserving at least 4 context points."
    )

  requested_holdout = eval_spec.days
  if requested_holdout > max_holdout:
    print(
        f"Evaluation window ({requested_holdout} days) truncated to {max_holdout} days to preserve minimum context.",
        file=sys.stderr,
    )
    holdout_len = max_holdout
  else:
    holdout_len = requested_holdout

  if holdout_len < 1:
    raise SystemExit("Evaluation window collapsed; provide more data or reduce --eval-window.")

  backtest_context_values = selected_values[:-holdout_len]
  backtest_context_times = selected_times[:-holdout_len]
  holdout_actual = selected_values[-holdout_len:]
  holdout_times = selected_times[-holdout_len:]
  context_len = len(backtest_context_values)
  if context_len < 4:
    raise SystemExit(
        f"Not enough context ({context_len}) after reserving the evaluation window; need at least 4 points."
    )
  if context_len < 10:
    print(
        f"Warning: only {context_len} context points available for the backtest; forecasts may be unstable.",
        file=sys.stderr,
    )

  if horizon_days > context_len * 3:
    print(
        f"Warning: horizon spans {horizon_days} days (~{horizon_spec.magnitude} {horizon_spec.unit_label}), exceeding 3x the backtest context length ({context_len} days).",
        file=sys.stderr,
    )

  if len(selected_times) >= 2:
    timestamp_cadence = selected_times[1] - selected_times[0]
  elif len(backtest_context_times) >= 2:
    timestamp_cadence = backtest_context_times[1] - backtest_context_times[0]
  else:
    raise SystemExit("Unable to infer timestamp cadence for the selected series.")
  if timestamp_cadence <= timedelta(0):
    raise SystemExit("Non-positive timestamp cadence detected; cannot proceed.")

  context_metadata: Dict[str, object] = {}
  if context_limit is not None:
    context_metadata["max_context"] = context_limit

  ts_context = TimeSeriesContext(
      series_id=args.query,
      timestamp_cadence=timestamp_cadence,
      full_history=full_raw_values,
      full_timestamps=full_raw_times,
      context_values=backtest_context_values,
      context_timestamps=backtest_context_times,
      model_values=selected_values,
      model_timestamps=selected_times,
      holdout_values=holdout_actual,
      holdout_timestamps=holdout_times,
      forecast_horizon=horizon_days,
      metadata=context_metadata,
  )

  bundles: List[ForecastBundle] = []
  for model_key in args.models:
    runner = FORECAST_MODEL_REGISTRY[model_key]
    model_label = MODEL_LABELS.get(model_key, model_key)

    backtest_context = ts_context.with_context_override(
        values=ts_context.context_values,
        timestamps=ts_context.context_timestamps,
        forecast_horizon=holdout_len,
    )
    backtest_result = runner(backtest_context, args)
    if backtest_result.point_forecast.shape[0] < holdout_len:
      raise SystemExit(
          f"{model_label} returned fewer steps than requested for the backtest window; cannot score MAPE."
      )
    backtest_forecast = backtest_result.point_forecast[:holdout_len]

    with np.errstate(divide="ignore", invalid="ignore"):
      denom = np.abs(ts_context.holdout_values)
      mask = denom > 1e-6
      if np.any(mask):
        mape = float(
            np.mean(
                np.abs((ts_context.holdout_values[mask] - backtest_forecast[mask]) / denom[mask])
            )
            * 100.0
        )
      else:
        mape = float("nan")

    future_context = ts_context.with_context_override(
        values=ts_context.model_values,
        timestamps=ts_context.model_timestamps,
        forecast_horizon=ts_context.forecast_horizon,
    )
    future_result = runner(future_context, args)

    future_point_forecast = future_result.point_forecast
    if future_point_forecast.size == 0:
      raise SystemExit(f"{model_label} returned an empty future forecast.")
    if future_point_forecast.shape[0] < horizon_days:
      print(
          f"Warning: {model_label} future forecast returned {future_point_forecast.shape[0]} steps (requested {horizon_days}); plotting uses the returned length.",
          file=sys.stderr,
      )

    future_quantile_forecast = future_result.quantile_forecast
    trimmed = False
    if future_quantile_forecast is not None:
      if future_quantile_forecast.ndim != 2 or future_quantile_forecast.shape[1] < 10:
        print(
            f"Warning: {model_label} quantiles missing 10th/90th percentiles; interval shading disabled.",
            file=sys.stderr,
        )
        future_quantile_forecast = None
      else:
        interval_cols = future_quantile_forecast[:, [1, 9]]
        finite_mask = np.isfinite(interval_cols).all(axis=1)
        if not np.any(finite_mask):
          print(
              f"Warning: {model_label} forecast quantile bounds were non-finite; dropping interval shading.",
              file=sys.stderr,
          )
          future_quantile_forecast = None
        elif not np.all(finite_mask):
          last_valid_idx = int(np.where(finite_mask)[0][-1])
          trim_len = last_valid_idx + 1
          future_point_forecast = future_point_forecast[:trim_len]
          future_quantile_forecast = future_quantile_forecast[:trim_len]
          trimmed = True
    if trimmed:
      print(
          f"Warning: {model_label} future forecast truncated to {len(future_point_forecast)} steps because quantile bounds were non-finite beyond that horizon.",
          file=sys.stderr,
      )

    future_quantile_shape = future_quantile_forecast.shape if future_quantile_forecast is not None else None
    print(
        f"\n[{model_label}] Backtest window: {holdout_len} days. Future forecast horizon: {len(future_point_forecast)} days (quantiles {future_quantile_shape})."
    )
    if not math.isnan(mape):
      print(f"[{model_label}] Backtest MAPE ({holdout_len} days): {mape:.3f}%")
    else:
      print(f"[{model_label}] Backtest MAPE: undefined (division by zero encountered).")

    residuals = ts_context.holdout_values - backtest_forecast
    bundle_metadata = {
        "mape": mape,
        "backtest_horizon": holdout_len,
        "future_horizon": len(future_point_forecast),
    }
    bundles.append(
        ForecastBundle(
            model_key=model_key,
            model_label=model_label,
            point_forecast=future_point_forecast.astype(np.float32, copy=True),
            quantile_forecast=
            future_quantile_forecast.astype(np.float32, copy=True)
            if future_quantile_forecast is not None
            else None,
            backtest_forecast=backtest_forecast.astype(np.float32, copy=True),
            residuals=residuals.astype(np.float32, copy=True),
            context_used=future_result.context_used,
            metadata=bundle_metadata,
        )
    )

  render_forecast_panels(
      ts_context,
      bundles,
      args.chart_path,
      raw_times=full_raw_times,
      raw_values=full_raw_values,
      smoothed=bool(smoothing_window),
      search_query=args.query,
  )

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
