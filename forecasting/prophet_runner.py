"""Adapter for running Prophet forecasts."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import ForecastResult


def _import_prophet():
  try:
    from prophet import Prophet
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "prophet package not found. Install with `pip install prophet` to enable the Prophet backend."
    ) from exc
  return Prophet


def _prepare_history(times: Sequence, values: np.ndarray):
  try:
    import pandas as pd
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required for Prophet forecasts; install with `pip install pandas`."
    ) from exc

  history = pd.DataFrame({"ds": list(times), "y": values})
  return history


def forecast_prophet(
    values: np.ndarray,
    times: Sequence,
    horizon: int,
    *,
    max_context: Optional[int] = None,
) -> ForecastResult:
  """Run a Prophet forecast for the provided sequence."""
  if len(values) == 0:
    raise ValueError("Context values must be non-empty for forecasting.")
  if len(times) != len(values):
    raise ValueError("Times and values must align for Prophet forecasts.")
  if horizon <= 0:
    raise ValueError("Horizon must be positive for Prophet forecasts.")

  context_len = len(values) if max_context is None else min(max_context, len(values))
  context_values = values[-context_len:]
  context_times = times[-context_len:]

  Prophet = _import_prophet()
  history_df = _prepare_history(context_times, context_values)

  model = Prophet(interval_width=0.8)
  model.fit(history_df)

  future_df = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
  forecast_df = model.predict(future_df)
  point_forecast = forecast_df["yhat"].to_numpy(dtype=np.float32)

  lower = forecast_df.get("yhat_lower")
  upper = forecast_df.get("yhat_upper")
  if lower is not None and upper is not None:
    quantiles = np.full((len(point_forecast), 10), np.nan, dtype=np.float32)
    quantiles[:, 1] = lower.to_numpy(dtype=np.float32)
    quantiles[:, 9] = upper.to_numpy(dtype=np.float32)
  else:
    quantiles = None

  return ForecastResult(
      point_forecast=point_forecast,
      quantile_forecast=quantiles,
      context_used=context_len,
  )
