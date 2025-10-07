"""Adapter for running exponential smoothing forecasts."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ForecastResult


def _import_statsmodels():
  try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "statsmodels is required for exponential smoothing; install with `pip install statsmodels`."
    ) from exc
  return ExponentialSmoothing


def forecast_exponential_smoothing(
    values: np.ndarray,
    horizon: int,
    *,
    max_context: Optional[int] = None,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None,
    bootstrap_samples: int = 250,
    random_state: Optional[int] = None,
) -> ForecastResult:
  """Run Holt-Winters exponential smoothing for the provided sequence."""
  if len(values) == 0:
    raise ValueError("Context values must be non-empty for forecasting.")
  if horizon <= 0:
    raise ValueError("Horizon must be positive for forecasting.")

  context_len = len(values) if max_context is None else min(max_context, len(values))
  context_values = values[-context_len:]

  ExponentialSmoothing = _import_statsmodels()
  model = ExponentialSmoothing(
      context_values,
      trend=trend,
      seasonal=seasonal,
      seasonal_periods=seasonal_periods,
      initialization_method="estimated",
  )
  fit = model.fit()
  forecast = fit.forecast(horizon)
  point_forecast = np.asarray(forecast, dtype=np.float32)

  quantile_forecast = None
  residuals = getattr(fit, "resid", None)
  if bootstrap_samples and residuals is not None:
    residuals = np.asarray(residuals, dtype=np.float32)
    residuals = residuals[~np.isnan(residuals)]
    if residuals.size > 0:
      residuals = residuals - np.mean(residuals)
      rng = np.random.default_rng(random_state)
      draws = rng.choice(residuals, size=(bootstrap_samples, horizon), replace=True)
      simulations = point_forecast + draws
      lower = np.quantile(simulations, 0.10, axis=0).astype(np.float32)
      upper = np.quantile(simulations, 0.90, axis=0).astype(np.float32)
      quantile_matrix = np.full((len(point_forecast), 10), np.nan, dtype=np.float32)
      quantile_matrix[:, 1] = lower
      quantile_matrix[:, 9] = upper
      quantile_forecast = quantile_matrix

  return ForecastResult(
      point_forecast=point_forecast,
      quantile_forecast=quantile_forecast,
      context_used=context_len,
  )
