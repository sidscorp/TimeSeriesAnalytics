"""Adapter for running ARIMA forecasts."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import ForecastResult


def _import_statsmodels():
  try:
    from statsmodels.tsa.arima.model import ARIMA
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "statsmodels is required for ARIMA forecasting; install with `pip install statsmodels`."
    ) from exc
  return ARIMA


def forecast_arima(
    values: np.ndarray,
    horizon: int,
    *,
    max_context: Optional[int] = None,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    alpha: float = 0.2,
) -> ForecastResult:
  """Run an ARIMA forecast for the provided sequence."""
  if len(values) == 0:
    raise ValueError("Context values must be non-empty for forecasting.")
  if horizon <= 0:
    raise ValueError("Horizon must be positive for forecasting.")

  context_len = len(values) if max_context is None else min(max_context, len(values))
  context_values = values[-context_len:]

  ARIMA = _import_statsmodels()
  model = ARIMA(context_values, order=order, seasonal_order=seasonal_order)
  fit = model.fit()
  forecast_res = fit.get_forecast(steps=horizon)
  point_forecast = forecast_res.predicted_mean.astype(np.float32)

  quantile_forecast = None
  try:
    conf_int = forecast_res.conf_int(alpha=alpha)
    conf_array = np.asarray(conf_int, dtype=np.float32)
    if conf_array.ndim == 2 and conf_array.shape[1] >= 2:
      lower = conf_array[:, 0]
      upper = conf_array[:, 1]
      quantile_matrix = np.full((len(point_forecast), 10), np.nan, dtype=np.float32)
      quantile_matrix[:, 1] = lower
      quantile_matrix[:, 9] = upper
      quantile_forecast = quantile_matrix
  except Exception:  # pragma: no cover - fallback if intervals unavailable
    quantile_forecast = None

  return ForecastResult(
      point_forecast=point_forecast,
      quantile_forecast=quantile_forecast,
      context_used=context_len,
  )
