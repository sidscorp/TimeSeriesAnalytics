"""Shared forecasting datatypes for model orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TimeSeriesContext:
  """Normalized series inputs passed to each forecasting backend."""

  series_id: str
  timestamp_cadence: timedelta
  full_history: np.ndarray
  full_timestamps: Sequence
  context_values: np.ndarray
  context_timestamps: Sequence
  model_values: np.ndarray
  model_timestamps: Sequence
  holdout_values: np.ndarray
  holdout_timestamps: Sequence
  forecast_horizon: int
  metadata: Dict[str, object] = field(default_factory=dict)

  def with_context_override(
      self,
      *,
      values: np.ndarray,
      timestamps: Sequence,
      forecast_horizon: Optional[int] = None,
      metadata: Optional[Dict[str, object]] = None,
  ) -> "TimeSeriesContext":
    return TimeSeriesContext(
        series_id=self.series_id,
        timestamp_cadence=self.timestamp_cadence,
        full_history=self.full_history,
        full_timestamps=self.full_timestamps,
        context_values=values,
        context_timestamps=timestamps,
        model_values=self.model_values,
        model_timestamps=self.model_timestamps,
        holdout_values=self.holdout_values,
        holdout_timestamps=self.holdout_timestamps,
        forecast_horizon=self.forecast_horizon if forecast_horizon is None else forecast_horizon,
        metadata=(self.metadata.copy() if metadata is None else metadata),
    )


@dataclass(frozen=True)
class ForecastBundle:
  """Standardized outputs produced by a forecasting backend."""

  model_key: str
  model_label: str
  point_forecast: np.ndarray
  quantile_forecast: Optional[np.ndarray]
  backtest_forecast: np.ndarray
  residuals: Optional[np.ndarray]
  context_used: Optional[int]
  metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastResult:
  """Legacy struct kept for backward compatibility with older runners."""

  point_forecast: np.ndarray
  quantile_forecast: Optional[np.ndarray] = None
  context_used: Optional[int] = None
