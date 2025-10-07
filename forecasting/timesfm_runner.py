"""Adapter for running TimesFM forecasts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base import ForecastResult


def _import_timesfm(timesfm_root: Optional[str]) -> Tuple[object, object]:
  """Import TimesFM classes, optionally adjusting sys.path."""
  if timesfm_root:
    sys.path.append(str(Path(timesfm_root).expanduser().resolve()))
  local_repo = Path(__file__).resolve().parent.parent / "timesfm" / "src"
  if local_repo.is_dir() and str(local_repo) not in sys.path:
    sys.path.append(str(local_repo))
  try:
    import timesfm  # type: ignore
  except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("timesfm package not found. Install or supply --timesfm-root.") from exc
  return timesfm.TimesFM_2p5_200M_torch, timesfm.ForecastConfig


def forecast_timesfm(
    values: np.ndarray,
    horizon: int,
    *,
    max_context: Optional[int] = None,
    timesfm_root: Optional[str] = None,
) -> ForecastResult:
  """Run a TimesFM forecast for the provided sequence."""
  if len(values) == 0:
    raise ValueError("Context values must be non-empty for forecasting.")

  model_cls, forecast_cfg_cls = _import_timesfm(timesfm_root)
  context_len = len(values) if max_context is None else min(max_context, len(values))
  context_values = values[-context_len:]

  model = model_cls.from_pretrained("google/timesfm-2.5-200m-pytorch")
  config = forecast_cfg_cls(
      max_context=context_len,
      max_horizon=horizon,
      normalize_inputs=True,
      use_continuous_quantile_head=False,
      force_flip_invariance=True,
      infer_is_positive=False,
      fix_quantile_crossing=True,
  )
  model.compile(config)
  point_forecast, quantile_forecast = model.forecast(
      horizon=horizon,
      inputs=[context_values],
  )
  return ForecastResult(
      point_forecast=point_forecast[0],
      quantile_forecast=quantile_forecast[0],
      context_used=context_len,
  )
