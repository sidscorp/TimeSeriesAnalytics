"""Shared forecasting datatypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ForecastResult:
  """Container for forecast outputs returned to orchestrators."""

  point_forecast: np.ndarray
  quantile_forecast: Optional[np.ndarray] = None
  context_used: Optional[int] = None
