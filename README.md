# TimeSeriesAnalytics

Utilities for fetching GDELT coverage timelines and benchmarking multiple forecasting backends (TimesFM, Prophet, Exponential Smoothing, ARIMA, …) on a shared Time Series Data Model (TSDM).

## Quick start

```bash
# Clone this repository
git clone <your-remote-url> TimeSeriesAnalytics
cd TimeSeriesAnalytics

# (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt  # (if you generate one)
pip install plotly prophet pandas numpy statsmodels

# Clone TimesFM (kept in a separate repo)
git clone git@github.com:sidscorp/timesfm-sn.git timesfm
```

## Fetch & forecast

```bash
python gdelt_timeseries.py "covid OR coronavirus"

# Run multiple models stacked in the output chart
python gdelt_timeseries.py "ai AND robotics" \
  --models timesfm prophet expsmooth arima \
  --timespan 104weeks \
  --max-context 52weeks \
  --horizon 26weeks \
  --chart-path outputs/ai_forecast.png
```

Key CLI options:
- `--timespan`, `--max-context`, `--horizon`, `--eval-window` accept numbers with unit suffixes (`26weeks`, `6months`, `180`).
- `--smooth` enables weekly/monthly moving averages before forecasting.
- `--models` lets you stack multiple backends (timesfm, prophet, expsmooth, arima); each panel reports its own backtest MAPE.

## Project layout

```
TimeSeriesAnalytics/
├── forecasting/            # TSDM dataclasses + model adapters (TimesFM, Prophet, ...)
├── gdelt_timeseries.py     # CLI entry point orchestration (fetch, evaluate, plot)
├── timesfm/                # cloned separately; ignored by this repo (see .gitignore)
├── Archive/                # personal scratch space (ignored)
└── .venv/                  # optional virtual environment (ignored)
```

## Workflow notes

- TimesFM lives in its own repository (`timesfm/`) so local tweaks can be committed and pushed separately. Clone your fork or add it as a submodule if you prefer.
- The main repository now standardizes data passing through a `TimeSeriesContext` (inputs) → `ForecastBundle` (outputs) flow, making it easy to drop in new forecasting adapters.
- Run `git status` before committing to keep TimesFM changes out of this repo; the root `.gitignore` already excludes the nested clone and local artifacts.
- For repeatable environment setup, consider adding a `requirements.txt` or `pyproject.toml` that captures Plotly, Prophet, pandas, numpy, etc.

## Roadmap

- Introduce a formal Time Series Data Model (TSDM) to standardize inputs/outputs across forecasting backends.
- Add more model adapters (ARIMA, AutoETS, etc.) once the TSDM is in place.
- Extend backtesting utilities for rolling windows and richer metrics.
