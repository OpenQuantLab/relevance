# relevance

Relevance-based statistical prediction ("Prediction Revisited") applied to
cross-sectional equity alpha signals.

## Alpha Lab

Interactive workbench for designing a signal, walking it forward and reading
the diagnostics.

```bash
pip install -r requirements.txt
python alpha_lab_app.py          # http://127.0.0.1:8051
```

In the sidebar you pick a universe and date range, tick the features that make
up the signal, choose how they are combined (the relevance model, or a linear
composite with your own weights), set the horizon, rebalance frequency and
target, then press **Run test**. The backtest runs in a worker thread with a
live progress bar; results land in the dashboard tabs:

| Tab | Contents |
| --- | --- |
| Overview | Headline metrics, cumulative long/short vs benchmark, drawdown, rolling Sharpe |
| Signal quality | Bin profile, predicted-rank vs realised-return scatter, IC time series |
| Stability | Period heatmap, year-by-year table, permutation-test null |
| Predictions | Latest cross-section and the full prediction history |
| Compare runs | Every saved run in one table, with overlaid equity curves |

Set *Sweep thresholds* to run a small parameter grid in one go. Every run is
saved under `lab/runs/<id>/` (spec, predictions, summary) and prices are cached
in `lab/cache/`, so reruns are instant.

`alpha_lab.py` holds the engine — the feature library, `SignalSpec`,
`build_panel`, `run_backtest` and `evaluate` — and is usable on its own from a
script or notebook.

Because jobs run in-process, serve the app with a single worker
(`gunicorn -w 1 --threads 8 alpha_lab_app:server`).

## Sharing it with other people

`deploy/DEPLOY.md` has copy-paste instructions for publishing Alpha Lab at
`datavik.no/projects/alpha-lab/` next to the existing app. Visitors get a
read-only dashboard — every saved run, chart and table is browsable, but the
Run button only unlocks for someone who types the password.

| Environment variable | Effect |
| --- | --- |
| `ALPHA_LAB_PASSWORD` | Turns on read-only mode; this password unlocks running |
| `ALPHA_LAB_URL_PREFIX` | Path the app is served under, e.g. `/projects/alpha-lab/` |
| `ALPHA_LAB_HOST` / `ALPHA_LAB_PORT` | Dev-server bind address and port |
| `ALPHA_LAB_DEBUG` | `0` turns off the dev-server debug tools |
| `EODHD_API_TOKEN` | Market-data key, overrides the default in `eodhd_data.py` |

Leave `ALPHA_LAB_PASSWORD` unset locally and everything is unlocked.

## Other entry points

| File | Purpose |
| --- | --- |
| `app.py` | Original dashboard for exploring the prediction model itself |
| `alpha_signal.py` | CLI walk-forward backtest with a text report |
| `alpha_report.py` | Self-contained HTML backtest report |
| `tests/alpha_test_suite.py` | Batch parameter grid with significance tests |
| `signal_history.py` | Generates and persists live (outcome-unknown) signals |

Market data comes from EODHD; set `EODHD_API_TOKEN` to use your own key.