# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Solar-XAI is a startup-pitch project: an end-to-end **photovoltaic (solar) power forecasting** pipeline with **XAI explanations** layered on top. The pitch story is *"accurate forecasts + explainable predictions for grid operators"*, so deliverables must serve both a research/engineering audience and a non-technical demo audience.

Note on terminology: the user sometimes says "태양열" (solar thermal), but the actual scope is **태양광 / photovoltaic (PV) power forecasting**. Clarify this distinction in user-facing docs (README) when relevant.

## Primary metric

**MAE is the primary tuning and reporting metric.** Always also report RMSE and MAPE alongside it, but optimize hyperparameters against validation MAE. The pitch frames MAE as "현업 친화적 (operator-friendly)" because it has the same units as power (kW/MW) and is robust to outliers.

When reporting results, **never blend our experimental numbers with numbers cited from papers** — keep paper benchmarks and our own runs in separate tables/sections.

## Modeling approach

- Build a ladder of models so the pitch can show "baseline → strong model" improvement:
  1. Persistence / naive baseline (yesterday-same-hour or last-observed)
  2. Linear / Ridge
  3. Gradient boosting: LightGBM (preferred for tabular time-series with weather features), XGBoost, or CatBoost
  4. At least one deep time-series model in PyTorch (candidates: LSTM/GRU, TFT, N-BEATS/N-HiTS, PatchTST, TimesNet — pick based on data shape and what reliably trains on the available GPU)
- Use **time-ordered** train/valid/test splits. No random shuffling. No leakage from future timestamps into features (rolling/lag features must use only past data).
- Fix all random seeds for reproducibility.
- Provide both `--quick` (fast smoke run, small search space) and `--full` (longer tuning) modes for any training script.
- Use the GPU when available (CUDA). LightGBM should use `device='gpu'` only if it's actually built with GPU support; otherwise leave on CPU and don't fight it.

## Hyperparameter tuning

- Use **Optuna** with `direction="minimize"` and objective = validation MAE.
- Persist trials, best params, and final test metrics to `outputs/` as CSV/JSON.
- Tune the strongest tree model thoroughly; the deep model gets a smaller search budget.

## XAI requirements

For tree models: **SHAP** (TreeExplainer) — summary plot, bar plot, dependence plots, and at least one local (single-prediction) force/waterfall plot.

For deep models: at least one of permutation importance, integrated gradients, attention visualization, or feature ablation.

Required XAI artifacts:
- Global feature importance plot
- SHAP summary (or equivalent) plot
- One **local explanation** for a specific date/hour
- A narrative on how weather variables, time-of-day variables, and lagged power features drive predictions
- A **Korean-language summary** the presenter can read off the slide ("발표자가 바로 설명할 수 있도록")

## Repository layout (target)

```
data/                  raw/processed datasets (gitignored)
scripts/
  prepare_data.py      download + clean + feature engineering
  train.py             trains all models, logs metrics
  evaluate.py          test-set metrics + comparison table
  explain.py           SHAP / XAI artifacts
run_pipeline.py        one-shot orchestrator (calls the four scripts)
src/                   reusable modules (features, models, metrics, plots)
outputs/
  metrics.csv          per-model MAE/RMSE/MAPE
  best_model_summary.md
  presentation_bullets.md
  figures/             all plots for the deck
REPORT.md              presentation-grade technical writeup
README.md              run instructions + dataset notes + result summary
requirements.txt
```

Each script must accept CLI flags (at minimum `--quick/--full`, `--data-dir`, `--output-dir`).

## Data sourcing

Preferred Kaggle datasets (try in order):
- "Solar Power Generation Data" (plant_1/plant_2 — generation + sensor data)
- DKASC (Desert Knowledge Australia Solar Centre) mirrors
- "Photovoltaic power generation forecasting" / weather + power pairs

Logic for `prepare_data.py`:
1. If `kaggle` CLI is configured (`~/.kaggle/kaggle.json` exists), download via `kaggle datasets download`.
2. Else, print a clear manual-download instruction with expected file paths under `data/raw/`.
3. **Always** ship a `--synthetic` fallback that generates a plausible PV time series (diurnal sinusoid + weather-driven noise + cloudy-day dropouts) so the entire pipeline runs end-to-end with no internet/auth. The pitch demo must never break because of a missing dataset.

Document the chosen dataset, its source URL, license, and download steps in README.md.

## Required deliverables

These files must exist and be populated before the pipeline is considered "done":
- `README.md` — run instructions, dataset description, model summary, headline results
- `REPORT.md` — pitch-grade technical report (problem, data, methods, results, XAI, takeaways)
- `outputs/metrics.csv` — one row per model with MAE, RMSE, MAPE on test set
- `outputs/best_model_summary.md` — the winning model, its hyperparameters, its metrics, an interpretation paragraph
- `outputs/presentation_bullets.md` — Korean-language slide bullets ready to paste
- `outputs/figures/` — actual-vs-predicted line plot, error-over-time plot, residual distribution, metric bar chart, feature importance / SHAP plots, at least one local-explanation plot

## Pitch framing rules (apply to all generated text)

- Lead with the XAI story: forecast accuracy is table stakes, **explainability is the differentiator**.
- Frame numbers as "baseline 대비 X% 개선" rather than absolute claims of being SOTA.
- Don't overclaim. If the deep model isn't beating LightGBM, say so and ship LightGBM as the production candidate — the pitch is stronger with an honest "right tool for the job" narrative than with inflated numbers.
- Korean-language summary text is for the presenter; English is fine for code comments and technical sections of REPORT.md.

## Working norms for this repo

- Run things end-to-end yourself before declaring success — don't claim metrics or figures exist without having generated them.
- If a dataset download fails, fall back to synthetic and note it explicitly in REPORT.md rather than silently skipping.
- Keep paper-cited benchmarks and our own measured numbers in clearly separated tables.
- Be proactive: the user has authorized running the full pipeline without intermediate confirmation. Pick reasonable defaults and proceed.
