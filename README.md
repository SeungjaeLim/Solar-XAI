# Solar-XAI

End-to-end **photovoltaic (PV) power forecasting** pipeline with **explainable AI (XAI)** layered on top. Designed for a startup pitch where two stories must land at once:

1. **Strong forecast accuracy** — beat a published SOTA paper's MAE on the same dataset.
2. **Operator-friendly explainability** — show *why* the model predicts what it predicts (SHAP for tree models, permutation + Integrated Gradients for the LSTM), so grid operators can trust the output.

> Terminology: the project owner sometimes says "태양열" (solar thermal). The actual scope is **태양광 / photovoltaic (PV) power forecasting** — i.e. electricity generated from PV panels, not heat from solar collectors.

## Headline result

- **Primary SOTA target**: X-LSTM-EO (Khan et al., PLOS One 2024) — reported MAE **0.229** on Kaggle `anikannal/solar-power-generation-data` Plant 1, normalized AC power.
- **Our headline**: see `outputs/best_model_summary.md` after running `python run_pipeline.py --full`. The pipeline reports `test_mae` on the same min-max-normalized scale and an explicit `BEAT` / `NOT BEAT` flag.
- **Paper-vs-ours separation**: paper-reported numbers live in `reference/benchmarks.md`. Our experimental numbers live in `outputs/metrics.csv`. **They are never blended into the same table.**

## Repository layout

```
.
├── README.md                this file
├── REPORT.md                pitch-grade technical report
├── CLAUDE.md                project rules for AI assistants
├── requirements.txt
├── run_pipeline.py          orchestrates the full pipeline
├── reference/               SOTA paper PDFs + benchmarks.md (read-only references)
├── scripts/
│   ├── prepare_data.py      downloads or synthesizes data, builds features, splits
│   ├── fetch_references.py  downloads SOTA paper PDFs
│   ├── train.py             trains every base model with Optuna (MAE objective)
│   ├── stack.py             builds Ridge meta-learner / simplex blend
│   ├── evaluate.py          metrics + plots + best_model_summary.md
│   └── explain.py           SHAP + LSTM XAI + Korean presentation_bullets.md
├── src/                     reusable modules (features, models, stacking, plots)
├── data/                    gitignored — downloaded or synthesized PV data
└── outputs/                 gitignored — metrics, figures, model artifacts
```

## Setup

```bash
pip install -r requirements.txt
# PyTorch with CUDA 12.1 (the requirements file has a placeholder; this is the canonical command):
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

GPU is optional; if it's not detected, XGBoost and CatBoost fall back to CPU and the LSTM trains on CPU (slower but functional).

## Data

**Primary dataset**: [`anikannal/solar-power-generation-data`](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) on Kaggle — 34-day, 15-minute-resolution generation + weather sensor data for two Indian PV plants (Plant 1 and Plant 2). License is Kaggle's default (educational/non-commercial); see the dataset page for current terms.

**Download**:
1. Get a Kaggle API token (Account → Create API Token) and place `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows).
2. `python scripts/prepare_data.py` will detect the credentials and download the dataset.
3. If credentials are missing or the download fails, the script automatically falls back to a **synthetic generator** (`src/synthetic.py`) so the rest of the pipeline still runs end-to-end.

**Synthetic mode**: the synthetic plant produces 15-minute observations of `AC_POWER`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION` driven by a diurnal solar curve, a Beta-distributed daily clearness factor, panel-temperature derate, and occasional cloudy days / outages. Synthetic mode is for runnability and sanity, **not** for the SOTA-comparison claim.

## Run the pipeline

End-to-end:

```bash
# Quick smoke run (synthetic, ~2 min on a desktop GPU)
python run_pipeline.py --quick --synthetic

# Full run (real Kaggle data if creds available, else synthetic; ~25–40 min on a desktop GPU)
python run_pipeline.py --full
```

Stage by stage:

```bash
python scripts/fetch_references.py        # downloads paper PDFs to reference/
python scripts/prepare_data.py            # data/processed/{features.parquet, splits.json}
python scripts/train.py --full            # outputs/{trials,preds,models}/
python scripts/stack.py                   # outputs/preds/ensemble_preds.npz
python scripts/evaluate.py                # outputs/metrics.csv + outputs/figures/*
python scripts/explain.py                 # SHAP + LSTM XAI + Korean bullets
```

## Models

Base learners (all use **MAE-aligned losses**):

| Model | Loss | Why included |
|---|---|---|
| Persistence (yesterday-same-hour) | – | Baseline for "baseline 대비 X% 개선" framing |
| Ridge on engineered features | L2 (linear coefficients) | Stacking diversity |
| **LightGBM** | `regression_l1` | Best tabular fit; SHAP-friendly |
| **XGBoost** | `reg:absoluteerror` | GPU-accelerated MAE booster |
| **CatBoost** | `MAE` | GPU; different splitting strategy |
| **LSTM (PyTorch)** | `nn.L1Loss()` | Required deep-time-series model; multi-seed averaged |

**Stacking**: out-of-fold predictions from each base feed into a meta-learner (Ridge across several α values) **and** a constrained simplex blend (non-negative weights, sum=1). Whichever scores lower OOF MAE is kept as the final ensemble.

**Free-MAE post-processing**: every prediction is clipped non-negative and force-zeroed when irradiation ≈ 0 or the timestamp is outside daylight hours.

## XAI

- **SHAP TreeExplainer** on the strongest tree model (LightGBM): summary beeswarm, mean |SHAP| bar, dependence plots for `IRRADIATION` / `clear_sky` / `MODULE_TEMPERATURE` / `cloud_index`, and two waterfall plots (a sunny-noon and a cloudy-noon test instance).
- **LSTM**: permutation importance over input features (ΔMAE on val) and Captum **Integrated Gradients** on a single high-irradiance test window (heatmap: time × feature).
- **Korean narrative**: `outputs/presentation_bullets.md` reads off the SHAP top features and translates them into pitch-deck-ready bullet points.

## Outputs

After a full run, the following files exist:

- `outputs/metrics.csv` — one row per model with `val_mae`, `mae`, `rmse`, `mape`, `mae_kw`, `beats_paper` flag
- `outputs/best_model_summary.md` — winner, metrics, hyperparameters, paper comparison
- `outputs/presentation_bullets.md` — Korean slide bullets for the pitch
- `outputs/figures/` — actual_vs_pred, error_over_time, residual_hist, metric_bars, shap_summary, shap_bar, shap_dependence_*, shap_waterfall_*, lstm_permutation, lstm_ig_window
- `outputs/models/` — trained model artifacts (`*.pkl`, `*.pt`) and best hyperparameters (`*_params.json`)
- `outputs/trials/` — per-model Optuna trial logs

## Determinism

`src/seed.py` seeds `random`, `numpy`, and (when available) `torch` for reproducibility. cuDNN deterministic mode is enabled. All scripts call `seed_everything()` before doing anything.

## Honesty notes

- The `< 0.229` MAE claim depends on training on the **same dataset** as the paper. If you run with `--synthetic` (no Kaggle credentials), the headline number is computed on synthetic data and is **not** an apples-to-apples beat of the paper. See `outputs/best_model_summary.md` — it always notes the data source.
- Paper-cited numbers live in `reference/benchmarks.md`. Our experimental numbers live in `outputs/metrics.csv`. These two are never blended.
- If the deep model (LSTM) doesn't beat the trees on a given dataset, that's reported honestly in the metrics table; the production candidate is whichever model wins, not whichever model is the most fashionable.
