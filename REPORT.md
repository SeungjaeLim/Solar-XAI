# Solar-XAI: PV Power Forecasting + Explainability — Technical Report

> Pitch-grade summary for the Solar-XAI startup. Runnable code in `run_pipeline.py`. Final numbers are produced by the pipeline and live in `outputs/metrics.csv` and `outputs/best_model_summary.md`. **Paper-reported numbers stay in `reference/benchmarks.md` and are never blended with our experimental results.**

## 1. Problem framing

Photovoltaic (PV) power output depends non-linearly on irradiance, panel temperature, atmospheric clearness, and time-of-day geometry. Grid operators need short-horizon forecasts (1 h to ~24 h ahead) to schedule reserve generation and avoid imbalance penalties. Two common pain points:

1. **Forecast accuracy is brittle on cloudy / shoulder-season days.** Mean absolute error (MAE) on power can spike 3–5× compared to clear-sky days.
2. **Forecast models are black boxes.** Operators are reluctant to act on numbers they cannot explain to their boss or regulator.

Solar-XAI's pitch: a forecasting service that is **simultaneously accurate (beats a published SOTA paper's MAE on the same dataset) and explainable (per-prediction SHAP / IG attributions)**, so an operator can both trust and defend the forecast.

> Terminology note: in Korean the project is sometimes referred to as 태양열 (solar thermal). The actual technical scope is 태양광 / photovoltaic power forecasting — the conversion of sunlight into electricity by PV panels, not heat from solar collectors.

## 2. Dataset

**Primary**: Kaggle [`anikannal/solar-power-generation-data`](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) — Plant 1, India, 34 days × 15-minute resolution. Inverter-level `AC_POWER` and weather-station `IRRADIATION` / `AMBIENT_TEMPERATURE` / `MODULE_TEMPERATURE`.

**Pre-processing**:
1. Aggregate `AC_POWER` across inverters per timestamp (per-plant total kW).
2. Resample to 1-hour mean.
3. Min-max normalize the target so MAE is reported on the same scale as the SOTA paper.
4. Time-ordered 70/15/15 train/val/test split — no shuffling.
5. Drop rows with NaN target after the lag/rolling features are computed.

**Synthetic fallback** (`src/synthetic.py`): when Kaggle credentials are missing the pipeline generates 2 plants × 2 years × 15-min observations from a physically motivated model: diurnal sine for solar geometry, Beta-distributed daily clearness, panel-temperature derate (`-0.4 %/°C` above 25 °C), Gaussian noise, occasional cloudy days and outage hours. Synthetic mode keeps the pipeline runnable end-to-end but does **not** support the SOTA-comparison claim — the final summary explicitly flags the data source.

## 3. Methods

### 3.1 Feature engineering

We compute three families of features, all leak-free (lag and rolling statistics are shifted before they enter row `t`):

| Family | Examples |
|---|---|
| Time / solar geometry | `hour`, `dayofyear`, `sin/cos(hour)`, `sin/cos(dayofyear)`, **solar zenith / cos zenith**, **air mass (Kasten)**, **clear-sky irradiance** estimate |
| Weather + interactions | `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `temp_delta`, `irrad²`, `irrad × cos_zenith`, `Δirrad_1h`, **`cloud_index = irrad / clear_sky`** |
| Power lags + calibration | `power_lag_{1,2,3,6,24,48,168}`, rolling mean/std/max over `{3,6,24}`h, **same-hour 3-day mean**, `yesterday_residual`, **`power_per_irrad_lag1`**, **`expected_power = capacity · clear_sky · cos_zenith · derate(module_temp)`** |

The clear-sky term and the physics-informed `expected_power` give the model a strong inductive bias: deviations from these priors carry the cloud / panel-degradation signal, which is the actual learning target.

### 3.2 Model ladder

| Tier | Model | Loss | Why |
|---|---|---|---|
| Naive | Persistence (`y[t] = y[t-24h]`) | – | Required baseline for "ΔMAE vs baseline" framing |
| Linear | Ridge on engineered features | L2 | Stacking diversity |
| Tree #1 | **LightGBM** | `regression_l1` (MAE) | Best tabular fit; native SHAP support |
| Tree #2 | **XGBoost** (GPU, `tree_method=hist`) | `reg:absoluteerror` (MAE) | Different splitting heuristic for ensemble diversity |
| Tree #3 | **CatBoost** (GPU) | `MAE` | Categorical-aware ordered boosting |
| Deep | **PyTorch LSTM** (2-layer, multi-seed averaged) | `nn.L1Loss()` (MAE) | Required deep time-series model; lightweight on 8 GB GPU |
| Meta | **Stacking**: Ridge meta-learner across α grid + non-negative simplex blend | OOF MAE | Pick whichever gives lower OOF MAE |

**Why the losses are MAE-aligned**: the headline metric is MAE. Training under L2 (MSE) and reporting MAE is a small but consistent leakage of objective into evaluation. Using `regression_l1` / `reg:absoluteerror` / `MAE` / `nn.L1Loss()` directly aligns optimization with reporting and typically buys 5–10 % MAE on top of an L2-trained baseline.

**Free-MAE post-processing**: every prediction is `clip(0, 1)` on the normalized scale and force-zeroed when timestamp is outside daylight hours or `IRRADIATION ≈ 0`. PV power cannot be negative and is exactly zero at night, so this is a free MAE win on the order of 5–15 % depending on dataset.

### 3.3 Hyperparameter tuning

[Optuna](https://optuna.org/) with the TPE sampler, `direction="minimize"`, validation MAE objective. Trial counts:

| Mode | LightGBM | XGBoost | CatBoost | LSTM |
|---|---:|---:|---:|---:|
| `--quick` | 25 | 20 | 15 | 6 |
| `--full` | 80 | 60 | 60 | 15 |

LSTM uses Adam with early stopping on val MAE (patience 6). After the best LSTM hyperparameters are locked, we re-train on `(42, 7, 123, 2024, 31337)` seeds and average their predictions.

### 3.4 Stacking

For every base model we already have validation predictions (the `val` split is also our OOF stand-in for a single-fold setup). We fit:
- A Ridge meta-learner with α ∈ {0.1, 1, 10, 50, 200} and pick the best α by OOF MAE.
- A non-negative simplex blend (weights ≥ 0, summing to 1) that directly minimizes OOF MAE via SLSQP.

Whichever has lower OOF MAE is the final ensemble.

## 4. Results

### 4.1 Headline number

The current best is recorded in `outputs/metrics.csv` after `python run_pipeline.py --full`. The summary file (`outputs/best_model_summary.md`) auto-includes:

- Test `MAE_normalized` (paper-comparable) and `MAE_kW` (ops-friendly)
- Test RMSE and MAPE (zero-floored at 5 % of max power to avoid blow-up at near-zero AC)
- A `BEAT` / `NOT BEAT` flag against the X-LSTM-EO paper's MAE = 0.229
- Best hyperparameters (or ensemble weights) for the winner

### 4.2 Paper-cited benchmarks (READ-ONLY, separate)

These numbers are **paper-reported, on the dataset each paper used, with the protocol each paper documented**. They live in `reference/benchmarks.md` and are **not blended** with our experimental table.

| Paper | Dataset | Reported headline |
|---|---|---|
| Khan et al. 2024 (X-LSTM-EO, PLOS One) | anikannal Plant 1 | MAE = 0.229 (normalized) — *primary target* |
| Solar ML comparison (PLOS One) | anikannal | per-model table |
| Stacking metamodel at extreme altitudes (Nature Sci Rep 2025) | High-altitude PV plant | MAE = 6.76 kW, R² = 0.9999 |
| LightGBM medium-term (MDPI Energies 2025) | Distinct PV plant | MAE = 37.49 kW, R² = 0.89 |
| Ultra-short-term (arXiv 2509.17095) | Multiple plants | model class reference |
| France renewables benchmark (arXiv 2504.16100) | France 2012-2023 | benchmarking framework |
| Quantum ML PV (arXiv 2312.16379) | PV plant | alternative model class |
| Ensemble post-processing (arXiv 2508.15508) | Operational forecasts | post-processing reference |

### 4.3 Our experimental results (computed by this repo)

See `outputs/metrics.csv` after running. Each row is a model, sorted by test MAE ascending. The pipeline also produces:

- `outputs/figures/actual_vs_pred_<winner>.png` — last 7 days of actual vs predicted
- `outputs/figures/error_over_time_<winner>.png` — daily mean absolute error
- `outputs/figures/residual_hist_<winner>.png` — residual histogram
- `outputs/figures/metric_bars.png` — grouped MAE/RMSE/MAPE bars by model

## 5. Explainability (XAI)

### 5.1 LightGBM — SHAP

`scripts/explain.py` runs `shap.TreeExplainer` on a 3 000-row sample of the test set. Artifacts:

- `figures/shap_summary.png` — beeswarm (per-instance SHAP value distribution by feature)
- `figures/shap_bar.png` — mean |SHAP| ranking, top 20
- `figures/shap_dependence_{IRRADIATION, clear_sky, MODULE_TEMPERATURE, cloud_index}.png` — partial-dependence-style plots showing how SHAP value scales with the feature
- `figures/shap_waterfall_{sunny,cloudy}_<datetime>.png` — single-instance waterfall plots for a sunny-noon prediction and a cloudy-noon prediction (operator-facing local explanation)

Expected pattern (confirmed by sanity gate): top features include `IRRADIATION`, `clear_sky`, `MODULE_TEMPERATURE`, `power_lag_{1,24}`, `cloud_index`. The model's "physics intuition" is right — high irradiance and high cos-zenith push prediction up; high module temperature pushes it down (panel derate); the residual against `clear_sky` (`cloud_index`) is the cloud-cover proxy.

### 5.2 LSTM — Permutation + Integrated Gradients

- **Permutation importance**: shuffle each feature within each window and measure ΔMAE on val. The output (`figures/lstm_permutation.png`) is comparable to the SHAP bar chart but is model-agnostic.
- **Integrated Gradients (Captum)**: pick a high-irradiance test window, compute IG attributions (`figures/lstm_ig_window.png`). The result is a `time × feature` heatmap showing which time-step / feature combinations drove the prediction up or down. We disable cuDNN during attribution because cuDNN's RNN backward isn't differentiable in eval mode.

### 5.3 Korean presentation bullets

`outputs/presentation_bullets.md` is auto-generated from the SHAP top features and the headline metrics. It is structured for direct paste into pitch slides:

1. 문제 정의
2. 데이터 & 전처리
3. 모델 사다리 (MAE 중심 튜닝)
4. 핵심 결과 — MAE 최우선
5. 왜 이렇게 예측이 나왔는가? — XAI 인사이트
6. 비즈니스 임팩트
7. 다음 단계

## 6. Limitations & honesty notes

- **Single dataset, single horizon (1 h ahead)**. Results may not transfer to other plants or longer horizons without re-tuning.
- **No probabilistic forecast** — point predictions only. Operationally, an operator wants a confidence band; that's tracked in §7.
- **Free-MAE post-processing assumes a known plant capacity and daylight window**. Both are derived from the data itself in this repo.
- **The deep model (LSTM) frequently does not beat the tree ensemble** on tabular weather + lag features. We report this honestly in the metrics table and ship whichever model wins, not whichever is fashionable.
- **Synthetic mode is for runnability, not benchmarking**. The `< 0.229` claim is meaningful only when the real Kaggle dataset is used.

## 7. Roadmap

1. Multi-site generalization — train on Plant 1 + Plant 2 + a third public dataset (NSRDB or DKASC), evaluate on held-out sites.
2. Multi-horizon (1, 3, 6, 24 h) — a single seq2seq model or H-headed boosting.
3. Probabilistic forecast — quantile loss in LightGBM (`objective='quantile'` for τ ∈ {0.1, 0.5, 0.9}); calibrated prediction intervals.
4. Operator dashboard — surface SHAP local explanations alongside each forecast, in Korean.
5. Online drift monitoring — alert when feature distributions or residual autocorrelation drift outside training-set bounds.

## 8. Reproducibility

- All randomness is seeded via `src/seed.py`.
- `cudnn.deterministic = True`, `cudnn.benchmark = False`.
- Quick smoke run finishes in a few minutes; full run finishes in ~25–40 minutes on an RTX 4060 Ti class GPU.
- Every script accepts `--data-dir`, `--output-dir`, `--seed`, plus `--quick / --full / --synthetic` where applicable.

```bash
python run_pipeline.py --full          # canonical run on real data when Kaggle is set up
python run_pipeline.py --quick --synthetic   # smoke test, no auth, ~2 min
```
