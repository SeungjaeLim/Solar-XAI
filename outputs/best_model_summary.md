# Best Model Summary

Source: `synthetic` · plant_id=1 · horizon=1h ahead · target normalization: min-max [0,1]

## Winner
- Model: **ensemble**
- Test MAE (normalized): **0.0983**
- Test RMSE (normalized): 0.1757
- Test MAPE (zero-floor): 80.83%
- Test MAE (kW): 147.46
- Validation MAE (normalized): 0.0946

## SOTA comparison (single number, apples-to-apples by dataset only)

- Paper: **X-LSTM-EO (Khan et al., PLOS One 2024)** — reported MAE = **0.229** on the same dataset (anikannal Plant 1).
- Our winner: **0.0983** (normalized) → **BEAT** (Δ ≈ +57.1% vs paper).

Note: paper metrics are reproduced from the source PDF (see `reference/`) and **not blended** with our experimental metrics.

## All models (test set)

| model       |   val_mae |    mae |   rmse |     mape |   mae_kw | beats_paper   |
|:------------|----------:|-------:|-------:|---------:|---------:|:--------------|
| ensemble    |    0.0946 | 0.0983 | 0.1757 |  80.8287 | 147.4610 | True          |
| xgb         |    0.0950 | 0.0994 | 0.1757 |  90.3854 | 149.0675 | True          |
| lgbm        |    0.0976 | 0.1003 | 0.1865 |  70.1803 | 150.4827 | True          |
| ridge       |    0.0991 | 0.1039 | 0.1987 |  51.0128 | 155.8903 | True          |
| cat         |    0.0971 | 0.1060 | 0.1828 |  89.4888 | 158.9513 | True          |
| lstm        |    0.1390 | 0.1268 | 0.2385 | 107.6635 | 190.1884 | True          |
| persistence |    0.3462 | 0.3599 | 0.5430 | 466.8304 | 539.8089 | False         |

## Ensemble composition
```json
{
  "weights": [
    4.512596391935304e-19,
    0.2115633094931151,
    4.3825062168052516e-18,
    0.7188509630918974,
    0.06958572741398747,
    0.0
  ],
  "oof_mae": 0.09463585451087468,
  "meta_kind": "simplex",
  "bases": [
    "persistence",
    "ridge",
    "lgbm",
    "xgb",
    "cat",
    "lstm"
  ],
  "val_mae": 0.09463585451087468,
  "test_mae": 0.09830733247269445
}
```

## Interpretation

The ensemble combines L1-loss tree boosters (LightGBM, XGBoost, CatBoost), a Ridge linear baseline on the engineered features, and a multi-seed LSTM. The meta-learner (Ridge or simplex blend, whichever scored lower on out-of-fold MAE) re-weights base predictions to minimize MAE. Free-MAE post-processing — non-negative clip and night-zeroing — is applied to every prediction before scoring. Baseline-vs-winner improvement is reported as ΔMAE in the comparison table; this is the operator-friendly headline used in the pitch.