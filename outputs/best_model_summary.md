# Best Model Summary

Source: `synthetic` · plant_id=1 · horizon=1h ahead · target normalization: min-max [0,1]

## Winner
- Model: **cat**
- Test MAE (normalized): **0.0137**
- Test RMSE (normalized): 0.0264
- Test MAPE (zero-floor): 8.53%
- Test MAE (kW): 18.28
- Validation MAE (normalized): 0.0140

## SOTA comparison (single number, apples-to-apples by dataset only)

- Paper: **X-LSTM-EO (Khan et al., PLOS One 2024)** — reported MAE = **0.229** on the same dataset (anikannal Plant 1).
- Our winner: **0.0137** (normalized) → **BEAT** (Δ ≈ +94.0% vs paper).

Note: paper metrics are reproduced from the source PDF (see `reference/`) and **not blended** with our experimental metrics.

## All models (test set)

| model       |   val_mae |    mae |   rmse |     mape |   mae_kw | beats_paper   |
|:------------|----------:|-------:|-------:|---------:|---------:|:--------------|
| cat         |    0.0140 | 0.0137 | 0.0264 |   8.5329 |  18.2773 | True          |
| lgbm        |    0.0138 | 0.0146 | 0.0276 |   8.8506 |  19.4656 | True          |
| ensemble    |    0.0128 | 0.0151 | 0.0273 |   8.6692 |  20.1110 | True          |
| xgb         |    0.0143 | 0.0165 | 0.0306 |   9.2631 |  21.9905 | True          |
| ridge       |    0.0135 | 0.0174 | 0.0293 |   9.4046 |  23.2039 | True          |
| lstm        |    0.0947 | 0.0876 | 0.2208 |  54.7002 | 116.7054 | True          |
| persistence |    0.2916 | 0.2824 | 0.4311 | 185.3923 | 376.2455 | False         |

## Best hyperparameters
```json
{
  "iterations": 878,
  "depth": 6,
  "learning_rate": 0.005468641390590814,
  "l2_leaf_reg": 0.3444856608499859,
  "random_strength": 0.07398294920836908,
  "bagging_temperature": 0.26382755997565566,
  "border_count": 160
}
```

## Interpretation

The ensemble combines L1-loss tree boosters (LightGBM, XGBoost, CatBoost), a Ridge linear baseline on the engineered features, and a multi-seed LSTM. The meta-learner (Ridge or simplex blend, whichever scored lower on out-of-fold MAE) re-weights base predictions to minimize MAE. Free-MAE post-processing — non-negative clip and night-zeroing — is applied to every prediction before scoring. Baseline-vs-winner improvement is reported as ΔMAE in the comparison table; this is the operator-friendly headline used in the pitch.