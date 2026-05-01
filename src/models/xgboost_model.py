"""XGBoost with reg:absoluteerror (MAE-aligned), GPU when available."""
from __future__ import annotations

import optuna
import xgboost as xgb

from ..metrics import mae


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def xgb_objective_factory(X_tr, y_tr, X_va, y_va):
    use_gpu = _gpu_available()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "verbosity": 0,
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        return mae(y_va, preds)

    return objective


def fit_xgboost(X_tr, y_tr, X_va, y_va, params: dict | None = None) -> xgb.XGBRegressor:
    use_gpu = _gpu_available()
    base = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "verbosity": 0,
        "random_state": 42,
    }
    if params:
        base.update(params)
    base.setdefault("n_estimators", 1500)
    model = xgb.XGBRegressor(**base, early_stopping_rounds=50)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model
