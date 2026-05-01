"""CatBoost with MAE loss, GPU when available."""
from __future__ import annotations

import catboost as cb
import optuna

from ..metrics import mae


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def cat_objective_factory(X_tr, y_tr, X_va, y_va):
    use_gpu = _gpu_available()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "task_type": "GPU" if use_gpu else "CPU",
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 254),
            "verbose": False,
            "random_seed": 42,
            "allow_writing_files": False,
        }
        if use_gpu:
            params["devices"] = "0"
        model = cb.CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_va)
        return mae(y_va, preds)

    return objective


def fit_catboost(X_tr, y_tr, X_va, y_va, params: dict | None = None) -> cb.CatBoostRegressor:
    use_gpu = _gpu_available()
    base = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "task_type": "GPU" if use_gpu else "CPU",
        "verbose": False,
        "random_seed": 42,
        "allow_writing_files": False,
    }
    if use_gpu:
        base["devices"] = "0"
    if params:
        base.update(params)
    base.setdefault("iterations", 1500)
    model = cb.CatBoostRegressor(**base)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    return model
