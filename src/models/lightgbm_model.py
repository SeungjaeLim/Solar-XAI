"""LightGBM with regression_l1 (MAE-aligned)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna

from ..metrics import mae


def lgbm_objective_factory(X_tr, y_tr, X_va, y_va):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 5.0, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "seed": 42,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_va)
        return mae(y_va, preds)

    return objective


def fit_lightgbm(X_tr, y_tr, X_va, y_va, params: dict | None = None) -> lgb.LGBMRegressor:
    base = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "seed": 42,
    }
    if params:
        base.update(params)
    base.setdefault("n_estimators", 1500)
    model = lgb.LGBMRegressor(**base)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model
