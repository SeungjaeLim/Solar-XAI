"""Optuna helpers — MAE objective, time-aware CV."""
from __future__ import annotations

from typing import Callable

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna(
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    seed: int = 42,
    study_name: str | None = None,
    direction: str = "minimize",
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def study_to_df(study: optuna.Study):
    return study.trials_dataframe(attrs=("number", "value", "params", "state"))
