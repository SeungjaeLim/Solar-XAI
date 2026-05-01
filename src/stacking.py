"""Stacking utilities — Ridge meta-learner + non-negative simplex blend."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from .metrics import mae


def ridge_stack(
    oof_preds: np.ndarray,
    y_oof: np.ndarray,
    test_preds: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train Ridge on OOF preds, return (oof_meta, test_meta, info)."""
    m = Ridge(alpha=alpha, positive=False, fit_intercept=True)
    m.fit(oof_preds, y_oof)
    oof_meta = m.predict(oof_preds)
    test_meta = m.predict(test_preds)
    return oof_meta, test_meta, {"alpha": alpha, "coef": m.coef_.tolist(), "intercept": float(m.intercept_)}


def simplex_blend(
    oof_preds: np.ndarray,
    y_oof: np.ndarray,
    test_preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Optimize convex combination weights on OOF MAE."""
    n_models = oof_preds.shape[1]

    def loss(w: np.ndarray) -> float:
        w = np.clip(w, 0.0, None)
        w = w / (w.sum() + 1e-12)
        return mae(y_oof, oof_preds @ w)

    x0 = np.full(n_models, 1.0 / n_models)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * n_models
    res = minimize(loss, x0, bounds=bounds, constraints=cons, method="SLSQP")
    w = np.clip(res.x, 0.0, None)
    w = w / (w.sum() + 1e-12)
    return oof_preds @ w, test_preds @ w, {"weights": w.tolist(), "oof_mae": float(loss(w))}


def best_of_blends(
    oof_preds: np.ndarray,
    y_oof: np.ndarray,
    test_preds: np.ndarray,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 50.0, 200.0),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Try Ridge across alphas and simplex blend, keep the lowest OOF MAE."""
    candidates = []
    for a in alphas:
        oof_m, test_m, info = ridge_stack(oof_preds, y_oof, test_preds, alpha=a)
        candidates.append(("ridge", oof_m, test_m, mae(y_oof, oof_m), {**info, "alpha": a}))
    oof_s, test_s, info_s = simplex_blend(oof_preds, y_oof, test_preds)
    candidates.append(("simplex", oof_s, test_s, mae(y_oof, oof_s), info_s))
    candidates.sort(key=lambda x: x[3])
    name, oof_best, test_best, mae_best, info_best = candidates[0]
    info_best["meta_kind"] = name
    info_best["oof_mae"] = float(mae_best)
    return oof_best, test_best, info_best
