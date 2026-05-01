"""Stage 2: train all base models with Optuna (MAE objective), save OOF + test preds."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import mae  # noqa: E402
from src.models.baselines import RidgeBaseline, persistence_predict  # noqa: E402
from src.models.catboost_model import cat_objective_factory, fit_catboost  # noqa: E402
from src.models.lightgbm_model import fit_lightgbm, lgbm_objective_factory  # noqa: E402
from src.models.xgboost_model import fit_xgboost, xgb_objective_factory  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.tuning import run_optuna, study_to_df  # noqa: E402

warnings.filterwarnings("ignore")


QUICK_TRIALS = {"lgbm": 25, "xgb": 20, "cat": 15, "lstm": 6, "ridge": 0}
FULL_TRIALS = {"lgbm": 80, "xgb": 60, "cat": 60, "lstm": 15, "ridge": 0}


def _split_arrays(features_df: pd.DataFrame, manifest: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feat_cols = manifest["feature_cols"]
    X = features_df[feat_cols].to_numpy(dtype=np.float32)
    y = features_df["target"].to_numpy(dtype=np.float32)
    tr_a, tr_b = manifest["train_idx"]
    va_a, va_b = manifest["val_idx"]
    te_a, te_b = manifest["test_idx"]
    return (
        X[tr_a:tr_b],
        y[tr_a:tr_b],
        X[va_a:va_b],
        y[va_a:va_b],
        X[te_a:te_b],
        y[te_a:te_b],
        feat_cols,
    )


def _save_preds(out_dir: Path, name: str, val: np.ndarray, test: np.ndarray) -> None:
    np.savez_compressed(out_dir / f"{name}_preds.npz", val=val, test=test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default="ridge,lgbm,xgb,cat,lstm",
        help="Comma-separated list of base models to train",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    (out_dir / "trials").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data_dir / "processed" / "splits.json").read_text())
    feats = pd.read_parquet(data_dir / "processed" / "features.parquet")
    X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols = _split_arrays(feats, manifest)

    irradiation_va = feats.iloc[manifest["val_idx"][0] : manifest["val_idx"][1]]["IRRADIATION"].to_numpy()
    irradiation_te = feats.iloc[manifest["test_idx"][0] : manifest["test_idx"][1]]["IRRADIATION"].to_numpy()
    ts_va = feats.iloc[manifest["val_idx"][0] : manifest["val_idx"][1]]["DATE_TIME"]
    ts_te = feats.iloc[manifest["test_idx"][0] : manifest["test_idx"][1]]["DATE_TIME"]

    selected = set(args.models.split(","))
    trials = QUICK_TRIALS if args.quick else FULL_TRIALS
    if not args.quick and not args.full:
        trials = QUICK_TRIALS

    summary: dict = {}

    # ---- Persistence ----
    if "persistence" not in selected and "ridge" in selected:
        # Always include persistence as a free baseline
        pass
    pers_val = persistence_predict(feats.iloc[manifest["val_idx"][0] : manifest["val_idx"][1]])
    pers_te = persistence_predict(feats.iloc[manifest["test_idx"][0] : manifest["test_idx"][1]])
    pers_val = postprocess(pers_val, timestamps=ts_va, irradiation=irradiation_va, clip_max=1.0)
    pers_te = postprocess(pers_te, timestamps=ts_te, irradiation=irradiation_te, clip_max=1.0)
    _save_preds(out_dir / "preds", "persistence", pers_val, pers_te)
    summary["persistence"] = {"val_mae": mae(y_va, pers_val), "test_mae": mae(y_te, pers_te)}
    print(f"[persistence] val_mae={summary['persistence']['val_mae']:.4f} test_mae={summary['persistence']['test_mae']:.4f}")

    # ---- Ridge ----
    if "ridge" in selected:
        ridge = RidgeBaseline(alpha=1.0).fit(X_tr, y_tr)
        ridge_val = postprocess(ridge.predict(X_va), timestamps=ts_va, irradiation=irradiation_va, clip_max=1.0)
        ridge_te = postprocess(ridge.predict(X_te), timestamps=ts_te, irradiation=irradiation_te, clip_max=1.0)
        _save_preds(out_dir / "preds", "ridge", ridge_val, ridge_te)
        joblib.dump(ridge, out_dir / "models" / "ridge.pkl")
        summary["ridge"] = {"val_mae": mae(y_va, ridge_val), "test_mae": mae(y_te, ridge_te)}
        print(f"[ridge] val_mae={summary['ridge']['val_mae']:.4f} test_mae={summary['ridge']['test_mae']:.4f}")

    # ---- LightGBM ----
    if "lgbm" in selected and trials["lgbm"] > 0:
        study = run_optuna(lgbm_objective_factory(X_tr, y_tr, X_va, y_va), n_trials=trials["lgbm"], seed=args.seed)
        study_to_df(study).to_csv(out_dir / "trials" / "lgbm.csv", index=False)
        best = dict(study.best_params)
        model = fit_lightgbm(X_tr, y_tr, X_va, y_va, params=best)
        v = postprocess(model.predict(X_va), timestamps=ts_va, irradiation=irradiation_va, clip_max=1.0)
        t = postprocess(model.predict(X_te), timestamps=ts_te, irradiation=irradiation_te, clip_max=1.0)
        _save_preds(out_dir / "preds", "lgbm", v, t)
        joblib.dump(model, out_dir / "models" / "lgbm.pkl")
        (out_dir / "models" / "lgbm_params.json").write_text(json.dumps(best, indent=2))
        summary["lgbm"] = {"val_mae": mae(y_va, v), "test_mae": mae(y_te, t), "params": best}
        print(f"[lgbm] val_mae={summary['lgbm']['val_mae']:.4f} test_mae={summary['lgbm']['test_mae']:.4f}")

    # ---- XGBoost ----
    if "xgb" in selected and trials["xgb"] > 0:
        study = run_optuna(xgb_objective_factory(X_tr, y_tr, X_va, y_va), n_trials=trials["xgb"], seed=args.seed + 1)
        study_to_df(study).to_csv(out_dir / "trials" / "xgb.csv", index=False)
        best = dict(study.best_params)
        model = fit_xgboost(X_tr, y_tr, X_va, y_va, params=best)
        v = postprocess(model.predict(X_va), timestamps=ts_va, irradiation=irradiation_va, clip_max=1.0)
        t = postprocess(model.predict(X_te), timestamps=ts_te, irradiation=irradiation_te, clip_max=1.0)
        _save_preds(out_dir / "preds", "xgb", v, t)
        joblib.dump(model, out_dir / "models" / "xgb.pkl")
        (out_dir / "models" / "xgb_params.json").write_text(json.dumps(best, indent=2))
        summary["xgb"] = {"val_mae": mae(y_va, v), "test_mae": mae(y_te, t), "params": best}
        print(f"[xgb] val_mae={summary['xgb']['val_mae']:.4f} test_mae={summary['xgb']['test_mae']:.4f}")

    # ---- CatBoost ----
    if "cat" in selected and trials["cat"] > 0:
        study = run_optuna(cat_objective_factory(X_tr, y_tr, X_va, y_va), n_trials=trials["cat"], seed=args.seed + 2)
        study_to_df(study).to_csv(out_dir / "trials" / "cat.csv", index=False)
        best = dict(study.best_params)
        model = fit_catboost(X_tr, y_tr, X_va, y_va, params=best)
        v = postprocess(model.predict(X_va), timestamps=ts_va, irradiation=irradiation_va, clip_max=1.0)
        t = postprocess(model.predict(X_te), timestamps=ts_te, irradiation=irradiation_te, clip_max=1.0)
        _save_preds(out_dir / "preds", "cat", v, t)
        joblib.dump(model, out_dir / "models" / "cat.pkl")
        (out_dir / "models" / "cat_params.json").write_text(json.dumps(best, indent=2))
        summary["cat"] = {"val_mae": mae(y_va, v), "test_mae": mae(y_te, t), "params": best}
        print(f"[cat] val_mae={summary['cat']['val_mae']:.4f} test_mae={summary['cat']['test_mae']:.4f}")

    # ---- LSTM ----
    if "lstm" in selected and trials["lstm"] > 0:
        try:
            import torch  # noqa: F401

            from src.models.lstm import (
                LSTMConfig,
                fit_lstm_multi_seed,
                lstm_objective_factory,
                predict_multi_seed,
            )

            obj = lstm_objective_factory(X_tr, y_tr, X_va, y_va, epochs=15 if args.quick else 30)
            study = run_optuna(obj, n_trials=trials["lstm"], seed=args.seed + 3)
            study_to_df(study).to_csv(out_dir / "trials" / "lstm.csv", index=False)
            best_params = dict(study.best_params)
            cfg = LSTMConfig(
                input_size=X_tr.shape[1],
                hidden_size=best_params["hidden_size"],
                num_layers=best_params["num_layers"],
                dropout=best_params["dropout"],
                window=best_params["window"],
                lr=best_params["lr"],
                batch_size=best_params["batch_size"],
                epochs=20 if args.quick else 60,
                patience=6,
                seed=42,
            )
            seeds = (42, 7, 123) if args.quick else (42, 7, 123, 2024, 31337)
            models, _ = fit_lstm_multi_seed(X_tr, y_tr, X_va, y_va, cfg, seeds=seeds)

            v_raw = predict_multi_seed(models, X_va, cfg.window)
            t_raw = predict_multi_seed(models, X_te, cfg.window)
            # Replace leading NaNs (window warm-up) with persistence values
            v_filled = np.where(np.isnan(v_raw), pers_val, v_raw)
            t_filled = np.where(np.isnan(t_raw), pers_te, t_raw)
            v = postprocess(v_filled, timestamps=ts_va, irradiation=irradiation_va, clip_max=1.0)
            t = postprocess(t_filled, timestamps=ts_te, irradiation=irradiation_te, clip_max=1.0)
            _save_preds(out_dir / "preds", "lstm", v, t)

            # Save first model + cfg for explanation step
            import torch as _torch

            _torch.save(models[0].state_dict(), out_dir / "models" / "lstm_seed0.pt")
            (out_dir / "models" / "lstm_params.json").write_text(json.dumps({**best_params, "window": cfg.window, "seeds": list(seeds)}, indent=2))
            summary["lstm"] = {"val_mae": mae(y_va, v), "test_mae": mae(y_te, t), "params": best_params}
            print(f"[lstm] val_mae={summary['lstm']['val_mae']:.4f} test_mae={summary['lstm']['test_mae']:.4f}")
        except Exception as e:
            print(f"[lstm] SKIPPED — {e}")
            summary["lstm"] = {"error": str(e)}

    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[train] summary written to {out_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
