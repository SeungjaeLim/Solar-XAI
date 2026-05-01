"""Stage 3: stack base predictions with Ridge meta + simplex blend, keep the lower OOF MAE."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import mae  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.stacking import best_of_blends  # noqa: E402


def load_preds(preds_dir: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack each base model's val and test predictions into 2D arrays."""
    val_cols, test_cols, used = [], [], []
    for n in names:
        path = preds_dir / f"{n}_preds.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            val_cols.append(data["val"])
            test_cols.append(data["test"])
        used.append(n)
    if not used:
        raise RuntimeError("No base predictions found.")
    val_arr = np.stack(val_cols, axis=1)
    test_arr = np.stack(test_cols, axis=1)
    return val_arr, test_arr, used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--bases",
        default="persistence,ridge,lgbm,xgb,cat,lstm",
        help="Base model names to stack (skipped if file missing)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    manifest = json.loads((data_dir / "processed" / "splits.json").read_text())
    feats = pd.read_parquet(data_dir / "processed" / "features.parquet")
    va_a, va_b = manifest["val_idx"]
    te_a, te_b = manifest["test_idx"]
    y_va = feats["target"].to_numpy()[va_a:va_b]
    y_te = feats["target"].to_numpy()[te_a:te_b]
    ts_te = feats.iloc[te_a:te_b]["DATE_TIME"]
    irr_te = feats.iloc[te_a:te_b]["IRRADIATION"].to_numpy()

    names = [n for n in args.bases.split(",") if n]
    val_arr, test_arr, used = load_preds(out_dir / "preds", names)
    print(f"[stack] using bases: {used}")

    # Drop bases that contain NaN (e.g. LSTM if it failed)
    keep = [i for i, n in enumerate(used) if not (np.isnan(val_arr[:, i]).any() or np.isnan(test_arr[:, i]).any())]
    if not keep:
        raise RuntimeError("All base predictions contain NaNs.")
    val_arr = val_arr[:, keep]
    test_arr = test_arr[:, keep]
    used = [used[i] for i in keep]

    oof_meta, test_meta, info = best_of_blends(val_arr, y_va, test_arr)
    test_meta = postprocess(test_meta, timestamps=ts_te, irradiation=irr_te, clip_max=1.0)
    oof_meta = postprocess(oof_meta, clip_max=1.0)

    np.savez_compressed(
        out_dir / "preds" / "ensemble_preds.npz", val=oof_meta, test=test_meta
    )

    info["bases"] = used
    info["val_mae"] = float(mae(y_va, oof_meta))
    info["test_mae"] = float(mae(y_te, test_meta))
    (out_dir / "models" / "ensemble_meta.json").write_text(json.dumps(info, indent=2))
    print(
        f"[stack] meta_kind={info['meta_kind']} val_mae={info['val_mae']:.4f} "
        f"test_mae={info['test_mae']:.4f}"
    )


if __name__ == "__main__":
    main()
