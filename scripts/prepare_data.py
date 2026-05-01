"""Stage 1: download / synthesize PV data, engineer features, time-ordered split.

Outputs to data/processed/:
  raw_hourly.parquet     cleaned hourly aggregate
  features.parquet       engineered features + target + ts column
  splits.json            train/val/test indices + capacity + normalization stats
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_features  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.splits import time_ordered_split  # noqa: E402
from src.synthetic import make_synthetic_dataset  # noqa: E402


def try_kaggle(raw_dir: Path) -> bool:
    """Try to download the anikannal solar dataset via the kaggle CLI."""
    try:
        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "anikannal/solar-power-generation-data",
                "-p",
                str(raw_dir),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        ok = result.returncode == 0 and any(raw_dir.glob("*.csv"))
        if not ok:
            print("[prepare_data] Kaggle download failed:", result.stderr.strip()[:300])
        return ok
    except FileNotFoundError:
        print("[prepare_data] Kaggle CLI not found.")
        return False
    except subprocess.TimeoutExpired:
        print("[prepare_data] Kaggle download timed out.")
        return False


def load_anikannal(raw_dir: Path) -> pd.DataFrame | None:
    """Merge plant generation + sensor data into a long-format df.

    Schema: DATE_TIME, PLANT_ID, AC_POWER, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION
    """
    gens = list(raw_dir.glob("Plant_?_Generation_Data.csv")) + list(
        raw_dir.glob("Plant_?_Generation_Data*.csv")
    )
    weas = list(raw_dir.glob("Plant_?_Weather_Sensor_Data.csv")) + list(
        raw_dir.glob("Plant_?_Weather_Sensor_Data*.csv")
    )
    if not gens or not weas:
        return None

    frames = []
    for g_path in gens:
        plant_num = int(g_path.stem.split("_")[1])
        w_path = next((w for w in weas if int(w.stem.split("_")[1]) == plant_num), None)
        if w_path is None:
            continue
        gen = pd.read_csv(g_path)
        wea = pd.read_csv(w_path)
        gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], errors="coerce", dayfirst=True)
        if gen["DATE_TIME"].isna().mean() > 0.3:
            gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], errors="coerce")
        wea["DATE_TIME"] = pd.to_datetime(wea["DATE_TIME"], errors="coerce")
        # Aggregate AC_POWER across inverters per timestamp
        gen_agg = gen.groupby("DATE_TIME", as_index=False)["AC_POWER"].sum()
        wea_agg = wea.groupby("DATE_TIME", as_index=False)[
            ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
        ].mean()
        merged = pd.merge(gen_agg, wea_agg, on="DATE_TIME", how="inner")
        merged["PLANT_ID"] = plant_num
        frames.append(merged)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).sort_values(["PLANT_ID", "DATE_TIME"])
    return out


def hourly_aggregate(df: pd.DataFrame, plant_id: int = 1) -> pd.DataFrame:
    sub = df[df["PLANT_ID"] == plant_id].copy()
    sub = sub.set_index("DATE_TIME").sort_index()
    sub_h = sub.resample("1h").mean(numeric_only=True).dropna(how="all")
    sub_h["PLANT_ID"] = plant_id
    sub_h = sub_h.reset_index()
    return sub_h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic fallback")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Generate small synthetic for smoke test")
    parser.add_argument("--plant-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    proc_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    long_df: pd.DataFrame | None = None
    source = "synthetic"

    if not args.synthetic:
        # Try Kaggle download if creds exist; else skip
        kj = Path.home() / ".kaggle" / "kaggle.json"
        if kj.exists():
            ok = try_kaggle(raw_dir)
            if ok:
                long_df = load_anikannal(raw_dir)
                if long_df is not None and len(long_df) > 0:
                    source = "kaggle_anikannal"
        else:
            print("[prepare_data] No ~/.kaggle/kaggle.json found — skipping Kaggle.")

    if long_df is None:
        print("[prepare_data] Generating synthetic fallback dataset.")
        days = 90 if args.quick else 730
        synth_path = raw_dir / "synthetic_long.parquet"
        long_df = make_synthetic_dataset(str(synth_path), days=days, seed=args.seed)

    hourly = hourly_aggregate(long_df, plant_id=args.plant_id)
    hourly.to_parquet(proc_dir / "raw_hourly.parquet", index=False)

    capacity_kw = float(np.percentile(hourly["AC_POWER"].dropna(), 99.9)) * 1.05
    capacity_kw = max(capacity_kw, 1.0)

    feats, feat_cols = build_features(
        hourly,
        target_col="AC_POWER",
        capacity_kw=capacity_kw,
        horizon=args.horizon,
    )
    if len(feats) < 200:
        raise RuntimeError(f"Too few rows after feature engineering: {len(feats)}")

    # Min-max normalize target so MAE is on the paper's normalized scale
    y_min = float(feats["AC_POWER"].min())
    y_max = float(feats["AC_POWER"].max())
    span = max(y_max - y_min, 1e-6)
    feats["target_kw"] = feats["target"]
    feats["target"] = (feats["target"] - y_min) / span

    splits = time_ordered_split(feats, train_frac=0.70, val_frac=0.15)

    feats.to_parquet(proc_dir / "features.parquet", index=False)
    manifest = {
        "source": source,
        "plant_id": args.plant_id,
        "horizon": args.horizon,
        "n_rows": int(len(feats)),
        "feature_cols": feat_cols,
        "capacity_kw": capacity_kw,
        "y_min_kw": y_min,
        "y_max_kw": y_max,
        "y_span_kw": span,
        "train_idx": [int(splits.train[0]), int(splits.train[-1] + 1)],
        "val_idx": [int(splits.val[0]), int(splits.val[-1] + 1)],
        "test_idx": [int(splits.test[0]), int(splits.test[-1] + 1)],
    }
    (proc_dir / "splits.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[prepare_data] source={source} rows={len(feats)} features={len(feat_cols)} "
        f"capacity_kw={capacity_kw:.1f}"
    )
    print(f"[prepare_data] train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")


if __name__ == "__main__":
    main()
