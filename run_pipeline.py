"""End-to-end orchestrator: prepare_data -> train -> stack -> evaluate -> explain."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}) after {dt:.1f}s: {' '.join(cmd)}")
    print(f"  done in {dt:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--skip-fetch", action="store_true", help="Skip reference PDF download step")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_fetch:
        try:
            run([py, "scripts/fetch_references.py"])
        except Exception as e:
            print(f"[run_pipeline] fetch_references degraded: {e}")

    prep_cmd = [py, "scripts/prepare_data.py", "--data-dir", args.data_dir, "--seed", str(args.seed)]
    if args.quick:
        prep_cmd.append("--quick")
    if args.synthetic:
        prep_cmd.append("--synthetic")
    run(prep_cmd)

    train_cmd = [
        py,
        "scripts/train.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        args.output_dir,
        "--seed",
        str(args.seed),
    ]
    if args.quick:
        train_cmd.append("--quick")
    if args.full:
        train_cmd.append("--full")
    run(train_cmd)

    run(
        [
            py,
            "scripts/stack.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
        ]
    )
    run(
        [
            py,
            "scripts/evaluate.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
        ]
    )
    run(
        [
            py,
            "scripts/explain.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
        ]
    )

    metrics = (Path(args.output_dir) / "metrics.csv").read_text()
    print("\n=== Final metrics.csv ===")
    print(metrics)
    print("=== Done ===")


if __name__ == "__main__":
    main()
