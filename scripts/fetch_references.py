"""Download SOTA paper PDFs to reference/ (best-effort)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


# Open-access direct PDF URLs. Some sites need user-agent header to allow scripted access.
PAPERS: list[dict] = [
    {
        "filename": "khan2024_x_lstm_eo.pdf",
        "title": "Khan et al. — Explainable AI and optimized solar power generation forecasting (X-LSTM-EO)",
        "url": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0308002&type=printable",
        "landing": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308002",
        "dataset": "anikannal Solar Power Generation Data (Plant 1)",
        "headline_metric": "MAE = 0.229 (normalized)",
        "tag": "PRIMARY SOTA TARGET",
    },
    {
        "filename": "solar_ml_comparison_plos.pdf",
        "title": "Solar energy prediction through ML: comparative regressor analysis",
        "url": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0315955&type=printable",
        "landing": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0315955",
        "dataset": "anikannal Solar Power Generation Data",
        "headline_metric": "see paper Table",
        "tag": "Same-dataset baselines",
    },
    {
        "filename": "stacking_metamodel_altitude.pdf",
        "title": "Optimizing photovoltaic power prediction at extreme altitudes using stacking metamodels and dimensionality reduction",
        "url": "https://www.nature.com/articles/s41598-025-22185-x.pdf",
        "landing": "https://www.nature.com/articles/s41598-025-22185-x",
        "dataset": "Bolivia high-altitude PV plant",
        "headline_metric": "MAE = 6.76 (kW), R^2 = 0.9999",
        "tag": "Stacking blueprint",
    },
    {
        "filename": "ultra_short_term_2509_17095.pdf",
        "title": "Ultra-short-term solar power forecasting by deep learning and data decomposition",
        "url": "https://arxiv.org/pdf/2509.17095",
        "landing": "https://arxiv.org/abs/2509.17095",
        "dataset": "Multiple PV plants",
        "headline_metric": "see paper",
        "tag": "Recent open-access reference",
    },
    {
        "filename": "france_renewables_benchmark.pdf",
        "title": "Towards Accurate Forecasting of Renewable Energy: France benchmarking",
        "url": "https://arxiv.org/pdf/2504.16100",
        "landing": "https://arxiv.org/abs/2504.16100",
        "dataset": "France solar + wind, 2012-2023 hourly",
        "headline_metric": "see paper",
        "tag": "Benchmarking framework",
    },
    {
        "filename": "ensemble_postprocessing_pv.pdf",
        "title": "Post-processing of ensemble photovoltaic power forecasts with distributional and quantile regression methods",
        "url": "https://arxiv.org/pdf/2508.15508",
        "landing": "https://arxiv.org/abs/2508.15508",
        "dataset": "Operational PV forecasts",
        "headline_metric": "see paper",
        "tag": "Post-processing reference",
    },
    {
        "filename": "lightgbm_medium_term_pv.pdf",
        "title": "LightGBM Medium-Term Photovoltaic Power Prediction Integrating Meteorological Features and Historical Data",
        "url": "https://www.mdpi.com/1996-1073/18/20/5526/pdf",
        "landing": "https://www.mdpi.com/1996-1073/18/20/5526",
        "dataset": "Medium-term PV plant",
        "headline_metric": "MAE = 37.49, R^2 = 0.89",
        "tag": "LightGBM-only PV baseline",
    },
    {
        "filename": "quantum_ml_pv_forecast.pdf",
        "title": "Photovoltaic power forecasting using quantum machine learning",
        "url": "https://arxiv.org/pdf/2312.16379",
        "landing": "https://arxiv.org/abs/2312.16379",
        "dataset": "PV plant",
        "headline_metric": "see paper",
        "tag": "Alternative model class",
    },
]


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def download(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        if not r.content[:4] == b"%PDF":
            return False
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  [WARN] {url} → {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", default=str(ROOT / "reference"))
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    success: list[dict] = []
    failed: list[dict] = []
    for p in PAPERS:
        dest = ref_dir / p["filename"]
        if dest.exists() and dest.stat().st_size > 50_000:
            print(f"[skip] {p['filename']} already exists.")
            success.append(p)
            continue
        print(f"[get ] {p['filename']} <- {p['url']}")
        ok = download(p["url"], dest)
        if ok:
            print(f"  OK  {dest.stat().st_size // 1024} KB")
            success.append(p)
        else:
            failed.append(p)

    # Index README
    lines = ["# SOTA Reference Library", "", "Papers we benchmark against. Numbers in `benchmarks.md` are paper-reported and never blended into our experimental results.", ""]
    if success:
        lines.append("## Downloaded")
        for p in success:
            lines.append(f"- **{p['filename']}** — {p['title']}  ")
            lines.append(f"  Dataset: {p['dataset']}  ")
            lines.append(f"  Reported: {p['headline_metric']}  ")
            lines.append(f"  Tag: _{p['tag']}_  ")
            lines.append(f"  Landing: <{p['landing']}>")
        lines.append("")
    if failed:
        lines.append("## Manual download required (link blocked or paywalled)")
        for p in failed:
            lines.append(f"- **{p['filename']}** — {p['title']}  ")
            lines.append(f"  Try: <{p['landing']}>  ")
            lines.append(f"  Then save the PDF to `reference/{p['filename']}`")
        lines.append("")
    (ref_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    # Benchmarks table
    bm = [
        "# Paper-reported benchmarks (READ-ONLY)",
        "",
        "Strictly separate from our `outputs/metrics.csv`. Do not blend.",
        "",
        "| Paper | Dataset | Headline metric | Tag |",
        "|---|---|---|---|",
    ]
    for p in PAPERS:
        bm.append(f"| {p['title']} | {p['dataset']} | {p['headline_metric']} | {p['tag']} |")
    (ref_dir / "benchmarks.md").write_text("\n".join(bm) + "\n", encoding="utf-8")

    print(f"[fetch_references] downloaded {len(success)}/{len(PAPERS)} PDFs to {ref_dir}")


if __name__ == "__main__":
    main()
