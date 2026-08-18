from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _load_json(path: Path) -> dict:
    import json

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Stage1 query-label noise and summarize robustness.")
    parser.add_argument("--config", required=True, help="Base config path.")
    parser.add_argument("--noise-levels", default="0.0,0.05,0.1,0.2,0.3", help="Comma-separated query noise ratios.")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds. Default uses config seed.")
    parser.add_argument("--out-root", default="", help="Override output root directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_cfg_path = Path(args.config).resolve()
    base_cfg = _load_yaml(base_cfg_path)
    base_out_dir = Path(str(base_cfg["project"]["out_dir"]))
    out_root = Path(args.out_root) if args.out_root else (base_out_dir.parent / f"{base_out_dir.name}_noise_robustness")
    generated_dir = out_root / "_generated_configs"
    pipeline_script = repo_root / "scripts" / "run_pipeline.py"

    noise_levels = [float(token.strip()) for token in args.noise_levels.split(",") if token.strip()]
    if args.seeds.strip():
        seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    else:
        seeds = [int(base_cfg["project"].get("seed", 42))]

    per_run_rows: list[dict[str, str]] = []
    for noise in noise_levels:
        for seed in seeds:
            cfg = deepcopy(base_cfg)
            cfg.setdefault("stage1", {})
            cfg["project"]["seed"] = seed
            cfg["stage1"]["query_label_noise"] = float(noise)
            noise_slug = str(noise).replace(".", "p")
            run_out_dir = out_root / f"noise_{noise_slug}" / f"seed_{seed}"
            cfg["project"]["out_dir"] = str(run_out_dir).replace("\\", "/")
            cfg_path = generated_dir / f"noise_{noise_slug}" / f"seed_{seed}.yaml"
            _write_yaml(cfg_path, cfg)

            print(f"[NoiseSuite] run noise={noise:.3f} seed={seed}")
            subprocess.run([sys.executable, str(pipeline_script), "--config", str(cfg_path)], check=True, cwd=str(repo_root))

            oracle_name = str(cfg.get("stage3", {}).get("oracle_name") or cfg.get("stage2", {}).get("oracle_name") or cfg["oracle_models"][0]["name"])
            stage1_metrics = _load_json(run_out_dir / "stage1" / oracle_name / "metrics.json")
            stage2_metrics = _load_json(run_out_dir / "stage2" / "metrics.json")
            stage3_metrics = _load_json(run_out_dir / "stage3" / "metrics.json")
            per_run_rows.append(
                {
                    "noise_ratio": f"{noise:.3f}",
                    "seed": str(seed),
                    "out_dir": str(run_out_dir),
                    "stage1_agreement": _fmt(stage1_metrics.get("agreement")),
                    "stage1_decision_score": _fmt(stage1_metrics.get("stage1_decision_score")),
                    "stage1_query_label_noise": _fmt(stage1_metrics.get("query_label_noise")),
                    "stage2_asr_oracle": _fmt(stage2_metrics.get("asr_oracle")),
                    "stage2_asr_surrogate": _fmt(stage2_metrics.get("asr_surrogate")),
                    "stage2_decision_score": _fmt(stage2_metrics.get("stage2_decision_score")),
                    "stage3_adv_benign_rate": _fmt(stage3_metrics.get("adv_benign_rate")),
                    "stage3_decision_score": _fmt(stage3_metrics.get("stage3_decision_score")),
                }
            )

    summary_path = out_root / "noise_robustness_runs.csv"
    if per_run_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_run_rows)

    agg_rows: list[dict[str, str]] = []
    for noise in noise_levels:
        noise_key = f"{noise:.3f}"
        group = [row for row in per_run_rows if row["noise_ratio"] == noise_key]
        if not group:
            continue
        agg_rows.append(
            {
                "noise_ratio": noise_key,
                "runs": str(len(group)),
                "stage1_agreement_mean": _fmt(_mean([float(r["stage1_agreement"]) for r in group if r["stage1_agreement"]])),
                "stage2_asr_oracle_mean": _fmt(_mean([float(r["stage2_asr_oracle"]) for r in group if r["stage2_asr_oracle"]])),
                "stage2_asr_surrogate_mean": _fmt(_mean([float(r["stage2_asr_surrogate"]) for r in group if r["stage2_asr_surrogate"]])),
                "stage3_adv_benign_rate_mean": _fmt(_mean([float(r["stage3_adv_benign_rate"]) for r in group if r["stage3_adv_benign_rate"]])),
                "stage3_decision_score_mean": _fmt(_mean([float(r["stage3_decision_score"]) for r in group if r["stage3_decision_score"]])),
            }
        )
    agg_path = out_root / "noise_robustness_summary.csv"
    if agg_rows:
        with agg_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)

    report_lines = [
        "# Noise Robustness Summary",
        "",
        "This sweep injects hard-label noise into Stage1 surrogate queries and records the downstream degradation in feature-space and remap-space attack performance.",
        "",
        "| Noise | Runs | Stage1 Agreement | Stage2 ASR(Oracle) | Stage2 ASR(Surrogate) | Stage3 Adv Benign Rate | Stage3 Decision |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in agg_rows:
        report_lines.append(
            f"| {row['noise_ratio']} | {row['runs']} | {row['stage1_agreement_mean']} | {row['stage2_asr_oracle_mean']} | {row['stage2_asr_surrogate_mean']} | {row['stage3_adv_benign_rate_mean']} | {row['stage3_decision_score_mean']} |"
        )
    report_lines.extend(
        [
            "",
            f"- Per-run CSV: `{summary_path}`",
            f"- Aggregated CSV: `{agg_path}`",
        ]
    )
    (out_root / "NOISE_ROBUSTNESS_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[NoiseSuite] runs {summary_path}")
    print(f"[NoiseSuite] summary {agg_path}")


if __name__ == "__main__":
    main()
