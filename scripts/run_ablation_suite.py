from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


VARIANTS: dict[str, dict] = {
    "full": {},
    "backbone_gan": {
        "stage2": {"generator_backbone": "gan"},
    },
    "w_o_stage1": {
        "stage1": {
            "extraction_mode": "baseline_only",
            "query_real_ratio": 0.0,
            "query_mix_ratio": 0.0,
            "real_warmup_steps": 0,
        },
        "stage2": {"surrogate_guidance_mode": "raw_only"},
    },
    "w_o_surrogate_embedding": {
        "stage2": {"surrogate_guidance_mode": "raw_only"},
    },
    "w_o_deep_semantics_logits": {
        "stage2": {"surrogate_guidance_mode": "logits"},
    },
    "w_o_deep_semantics_hard_label": {
        "stage2": {"surrogate_guidance_mode": "hard_label"},
    },
    "backbone_cgan": {
        "stage2": {"generator_backbone": "cgan"},
    },
    "backbone_wgan": {
        "stage2": {"generator_backbone": "wgan"},
    },
    "random_remap": {
        "stage3": {"remap_mode": "random"},
    },
    "w_o_protocol_projection": {
        "stage2": {
            "constraints": {"enable": False},
            "deployable_constraints": {"enable": False},
        },
        "stage3": {"protocol_auto_fix": False},
    },
    "w_o_payload_preservation": {
        "stage2": {"lambda_preserve": 0.0},
    },
    "w_o_auto_fix": {
        "stage3": {"protocol_auto_fix": False},
    },
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _deep_update(target: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_json(path: Path) -> dict:
    import json

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run built-in ablation variants from a base config.")
    parser.add_argument("--config", required=True, help="Base config path.")
    parser.add_argument(
        "--variants",
        default="full,w_o_stage1,backbone_gan,random_remap",
        help="Comma-separated variant names.",
    )
    parser.add_argument("--out-root", default="", help="Override output root directory.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_cfg_path = Path(args.config).resolve()
    base_cfg = _load_yaml(base_cfg_path)
    repo_root = Path(__file__).resolve().parents[1]
    base_out_dir = Path(str(base_cfg["project"]["out_dir"]))
    out_root = Path(args.out_root) if args.out_root else (base_out_dir.parent / f"{base_out_dir.name}_ablations")
    generated_dir = out_root / "_generated_configs"
    out_root.mkdir(parents=True, exist_ok=True)

    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in selected if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    summary_rows: list[dict[str, str]] = []
    for name in selected:
        cfg = deepcopy(base_cfg)
        _deep_update(cfg, deepcopy(VARIANTS[name]))
        cfg["project"]["out_dir"] = str(out_root / name)
        cfg_path = generated_dir / f"{name}.yaml"
        _write_yaml(cfg_path, cfg)

        metrics_path = repo_root / cfg["project"]["out_dir"] / "pipeline" / "summary_all_metrics.csv"
        if args.skip_existing and metrics_path.exists():
            print(f"[Ablation] skip existing {name}")
        else:
            print(f"[Ablation] run {name}")
            subprocess.run(
                [sys.executable, str(repo_root / "scripts" / "run_pipeline.py"), "--config", str(cfg_path)],
                check=True,
                cwd=str(repo_root),
            )

        stage2_metrics = _load_json(repo_root / cfg["project"]["out_dir"] / "stage2" / "metrics.json")
        stage3_metrics = _load_json(repo_root / cfg["project"]["out_dir"] / "stage3" / "metrics.json")
        summary_rows.append(
            {
                "variant": name,
                "out_dir": str(repo_root / cfg["project"]["out_dir"]),
                "generator_backbone": str(stage2_metrics.get("generator_backbone", cfg.get("stage2", {}).get("generator_backbone", ""))),
                "surrogate_guidance_mode": str(stage2_metrics.get("surrogate_guidance_mode", cfg.get("stage2", {}).get("surrogate_guidance_mode", ""))),
                "remap_mode": str(stage3_metrics.get("remap_mode", cfg.get("stage3", {}).get("remap_mode", ""))),
                "protocol_auto_fix": str(stage3_metrics.get("protocol_auto_fix", cfg.get("stage3", {}).get("protocol_auto_fix", ""))),
                "stage2_decision_score": str(stage2_metrics.get("stage2_decision_score", "")),
                "asr_oracle": str(stage2_metrics.get("asr_oracle", "")),
                "asr_surrogate": str(stage2_metrics.get("asr_surrogate", "")),
                "norm_FFD": str(stage2_metrics.get("norm_FFD", "")),
                "norm_SWD": str(stage2_metrics.get("norm_SWD", "")),
                "stage3_decision_score": str(stage3_metrics.get("stage3_decision_score", "")),
                "stage3_deployability_score": str(stage3_metrics.get("stage3_decision_pcap_deployability_score", "")),
                "stage3_remap_quality_score": str(stage3_metrics.get("stage3_decision_remap_quality_score", "")),
                "pcap_valid_fatal_rate": str(stage3_metrics.get("pcap_valid_fatal_rate", "")),
                "pcap_validfatal_at_0": str(stage3_metrics.get("pcap_validfatal_at_0", "")),
                "pcap_skip_reason": str(stage3_metrics.get("pcap_skip_reason", "")),
            }
        )

    summary_path = out_root / "ablation_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["variant"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Ablation] summary {summary_path}")


if __name__ == "__main__":
    main()
