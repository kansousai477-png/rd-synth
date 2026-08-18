from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _bootstrap import config_path_from_env

from rdsynth.data.csv_datasets import resolve_dataset_profile
from rdsynth.pipeline.reporting import load_json
from rdsynth.utils.pipeline_config import resolve_oracle_name

ABLATION_VARIANTS: dict[str, dict] = {
    "full": {},
    "backbone_gan": {
        "stage2": {"generator_backbone": "gan"},
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


DATASET_PRESETS: dict[str, dict[str, object]] = {
    "nb15": {
        "display_name": "CIC NB15",
        "log_tag": "NB15",
        "config": "configs/nb15_attack_sweep.yaml",
        "default_attacks": [
            "Exploits",
            "Fuzzers",
            "Reconnaissance",
            "DoS",
            "Generic",
            "Shellcode",
            "Worms",
            "Backdoor",
            "Analysis",
        ],
    },
    "2018": {
        "display_name": "CIC-IDS2018",
        "log_tag": "IDS2018",
        "config": "configs/cic_ids2018_attack_sweep.yaml",
        "default_attacks": [
            "DDOS attack-HOIC",
            "DDoS attacks-LOIC-HTTP",
            "DDOS attack-LOIC-UDP",
            "DoS attacks-Hulk",
            "DoS attacks-GoldenEye",
            "DoS attacks-Slowloris",
            "DoS attacks-SlowHTTPTest",
            "Bot",
            "FTP-BruteForce",
            "SSH-Bruteforce",
            "Brute Force -Web",
            "Brute Force -XSS",
            "SQL Injection",
            "Infilteration",
        ],
    },
    "2017": {
        "display_name": "CIC-IDS2017",
        "log_tag": "IDS2017",
        "config": "configs/cic_ids2017_attack_sweep.yaml",
        "default_attacks": [
            "DDoS",
            "PortScan",
            "Bot",
            "DoS Hulk",
            "DoS GoldenEye",
            "DoS slowloris",
            "DoS Slowhttptest",
            "FTP-Patator",
            "SSH-Patator",
            "Web Attack - Brute Force",
            "Web Attack - XSS",
            "Web Attack - Sql Injection",
            "Infiltration",
            "Heartbleed",
        ],
    },
    "iot23": {
        "display_name": "CIC IoT-2023",
        "log_tag": "IoT2023",
        "config": "configs/iot23_attack_sweep.yaml",
        "default_attacks": [
            "DDoS-SYN_Flood",
            "DDoS-HTTP_Flood",
            "DDoS-TCP_Flood",
            "DDoS-UDP_Flood",
            "DoS-TCP_Flood",
            "Recon-PortScan",
            "Mirai-udpplain",
        ],
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


def _load_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = {}
        for row in csv.DictReader(f):
            key = row.get("attack_variant") or row.get("attack_type")
            if key:
                rows[key] = row
        return rows


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _bootstrap_ci(values: list[float], *, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float | None, float | None]:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return None, None
    if vals.size == 1:
        v = float(vals[0])
        return v, v
    rng = np.random.default_rng(42)
    samples = rng.choice(vals, size=(n_boot, vals.size), replace=True)
    means = np.mean(samples, axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _paired_permutation_pvalue(a: list[float], b: list[float], *, n_perm: int = 5000) -> float | None:
    if len(a) != len(b) or len(a) == 0:
        return None
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if not np.all(np.isfinite(da)) or not np.all(np.isfinite(db)):
        return None
    diff = da - db
    obs = float(abs(np.mean(diff)))
    if obs <= 1.0e-15:
        return 1.0
    rng = np.random.default_rng(123)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_perm, diff.size), replace=True)
    perm = np.abs(np.mean(signs * diff[None, :], axis=1))
    return float((np.sum(perm >= obs) + 1.0) / (n_perm + 1.0))


def _mean(rows: list[dict[str, str]], key: str) -> float | None:
    vals = [_to_float(row.get(key)) for row in rows]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _first_available(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key, "")
        if str(value).strip():
            return str(value)
    return ""


def _first_available_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _mean_any(rows: list[dict[str, str]], keys: tuple[str, ...]) -> float | None:
    vals = [_first_available_float(row, keys) for row in rows]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _conflict_score(row: dict[str, str]) -> float | None:
    deploy = _first_available_float(row, ("stage3_deployability_score", "stage3_adv_benign_rate"))
    remap = _first_available_float(row, ("stage3_remap_quality_score", "stage3_remap_r2"))
    target_l2 = _to_float(row.get("stage3_pcap_target_l2_mean"))
    target_mae = _to_float(row.get("stage3_pcap_target_mae_mean"))
    align = _to_float(row.get("stage3_pcap_alignment_coverage"))
    missing = _to_float(row.get("stage3_pcap_alignment_missing"))
    if deploy is None and remap is None and target_l2 is None and align is None:
        return None
    score = 0.0
    if remap is not None and deploy is not None:
        score += max(0.0, remap - deploy)
    if target_l2 is not None:
        score += min(target_l2 / 25.0, 2.0)
    if target_mae is not None:
        score += min(target_mae / 2.0, 2.0)
    if align is not None:
        score += max(0.0, 1.0 - align)
    if missing is not None:
        score += min(missing / 10.0, 1.0)
    return score


def _failure_reason(row: dict[str, str]) -> str:
    reasons: list[str] = []
    remap_mode = str(row.get("stage3_remap_mod_source", "")).strip()
    if remap_mode in {"direct", "blended"}:
        reasons.append(f"remap fell back to {remap_mode}")
    skip_reason = str(row.get("stage3_pcap_skip_reason", "")).strip()
    if skip_reason:
        reasons.append(f"pcap skipped: {skip_reason}")
    align = _to_float(row.get("stage3_pcap_alignment_coverage"))
    if align is not None and align < 0.95:
        reasons.append(f"alignment coverage dropped to {align:.3f}")
    target_l2 = _to_float(row.get("stage3_pcap_target_l2_mean"))
    if target_l2 is not None and target_l2 > 10.0:
        reasons.append(f"feature target mismatch remained high (L2={target_l2:.3f})")
    collapse = _to_float(row.get("stage3_remap_collapse_ratio"))
    if collapse is not None and collapse < 0.5:
        reasons.append(f"learned remap variance collapsed (ratio={collapse:.3f})")
    missing = _to_float(row.get("stage3_pcap_alignment_missing"))
    if missing is not None and missing > 0:
        reasons.append(f"missing aligned features={int(missing)}")
    if not reasons:
        deploy = _first_available_float(row, ("stage3_deployability_score",))
        remap = _first_available_float(row, ("stage3_remap_quality_score",))
        if deploy is not None and remap is not None and remap > deploy:
            reasons.append("remap fit remained high but deployment score still lagged")
    return "; ".join(reasons) if reasons else "no dominant failure signature extracted"


def _family(name: str) -> str:
    for token in (" - ", "-", "_", " "):
        if token in name:
            return name.split(token, 1)[0].strip()
    return name.strip()


def _write_family_summary(path: Path, rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        attack = row.get("attack_type", "")
        if not attack:
            continue
        family = _family(attack)
        variant = str(row.get("variant", "")).strip()
        group_name = f"{variant}:{family}" if variant else family
        grouped.setdefault(group_name, []).append(row)

    summary_rows: list[dict[str, str]] = []
    for family in sorted(grouped):
        group = grouped[family]
        record: dict[str, str] = {"family": family, "attacks": str(len(group))}
        for key in (
            "variant",
            "stage1_decision_score",
            "stage2_decision_score",
            "stage2_attack_score",
            "stage2_fidelity_score",
            "stage2_constraint_score",
            "stage3_decision_score",
            "stage3_deployability_score",
            "stage3_remap_quality_score",
        ):
            if key == "variant":
                vals = [str(row.get("variant", "")).strip() for row in group]
                record[key] = vals[0] if vals else ""
                continue
            vals = [_to_float(row.get(key)) for row in group]
            vals = [value for value in vals if value is not None]
            record[key] = _fmt_float(sum(vals) / len(vals)) if vals else ""
        summary_rows.append(record)

    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["family", "attacks"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _aggregate_attack_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metric_keys = [
        "stage1_oracle_acc",
        "stage1_oracle_f1",
        "stage1_agreement",
        "stage1_baseline_agreement",
        "stage1_decision_score",
        "stage2_decision_score",
        "stage2_attack_score",
        "stage2_fidelity_score",
        "stage2_constraint_score",
        "stage2_norm_ffd",
        "stage2_norm_swd",
        "stage3_decision_score",
        "stage3_deployability_score",
        "stage3_remap_quality_score",
        "stage3_remap_r2",
        "stage3_remap_port_acc",
        "stage3_pcap_alignment_coverage",
        "stage3_pcap_target_l2_mean",
        "stage3_pcap_target_mae_mean",
        "stage3_pcap_valid_fatal_rate",
        "stage2_asr_oracle",
        "stage2_asr_surrogate",
        "stage2_adv_pmal_oracle",
        "stage2_train_best_epoch",
        "stage2_train_best_score",
        "stage3_remap_best_epoch",
        "stage3_remap_best_score",
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("variant", "")), str(row.get("attack_type", "")))
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, str]] = []
    for (variant, attack), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        record: dict[str, str] = {
            "variant": variant,
            "attack_type": attack,
            "attack_variant": f"{attack}::{variant}",
            "n_runs": str(len(group)),
            "seeds": ",".join(sorted({str(r.get("seed", "")) for r in group if str(r.get("seed", "")).strip()})),
            "out_dir": str(group[0].get("out_dir", "")),
            "stage2_generator_backbone": str(group[0].get("stage2_generator_backbone", "")),
            "stage2_surrogate_guidance_mode": str(group[0].get("stage2_surrogate_guidance_mode", "")),
            "stage3_remap_mode": str(group[0].get("stage3_remap_mode", "")),
            "stage3_protocol_auto_fix": str(group[0].get("stage3_protocol_auto_fix", "")),
        }
        for metric in metric_keys:
            vals = [_to_float(row.get(metric)) for row in group]
            vals = [float(v) for v in vals if v is not None]
            mean_val = float(np.mean(vals)) if vals else None
            ci_lo, ci_hi = _bootstrap_ci(vals) if vals else (None, None)
            record[metric] = _fmt_float(mean_val)
            record[f"{metric}_ci95_lo"] = _fmt_float(ci_lo)
            record[f"{metric}_ci95_hi"] = _fmt_float(ci_hi)
            record[f"{metric}_std"] = _fmt_float(float(np.std(vals, ddof=1))) if len(vals) > 1 else ""
        out_rows.append(record)
    return out_rows


def _write_significance_summary(path: Path, rows: list[dict[str, str]]) -> None:
    by_variant_attack_seed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("variant", "")), str(row.get("attack_type", "")), str(row.get("seed", "")))
        by_variant_attack_seed[key] = row

    variants = sorted({str(r.get("variant", "")) for r in rows})
    if "full" not in variants:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["variant", "metric", "n_pairs", "mean_delta_vs_full", "p_value"])
            writer.writeheader()
        return

    metrics = ["stage2_decision_score", "stage3_decision_score", "stage3_deployability_score"]
    records: list[dict[str, str]] = []
    for variant in variants:
        if variant == "full":
            continue
        for metric in metrics:
            deltas: list[float] = []
            for (_, attack, seed), row in by_variant_attack_seed.items():
                if row.get("variant") != variant:
                    continue
                base = by_variant_attack_seed.get(("full", attack, seed))
                if base is None:
                    continue
                v = _to_float(row.get(metric))
                b = _to_float(base.get(metric))
                if v is None or b is None:
                    continue
                deltas.append(float(v - b))
            p = _paired_permutation_pvalue(deltas, [0.0 for _ in deltas]) if deltas else None
            records.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n_pairs": str(len(deltas)),
                    "mean_delta_vs_full": _fmt_float(float(np.mean(deltas)) if deltas else None),
                    "p_value": _fmt_float(p),
                }
            )
    fieldnames = ["variant", "metric", "n_pairs", "mean_delta_vs_full", "p_value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _make_paper_stage2_table_agg_rows(agg_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in agg_rows:
        method = str(row.get("attack_variant", "main"))
        rows.append(
            {
                "method": method,
                "family": "ours",
                "feature_space": "true",
                "traffic_space": "true",
                "n_runs": str(row.get("n_runs", "")),
                "seeds": str(row.get("seeds", "")),
                "asr_oracle": str(row.get("stage2_asr_oracle", "")),
                "asr_oracle_ci95": f"[{row.get('stage2_asr_oracle_ci95_lo', '')}, {row.get('stage2_asr_oracle_ci95_hi', '')}]",
                "asr_surrogate": str(row.get("stage2_asr_surrogate", "")),
                "asr_surrogate_ci95": f"[{row.get('stage2_asr_surrogate_ci95_lo', '')}, {row.get('stage2_asr_surrogate_ci95_hi', '')}]",
                "adv_pmal_oracle": str(row.get("stage2_adv_pmal_oracle", "")),
                "adv_pmal_oracle_ci95": f"[{row.get('stage2_adv_pmal_oracle_ci95_lo', '')}, {row.get('stage2_adv_pmal_oracle_ci95_hi', '')}]",
                "decision_score": str(row.get("stage2_decision_score", "")),
                "decision_score_ci95": f"[{row.get('stage2_decision_score_ci95_lo', '')}, {row.get('stage2_decision_score_ci95_hi', '')}]",
                "attack_score": str(row.get("stage2_attack_score", "")),
                "attack_score_ci95": f"[{row.get('stage2_attack_score_ci95_lo', '')}, {row.get('stage2_attack_score_ci95_hi', '')}]",
                "fidelity_score": str(row.get("stage2_fidelity_score", "")),
                "fidelity_score_ci95": f"[{row.get('stage2_fidelity_score_ci95_lo', '')}, {row.get('stage2_fidelity_score_ci95_hi', '')}]",
                "constraint_score": str(row.get("stage2_constraint_score", "")),
                "constraint_score_ci95": f"[{row.get('stage2_constraint_score_ci95_lo', '')}, {row.get('stage2_constraint_score_ci95_hi', '')}]",
            }
        )
    return rows


def _make_paper_stage3_table_agg_rows(agg_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in agg_rows:
        method = str(row.get("attack_variant", "main"))
        rows.append(
            {
                "method": method,
                "family": "ours",
                "n_runs": str(row.get("n_runs", "")),
                "seeds": str(row.get("seeds", "")),
                "decision_score": str(row.get("stage3_decision_score", "")),
                "decision_score_ci95": f"[{row.get('stage3_decision_score_ci95_lo', '')}, {row.get('stage3_decision_score_ci95_hi', '')}]",
                "deployability_score": str(row.get("stage3_deployability_score", "")),
                "deployability_score_ci95": f"[{row.get('stage3_deployability_score_ci95_lo', '')}, {row.get('stage3_deployability_score_ci95_hi', '')}]",
                "pcap_alignment_coverage": str(row.get("stage3_pcap_alignment_coverage", "")),
                "pcap_alignment_coverage_ci95": f"[{row.get('stage3_pcap_alignment_coverage_ci95_lo', '')}, {row.get('stage3_pcap_alignment_coverage_ci95_hi', '')}]",
                "pcap_target_l2_mean": str(row.get("stage3_pcap_target_l2_mean", "")),
                "pcap_target_l2_mean_ci95": f"[{row.get('stage3_pcap_target_l2_mean_ci95_lo', '')}, {row.get('stage3_pcap_target_l2_mean_ci95_hi', '')}]",
                "pcap_target_mae_mean": str(row.get("stage3_pcap_target_mae_mean", "")),
                "pcap_target_mae_mean_ci95": f"[{row.get('stage3_pcap_target_mae_mean_ci95_lo', '')}, {row.get('stage3_pcap_target_mae_mean_ci95_hi', '')}]",
            }
        )
    return rows


RQ_QUESTIONS: list[tuple[str, str]] = [
    ("RQ1", "How effective is the surrogate recovered under black-box interaction?"),
    ("RQ2", "Does diffusion training mitigate instability observed in other traffic generators?"),
    ("RQ3", "How valid are the features generated by RD-Synth?"),
    ("RQ4", "How evasive are the adversarial features generated by RD-Synth?"),
    ("RQ5", "How realistic are the remapped packets?"),
    ("RQ6", "Online validation is intentionally excluded from this offline auto-report."),
]


def _rows_for_variant(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("variant", "")).strip() == variant]


def _primary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    full_rows = _rows_for_variant(rows, "full")
    if full_rows:
        return full_rows
    variants = sorted({str(row.get("variant", "")).strip() for row in rows if str(row.get("variant", "")).strip()})
    if not variants:
        return rows
    return _rows_for_variant(rows, variants[0])


def _metric_mean(rows: list[dict[str, str]], key: str) -> float | None:
    vals = [_to_float(row.get(key)) for row in rows]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _top_bottom_rows(rows: list[dict[str, str]], key: str, limit: int = 5) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    valid = [row for row in rows if _to_float(row.get(key)) is not None]
    top = sorted(valid, key=lambda row: _to_float(row.get(key)) or float("-inf"), reverse=True)[:limit]
    bottom = sorted(valid, key=lambda row: _to_float(row.get(key)) or float("inf"))[:limit]
    return top, bottom


def _variant_table_rows(rows: list[dict[str, str]], variants: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for variant in variants:
        subset = _rows_for_variant(rows, variant)
        out.append(
            {
                "variant": variant,
                "n_attacks": str(len(subset)),
                "stage2_decision_score": _fmt_float(_metric_mean(subset, "stage2_decision_score")),
                "stage2_fidelity_score": _fmt_float(_metric_mean(subset, "stage2_fidelity_score")),
                "stage2_attack_score": _fmt_float(_metric_mean(subset, "stage2_attack_score")),
                "stage3_decision_score": _fmt_float(_metric_mean(subset, "stage3_decision_score")),
                "stage3_deployability_score": _fmt_float(_metric_mean(subset, "stage3_deployability_score")),
                "stage3_target_l2": _fmt_float(_metric_mean(subset, "stage3_pcap_target_l2_mean")),
            }
        )
    return out


def _rq_coverage_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    variants = {str(row.get("variant", "")).strip() for row in rows if str(row.get("variant", "")).strip()}
    primary = _primary_rows(rows)
    return [
        {
            "rq_id": "RQ1",
            "question": RQ_QUESTIONS[0][1],
            "coverage": "covered" if _metric_mean(primary, "stage1_agreement") is not None else "missing",
            "evidence": "stage1_agreement, stage1_baseline_agreement, stage1_decision_score",
        },
        {
            "rq_id": "RQ2",
            "question": RQ_QUESTIONS[1][1],
            "coverage": "covered" if {"backbone_wgan", "backbone_cgan"} & variants else "partial",
            "evidence": "variant comparison over backbone_wgan/backbone_cgan/full",
        },
        {
            "rq_id": "RQ3",
            "question": RQ_QUESTIONS[2][1],
            "coverage": "covered" if _metric_mean(primary, "stage2_fidelity_score") is not None else "partial",
            "evidence": "stage2_fidelity_score, stage2_norm_ffd, stage2_norm_swd",
        },
        {
            "rq_id": "RQ4",
            "question": RQ_QUESTIONS[3][1],
            "coverage": "covered" if _metric_mean(primary, "stage2_attack_score") is not None else "partial",
            "evidence": "stage2_attack_score, stage2_asr_oracle, stage2_adv_pmal_oracle",
        },
        {
            "rq_id": "RQ5",
            "question": RQ_QUESTIONS[4][1],
            "coverage": "covered" if _metric_mean(primary, "stage3_remap_quality_score") is not None else "partial",
            "evidence": "stage3_remap_quality_score, stage3_pcap_alignment_coverage, stage3_pcap_target_l2_mean",
        },
        {
            "rq_id": "RQ6",
            "question": RQ_QUESTIONS[5][1],
            "coverage": "skipped_by_design",
            "evidence": "explicitly excluded by current offline automation scope",
        },
    ]


def _rq_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    primary = _primary_rows(rows)
    variants = {str(row.get("variant", "")).strip() for row in rows if str(row.get("variant", "")).strip()}
    r1_gain = None
    agreement = _metric_mean(primary, "stage1_agreement")
    baseline_agreement = _metric_mean(primary, "stage1_baseline_agreement")
    if agreement is not None and baseline_agreement is not None:
        r1_gain = agreement - baseline_agreement
    r2_note = (
        "Compared against backbone variants in the same sweep."
        if {"backbone_wgan", "backbone_cgan"} & variants
        else "Backbone ablations absent in this run; only diffusion-main evidence is available."
    )
    return [
        {
            "rq_id": "RQ1",
            "question": RQ_QUESTIONS[0][1],
            "primary_metric": "stage1_agreement_gain",
            "value": _fmt_float(r1_gain),
            "supporting_metrics": f"agreement={_fmt_float(agreement)}, baseline={_fmt_float(baseline_agreement)}, score={_fmt_float(_metric_mean(primary, 'stage1_decision_score'))}",
        },
        {
            "rq_id": "RQ2",
            "question": RQ_QUESTIONS[1][1],
            "primary_metric": "stage2_decision_score(full)",
            "value": _fmt_float(_metric_mean(_rows_for_variant(rows, 'full') or primary, 'stage2_decision_score')),
            "supporting_metrics": r2_note,
        },
        {
            "rq_id": "RQ3",
            "question": RQ_QUESTIONS[2][1],
            "primary_metric": "stage2_fidelity_score",
            "value": _fmt_float(_metric_mean(primary, 'stage2_fidelity_score')),
            "supporting_metrics": f"FFD={_fmt_float(_metric_mean(primary, 'stage2_norm_ffd'))}, SWD={_fmt_float(_metric_mean(primary, 'stage2_norm_swd'))}",
        },
        {
            "rq_id": "RQ4",
            "question": RQ_QUESTIONS[3][1],
            "primary_metric": "stage2_attack_score",
            "value": _fmt_float(_metric_mean(primary, 'stage2_attack_score')),
            "supporting_metrics": f"ASR={_fmt_float(_metric_mean(primary, 'stage2_asr_oracle'))}, adv_pmal={_fmt_float(_metric_mean(primary, 'stage2_adv_pmal_oracle'))}",
        },
        {
            "rq_id": "RQ5",
            "question": RQ_QUESTIONS[4][1],
            "primary_metric": "stage3_remap_quality_score",
            "value": _fmt_float(_metric_mean(primary, 'stage3_remap_quality_score')),
            "supporting_metrics": f"align={_fmt_float(_metric_mean(primary, 'stage3_pcap_alignment_coverage'))}, target_l2={_fmt_float(_metric_mean(primary, 'stage3_pcap_target_l2_mean'))}, fatal={_fmt_float(_metric_mean(primary, 'stage3_pcap_valid_fatal_rate'))}",
        },
    ]


def _write_markdown_report(
    path: Path,
    rows: list[dict[str, str]],
    family_path: Path,
    dataset_name: str,
    out_root: Path,
) -> None:
    s1_keys = ("stage1_decision_score", "stage1_agreement", "stage1_oracle_acc")
    s2_keys = ("stage2_decision_score", "stage2_asr_oracle")
    s2_attack_keys = ("stage2_attack_score", "stage2_asr_oracle")
    s2_fidelity_keys = ("stage2_fidelity_score", "stage2_norm_ffd")
    s2_constraint_keys = ("stage2_constraint_score", "stage2_norm_swd")
    s3_keys = ("stage3_decision_score", "stage3_adv_benign_rate")
    s3_deploy_keys = ("stage3_deployability_score", "stage3_adv_benign_rate")
    s3_remap_keys = ("stage3_remap_quality_score", "stage3_remap_r2")

    def _label(row: dict[str, str]) -> str:
        return str(row.get("attack_variant") or row.get("attack_type", ""))

    ordered = sorted(
        rows,
        key=lambda row: (
            _first_available_float(row, s2_keys) or float("-inf"),
            _first_available_float(row, s3_keys) or float("-inf"),
        ),
        reverse=True,
    )
    top_rows = ordered[:10]
    hard_rows = sorted(rows, key=lambda row: _first_available_float(row, s2_keys) or float("inf"))[:8]
    stage3_hard_rows = sorted(rows, key=lambda row: _first_available_float(row, s3_keys) or float("inf"))[:8]
    failure_rows = sorted(
        rows,
        key=lambda row: _conflict_score(row) if _conflict_score(row) is not None else float("-inf"),
        reverse=True,
    )[:5]
    family_rows: list[dict[str, str]] = []
    if family_path.exists():
        with family_path.open("r", newline="", encoding="utf-8") as f:
            family_rows = list(csv.DictReader(f))
    top_family_rows = sorted(
        family_rows,
        key=lambda row: _first_available_float(row, ("stage2_decision_score", "avg_stage2_asr_oracle")) or float("-inf"),
        reverse=True,
    )[:8]

    s1_mean = _mean_any(rows, s1_keys)
    s2_mean = _mean_any(rows, s2_keys)
    s2_attack_mean = _mean_any(rows, s2_attack_keys)
    s2_fidelity_mean = _mean_any(rows, s2_fidelity_keys)
    s2_constraint_mean = _mean_any(rows, s2_constraint_keys)
    s3_mean = _mean_any(rows, s3_keys)
    s3_deploy_mean = _mean_any(rows, s3_deploy_keys)
    s3_remap_mean = _mean_any(rows, s3_remap_keys)
    align_mean = _mean(rows, "stage3_pcap_alignment_coverage")
    target_l2_mean = _mean(rows, "stage3_pcap_target_l2_mean")
    target_mae_mean = _mean(rows, "stage3_pcap_target_mae_mean")
    primary_rows = _primary_rows(rows)
    rq_coverage = _rq_coverage_rows(rows)
    rq_summary = _rq_summary_rows(rows)
    rq1_top, rq1_bottom = _top_bottom_rows(primary_rows, "stage1_agreement")
    rq3_top, rq3_bottom = _top_bottom_rows(primary_rows, "stage2_fidelity_score")
    rq4_top, rq4_bottom = _top_bottom_rows(primary_rows, "stage2_attack_score")
    rq5_top, rq5_bottom = _top_bottom_rows(primary_rows, "stage3_remap_quality_score")
    backbone_variants = [variant for variant in ["full", "backbone_wgan", "backbone_cgan"] if _rows_for_variant(rows, variant)]
    backbone_rows = _variant_table_rows(rows, backbone_variants)

    strongest_attack = _label(top_rows[0]) if top_rows else ""
    weakest_attack = _label(hard_rows[0]) if hard_rows else ""
    weakest_stage3_attack = _label(stage3_hard_rows[0]) if stage3_hard_rows else ""

    lines = [
        f"# {dataset_name} Attack Sweep Report",
        "",
        "## Experimental Setup",
        "",
        f"- Number of attack types: `{len(rows)}`",
        f"- Raw per-run summary: `{path.parent / 'attack_sweep_summary.csv'}`",
        f"- Aggregated summary (mean/CI): `{path.parent / 'attack_sweep_summary_agg.csv'}`",
        f"- Paper Stage2 table (mean/CI): `{path.parent / 'paper_stage2_table.csv'}`",
        f"- Paper Stage3 table (mean/CI): `{path.parent / 'paper_stage3_pcap_table.csv'}`",
        f"- Significance summary: `{path.parent / 'significance_summary.csv'}`",
        f"- Family summary: `{family_path}`",
        "",
        "This report treats the three-stage pipeline as a chained decision process:",
        "",
        "- `Stage1`: surrogate extraction quality",
        "- `Stage2`: feature-space attack quality",
        "- `Stage3`: remap and deployment quality",
        "",
        "If decision-score columns are unavailable in the current summary file, this report falls back to legacy proxies such as `agreement`, `ASR`, `FFD`, `SWD`, `adv_benign_rate`, and remap `R2`.",
        "",
        "## RQ Coverage",
        "",
        "| RQ | Coverage | Question | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in rq_coverage:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rq_id", "")),
                    str(row.get("coverage", "")),
                    str(row.get("question", "")),
                    str(row.get("evidence", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## RQ-Oriented Discussion",
            "",
            "| RQ | Primary Metric | Value | Supporting Metrics / Notes |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in rq_summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rq_id", "")),
                    str(row.get("primary_metric", "")),
                    str(row.get("value", "")),
                    str(row.get("supporting_metrics", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### RQ1. Surrogate Extraction Effectiveness",
            "",
            f"- Primary scope: variant `{primary_rows[0].get('variant', '') if primary_rows else ''}`." if primary_rows else "- No primary variant rows available.",
            f"- Mean surrogate agreement is `{_fmt_float(_metric_mean(primary_rows, 'stage1_agreement'))}`; baseline agreement is `{_fmt_float(_metric_mean(primary_rows, 'stage1_baseline_agreement'))}`; gain is `{_fmt_float((_metric_mean(primary_rows, 'stage1_agreement') or 0.0) - (_metric_mean(primary_rows, 'stage1_baseline_agreement') or 0.0) if _metric_mean(primary_rows, 'stage1_agreement') is not None and _metric_mean(primary_rows, 'stage1_baseline_agreement') is not None else None)}`.",
            "- This RQ should be interpreted through agreement gain first, then the stage1 decision score as the calibrated aggregate.",
            "",
            "| Best extraction cases | Agreement | Decision score |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in rq1_top[:5]:
        lines.append(f"| {_label(row)} | {_first_available(row, ('stage1_agreement',))} | {_first_available(row, ('stage1_decision_score',))} |")
    lines.extend(
        [
            "",
            "| Hard extraction cases | Agreement | Decision score |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in rq1_bottom[:5]:
        lines.append(f"| {_label(row)} | {_first_available(row, ('stage1_agreement',))} | {_first_available(row, ('stage1_decision_score',))} |")
    lines.extend(
        [
            "",
            "### RQ2. Diffusion Training Stability",
            "",
        ]
    )
    if backbone_rows:
        lines.extend(
            [
                "Backbone ablation evidence available in this sweep:",
                "",
                "| Variant | Attacks | Stage2 score | Stage2 fidelity | Stage2 attack | Stage3 score | Stage3 deploy | Target L2 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in backbone_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("variant", "")),
                        str(row.get("n_attacks", "")),
                        str(row.get("stage2_decision_score", "")),
                        str(row.get("stage2_fidelity_score", "")),
                        str(row.get("stage2_attack_score", "")),
                        str(row.get("stage3_decision_score", "")),
                        str(row.get("stage3_deployability_score", "")),
                        str(row.get("stage3_target_l2", "")),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "- Use this table to answer RQ2 directly: if `full` remains stronger or more stable than `backbone_wgan` / `backbone_cgan`, the evidence supports the diffusion backbone choice.",
            ]
        )
    else:
        lines.append("- This run does not include `backbone_wgan` / `backbone_cgan`, so RQ2 cannot be fully answered from ablation evidence here.")
    lines.extend(
        [
            "",
            "### RQ3. Feature Validity",
            "",
            f"- Mean fidelity score is `{_fmt_float(_metric_mean(primary_rows, 'stage2_fidelity_score'))}` with FFD `{_fmt_float(_metric_mean(primary_rows, 'stage2_norm_ffd'))}` and SWD `{_fmt_float(_metric_mean(primary_rows, 'stage2_norm_swd'))}`.",
            "- This RQ asks whether generated features stay statistically plausible rather than merely whether they fool the detector.",
            "",
            "| Most valid feature cases | Fidelity score | FFD | SWD |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rq3_top[:5]:
        lines.append(f"| {_label(row)} | {_first_available(row, ('stage2_fidelity_score',))} | {_first_available(row, ('stage2_norm_ffd',))} | {_first_available(row, ('stage2_norm_swd',))} |")
    lines.extend(
        [
            "",
            "| Least valid feature cases | Fidelity score | FFD | SWD |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rq3_bottom[:5]:
        lines.append(f"| {_label(row)} | {_first_available(row, ('stage2_fidelity_score',))} | {_first_available(row, ('stage2_norm_ffd',))} | {_first_available(row, ('stage2_norm_swd',))} |")
    lines.extend(
        [
            "",
            "### RQ4. Feature-Space Evasiveness",
            "",
            f"- Mean attack score is `{_fmt_float(_metric_mean(primary_rows, 'stage2_attack_score'))}`; oracle ASR is `{_fmt_float(_metric_mean(primary_rows, 'stage2_asr_oracle'))}`; adversarial malicious probability is `{_fmt_float(_metric_mean(primary_rows, 'stage2_adv_pmal_oracle'))}`.",
            "- This RQ should be read together with RQ3: high evasiveness with poor fidelity is not the same claim as high evasiveness with valid feature geometry.",
            "",
            "| Most evasive feature cases | Attack score | ASR | Adv malicious prob. |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rq4_top[:5]:
        lines.append(f"| {_label(row)} | {_first_available(row, ('stage2_attack_score',))} | {_first_available(row, ('stage2_asr_oracle',))} | {_first_available(row, ('stage2_adv_pmal_oracle',))} |")
    lines.extend(
        [
            "",
            "| Hard evasive cases | Attack score | ASR | Adv malicious prob. |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rq4_bottom[:5]:
        lines.append(f"| {_label(row)} | {_first_available(row, ('stage2_attack_score',))} | {_first_available(row, ('stage2_asr_oracle',))} | {_first_available(row, ('stage2_adv_pmal_oracle',))} |")
    lines.extend(
        [
            "",
            "### RQ5. Packet Realism After Remapping",
            "",
            f"- Mean remap-quality score is `{_fmt_float(_metric_mean(primary_rows, 'stage3_remap_quality_score'))}`; alignment coverage is `{_fmt_float(_metric_mean(primary_rows, 'stage3_pcap_alignment_coverage'))}`; target L2 is `{_fmt_float(_metric_mean(primary_rows, 'stage3_pcap_target_l2_mean'))}`; fatal-rate is `{_fmt_float(_metric_mean(primary_rows, 'stage3_pcap_valid_fatal_rate'))}`.",
            "- This RQ is about whether the feature target survives packet-space realization without collapsing into protocol-invalid or heavily mismatched traffic.",
            "",
            "| Most realistic remap cases | Remap score | Align Cov. | Target L2 | Fatal rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rq5_top[:5]:
        lines.append(f"| {_label(row)} | {_first_available(row, ('stage3_remap_quality_score',))} | {_first_available(row, ('stage3_pcap_alignment_coverage',))} | {_first_available(row, ('stage3_pcap_target_l2_mean',))} | {_first_available(row, ('stage3_pcap_valid_fatal_rate',))} |")
    lines.extend(
        [
            "",
            "| Least realistic remap cases | Remap score | Align Cov. | Target L2 | Fatal rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rq5_bottom[:5]:
        lines.append(f"| {_label(row)} | {_first_available(row, ('stage3_remap_quality_score',))} | {_first_available(row, ('stage3_pcap_alignment_coverage',))} | {_first_available(row, ('stage3_pcap_target_l2_mean',))} | {_first_available(row, ('stage3_pcap_valid_fatal_rate',))} |")
    lines.extend(
        [
            "",
            "### RQ6. Online Validation",
            "",
            "- RQ6 is intentionally excluded from this offline automation pass. The current report stops at packet-level replay and deployability proxies, as requested.",
            "",
            "## Aggregate Findings",
            "",
            "| Metric | Mean |",
            "| --- | ---: |",
            f"| Stage1 decision score | {_fmt_float(s1_mean)} |",
            f"| Stage2 decision score | {_fmt_float(s2_mean)} |",
            f"| Stage2 attack score | {_fmt_float(s2_attack_mean)} |",
            f"| Stage2 fidelity score | {_fmt_float(s2_fidelity_mean)} |",
            f"| Stage2 constraint score | {_fmt_float(s2_constraint_mean)} |",
            f"| Stage3 decision score | {_fmt_float(s3_mean)} |",
            f"| Stage3 deployability score | {_fmt_float(s3_deploy_mean)} |",
            f"| Stage3 remap quality score | {_fmt_float(s3_remap_mean)} |",
            f"| Stage3 alignment coverage | {_fmt_float(align_mean)} |",
            f"| Stage3 target L2 | {_fmt_float(target_l2_mean)} |",
            f"| Stage3 target MAE | {_fmt_float(target_mae_mean)} |",
            "",
            "### Overall Reading",
            "",
            f"- The strongest attack under the current decision metric is `{strongest_attack}`." if strongest_attack else "- No strongest attack identified.",
            f"- The weakest Stage2 case is `{weakest_attack}`." if weakest_attack else "- No weak Stage2 case identified.",
            f"- The weakest Stage3 case is `{weakest_stage3_attack}`." if weakest_stage3_attack else "- No weak Stage3 case identified.",
            "- When `Stage2 attack score` is high but `Stage2 fidelity score` is lower, the current method is succeeding by moving away from the benign manifold rather than by subtle concealment.",
            "- When `Stage3 remap quality score` remains high but `Stage3 deployability score` is lower, the bottleneck is no longer remapping itself but downstream validity or concealment.",
            "",
            "## Attack-Level Ranking",
            "",
            "| Attack | S1 Score | S2 Score | S2 Attack | S2 Fidelity | S3 Score | S3 Deploy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _label(row),
                    _first_available(row, s1_keys),
                    _first_available(row, s2_keys),
                    _first_available(row, s2_attack_keys),
                    _first_available(row, s2_fidelity_keys),
                    _first_available(row, s3_keys),
                    _first_available(row, s3_deploy_keys),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Family-Level Comparison",
            "",
            "| Family | Attacks | S1 Score | S2 Score | S2 Attack | S2 Fidelity | S3 Score |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_family_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("family", ""),
                    _first_available(row, ("attacks", "count")),
                    _first_available(row, ("stage1_decision_score", "avg_stage1_oracle_acc")),
                    _first_available(row, ("stage2_decision_score", "avg_stage2_asr_oracle")),
                    _first_available(row, ("stage2_attack_score", "avg_stage2_asr_oracle")),
                    _first_available(row, ("stage2_fidelity_score", "avg_stage2_norm_ffd")),
                    _first_available(row, ("stage3_decision_score", "avg_stage3_adv_benign_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Hard Cases",
            "",
            "### Lowest Stage2 Decision Score",
            "",
            "| Attack | S2 Score | S2 Attack | S2 Fidelity | S2 Constraint |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in hard_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _label(row),
                    _first_available(row, s2_keys),
                    _first_available(row, s2_attack_keys),
                    _first_available(row, s2_fidelity_keys),
                    _first_available(row, s2_constraint_keys),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Lowest Stage3 Decision Score",
            "",
            "| Attack | S3 Score | S3 Deploy | S3 Remap |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in stage3_hard_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _label(row),
                    _first_available(row, s3_keys),
                    _first_available(row, s3_deploy_keys),
                    _first_available(row, s3_remap_keys),
                ]
            )
            + " |"
        )
    benchmark_rows: list[dict[str, str]] = []
    for row in rows:
        s2 = _first_available_float(row, s2_keys)
        s3 = _first_available_float(row, s3_keys)
        s1 = _first_available_float(row, s1_keys)
        if s2 is None or s3 is None or s1 is None:
            continue
        if s2 < 0.85 or s3 < 0.85 or s1 < 0.90:
            benchmark_rows.append(row)
    benchmark_rows = sorted(
        benchmark_rows,
        key=lambda row: (
            _first_available_float(row, s2_keys) or float("inf"),
            _first_available_float(row, s3_keys) or float("inf"),
        ),
    )[:8]
    lines.extend(
        [
            "",
            "## Recommended Benchmark Subset",
            "",
            "These attacks are better stress tests than near-trivial cases because at least one stage remains non-saturated.",
            "",
            "| Attack | S1 Score | S2 Score | S3 Score |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in benchmark_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _label(row),
                    _first_available(row, s1_keys),
                    _first_available(row, s2_keys),
                    _first_available(row, s3_keys),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failure Boundary Analysis",
            "",
            "| Attack | Conflict Score | S3 Deploy | S3 Remap | Align Cov. | Target L2 | Remap Mode | Failure Signature |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in failure_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _label(row),
                    _fmt_float(_conflict_score(row)),
                    _first_available(row, s3_deploy_keys),
                    _first_available(row, s3_remap_keys),
                    _first_available(row, ("stage3_pcap_alignment_coverage",)),
                    _first_available(row, ("stage3_pcap_target_l2_mean",)),
                    _first_available(row, ("stage3_remap_mod_source",)),
                    _failure_reason(row),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Artifact Map",
            "",
            f"- Raw per-run summary: `{path.parent / 'attack_sweep_summary.csv'}`",
            f"- Aggregated summary: `{path.parent / 'attack_sweep_summary_agg.csv'}`",
            f"- Paper Stage2 table: `{path.parent / 'paper_stage2_table.csv'}`",
            f"- Paper Stage3 table: `{path.parent / 'paper_stage3_pcap_table.csv'}`",
            f"- Family-level summary: `{family_path}`",
            f"- Per-run pipeline outputs: `{out_root}/<variant>/seed_<seed>/<attack>/`",
            f"- Per-run Stage2 metrics: `{out_root}/<variant>/seed_<seed>/<attack>/stage2/metrics.json`",
            f"- Per-run Stage3 metrics: `{out_root}/<variant>/seed_<seed>/<attack>/stage3/metrics.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_csv_paths(data_cfg: dict) -> list[Path]:
    profile = resolve_dataset_profile(data_cfg)
    csv_path = str(profile.csv_path or "").strip()
    csv_dir = str(profile.csv_dir or "").strip()
    csv_glob = str(profile.csv_glob)
    paths: list[Path] = []
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        paths.append(path)
    if csv_dir:
        directory = Path(csv_dir)
        if not directory.exists():
            raise FileNotFoundError(f"CSV directory not found: {directory}")
        paths.extend(sorted(path for path in directory.glob(csv_glob) if path.is_file()))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No CSV files matched the configured dataset source.")
    return unique


def _normalize_label(value: object) -> str:
    return " ".join(str(value).replace("\ufeff", "").strip().split())


def _canonicalize_label(value: str) -> str:
    text = _normalize_label(value)
    text = text.replace("\ufffd", "-")
    text = text.replace("–", "-").replace("—", "-")
    while "  " in text:
        text = text.replace("  ", " ")
    for token in (" - ", "- ", " -"):
        text = text.replace(token, "-")
    return text.casefold()


def _discover_label_counts(data_cfg: dict) -> dict[str, int]:
    profile = resolve_dataset_profile(data_cfg)
    benign_labels = {_normalize_label(label) for label in profile.benign_labels}
    label_source = str(profile.label_source)
    label_col = str(profile.label_col) if label_source == "column" else None
    counts: dict[str, int] = {}
    for path in _resolve_csv_paths(data_cfg):
        if label_source == "parent_dir":
            label = _normalize_label(path.parent.name)
            if label and label not in benign_labels:
                with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue
                    row_count = sum(1 for _ in reader)
                counts[label] = counts.get(label, 0) + row_count
        else:
            with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                fieldnames = [_normalize_label(name) for name in header]
                if label_col not in fieldnames:
                    raise ValueError(f"Label column '{label_col}' not found in CSV: {path}")
                label_idx = fieldnames.index(label_col)
                for row in reader:
                    if label_idx >= len(row):
                        continue
                    label = _normalize_label(row[label_idx])
                    if not label or label == label_col or label in benign_labels:
                        continue
                    counts[label] = counts.get(label, 0) + 1
    return counts


def _load_cached_counts(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        return None
    return {str(key): int(value) for key, value in counts.items()}


def _write_cached_counts(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"counts": counts}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_attack_tokens(counts: dict[str, int], tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    canonical_map: dict[str, list[str]] = {}
    for raw in counts:
        canonical_map.setdefault(_canonicalize_label(raw), []).append(raw)

    resolved: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        raw = token.strip()
        if not raw:
            continue
        if raw in counts:
            chosen = raw
        else:
            matches = canonical_map.get(_canonicalize_label(raw), [])
            if len(matches) == 1:
                chosen = matches[0]
            else:
                missing.append(raw)
                continue
        if chosen not in seen:
            seen.add(chosen)
            resolved.append(chosen)
    if missing:
        raise SystemExit(f"Unknown or ambiguous attack labels: {', '.join(missing)}")
    return resolved


def _select_attacks(
    counts: dict[str, int],
    preferred: list[str],
    mode: str,
    limit: int,
) -> list[str]:
    all_attacks = sorted(counts, key=lambda name: (-counts[name], name))
    if mode == "all":
        return all_attacks[:limit] if limit > 0 else all_attacks
    if mode == "representative":
        chosen: list[str] = []
        seen: set[str] = set()
        for attack in all_attacks:
            family = _family(attack)
            if family in seen:
                continue
            seen.add(family)
            chosen.append(attack)
        return chosen[:limit] if limit > 0 else chosen
    chosen = _resolve_attack_tokens(counts, preferred)
    if not chosen:
        chosen = _select_attacks(counts, preferred, "representative", limit or 8)
    return chosen[:limit] if limit > 0 else chosen


def _slugify(name: str) -> str:
    slug = name.strip()
    for old, new in (("/", "_"), ("\\", "_"), (" ", "_"), (":", "_")):
        slug = slug.replace(old, new)
    return slug


def main(default_preset: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CSV attack-label sweep experiments.")
    parser.add_argument("--dataset-preset", choices=sorted(DATASET_PRESETS), default=default_preset or "nb15", help="Dataset preset")
    parser.add_argument("--config", default="")
    parser.add_argument("--mode", choices=["default", "representative", "all"], default="default")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attacks", default="", help="Comma-separated attack labels to run.")
    parser.add_argument("--variant", default="", help="Single ablation variant name.")
    parser.add_argument("--variants", default="", help="Comma-separated ablation variant names.")
    parser.add_argument("--seeds", default="", help="Comma-separated random seeds. Default uses config project.seed.")
    parser.add_argument("--list-attacks", action="store_true", help="List discovered attack labels and exit.")
    args = parser.parse_args()

    preset = DATASET_PRESETS[args.dataset_preset]
    default_config = str(config_path_from_env(str(preset["config"])))
    base_cfg_path = Path(args.config or default_config)
    base_cfg = _load_yaml(base_cfg_path)
    repo_root = Path(__file__).resolve().parents[1]
    display_name = str(preset["display_name"])
    log_tag = str(preset["log_tag"])
    preferred = [str(value) for value in preset["default_attacks"]]
    base_out_dir = Path(str(base_cfg["project"].get("out_dir", f"outputs/{args.dataset_preset}/default")))
    out_root = base_out_dir.parent
    counts_cache_path = out_root / "_discovered_attack_counts.json"

    counts = _load_cached_counts(counts_cache_path)
    if counts is None:
        counts = _discover_label_counts(base_cfg["data"])
        _write_cached_counts(counts_cache_path, counts)
    if args.list_attacks:
        for attack in sorted(counts, key=lambda name: (-counts[name], name)):
            canonical = _canonicalize_label(attack)
            display = attack if canonical == attack.casefold() else f"{attack}\t(alias: {canonical})"
            print(f"{display}\t{counts[attack]}")
        return

    if args.attacks:
        requested = [token.strip() for token in args.attacks.split(",") if token.strip()]
        attacks = _resolve_attack_tokens(counts, requested)
    else:
        attacks = _select_attacks(counts, preferred, args.mode, args.limit)
    if not attacks:
        raise SystemExit("No attack types selected.")

    variant_tokens = [token.strip() for token in args.variants.split(",") if token.strip()]
    if args.variant.strip():
        variant_tokens.append(args.variant.strip())
    variants = variant_tokens or ["full"]
    unknown_variants = [name for name in variants if name not in ABLATION_VARIANTS]
    if unknown_variants:
        raise SystemExit(f"Unknown variants: {', '.join(unknown_variants)}")
    if args.seeds.strip():
        seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    else:
        seeds = [int(base_cfg["project"].get("seed", 42))]
    if not seeds:
        raise SystemExit("No seeds selected.")

    generated_dir = out_root / "_generated_configs"
    summary_path = out_root / "attack_sweep_summary.csv"
    summary_agg_path = out_root / "attack_sweep_summary_agg.csv"
    paper_stage2_table_path = out_root / "paper_stage2_table.csv"
    paper_stage3_table_path = out_root / "paper_stage3_pcap_table.csv"
    family_summary_path = out_root / "family_summary.csv"
    significance_path = out_root / "significance_summary.csv"
    rq_summary_path = out_root / "rq_summary.csv"
    rq_coverage_path = out_root / "rq_coverage.csv"
    report_path = out_root / "EXPERIMENT_REPORT.md"
    benign_labels = [str(label) for label in base_cfg["data"].get("benign_labels", [])]
    rows_by_attack = _load_existing_rows(summary_path)

    for variant in variants:
        variant_patch = ABLATION_VARIANTS[variant]
        for attack in attacks:
            for seed in seeds:
                cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
                _deep_update(cfg, yaml.safe_load(yaml.safe_dump(variant_patch)))
                slug = _slugify(attack)
                variant_dir = out_root / variant / f"seed_{seed}"
                cfg["project"]["seed"] = int(seed)
                cfg["project"]["out_dir"] = str(variant_dir / slug)
                cfg["data"]["include_labels"] = [*benign_labels, attack]
                cfg["data"]["benign_labels"] = benign_labels
                stage3_cfg = cfg.setdefault("stage3", {})
                stage3_cfg["pcap_dataset"] = str(cfg["data"].get("dataset", ""))
                stage3_cfg["pcap_attack_label"] = attack
                stage3_cfg.setdefault("pcap_semantic_filter", True)
                cfg_path = generated_dir / variant / f"seed_{seed}" / f"{slug}.yaml"
                _write_yaml(cfg_path, cfg)

                print(f"[{log_tag}] running {attack} variant={variant} seed={seed}")
                timeout_sec = int(base_cfg.get("project", {}).get("stage_timeout_sec", 0) or 0) or 7200
                try:
                    subprocess.run(
                        [sys.executable, str(repo_root / "scripts" / "run_pipeline.py"), "--config", str(cfg_path)],
                        check=True,
                        cwd=str(repo_root),
                        timeout=timeout_sec,
                    )
                except subprocess.CalledProcessError as exc:
                    print(f"[{log_tag}] FAILED {attack} variant={variant} seed={seed}: exit code {exc.returncode}")
                    if hasattr(exc, "stderr") and exc.stderr:
                        print(f"[{log_tag}] stderr: {exc.stderr[:500]}")
                    continue
                except subprocess.TimeoutExpired as exc:
                    print(f"[{log_tag}] TIMEOUT {attack} variant={variant} seed={seed} after {timeout_sec}s")
                    continue

                out_dir = repo_root / cfg["project"]["out_dir"]
                oracle_name = resolve_oracle_name(cfg)
                stage1_metrics = load_json(out_dir / "stage1" / oracle_name / "metrics.json") if oracle_name else {}
                stage2_metrics = load_json(out_dir / "stage2" / "metrics.json")
                stage3_metrics = load_json(out_dir / "stage3" / "metrics.json")
                row = {
                    "variant": variant,
                    "seed": str(seed),
                    "attack_type": attack,
                    "attack_variant": f"{attack}::{variant}::seed{seed}",
                    "out_dir": str(out_dir),
                    "stage1_oracle_acc": str(stage1_metrics.get("oracle_eval_acc", "")),
                    "stage1_oracle_f1": str(stage1_metrics.get("oracle_eval_f1", "")),
                    "stage1_agreement": str(stage1_metrics.get("agreement", "")),
                    "stage1_baseline_agreement": str(stage1_metrics.get("baseline_agreement", "")),
                    "stage1_decision_score": str(stage1_metrics.get("stage1_decision_score", "")),
                    "stage2_generator_backbone": str(stage2_metrics.get("generator_backbone", "")),
                    "stage2_surrogate_guidance_mode": str(stage2_metrics.get("surrogate_guidance_mode", "")),
                    "stage2_asr_oracle": str(stage2_metrics.get("asr_oracle", "")),
                    "stage2_asr_surrogate": str(stage2_metrics.get("asr_surrogate", "")),
                    "stage2_adv_pmal_oracle": str(stage2_metrics.get("adv_prob_malicious_mean_oracle", "")),
                    "stage2_norm_ffd": str(stage2_metrics.get("norm_FFD", "")),
                    "stage2_norm_swd": str(stage2_metrics.get("norm_SWD", "")),
                    "stage2_decision_score": str(stage2_metrics.get("stage2_decision_score", "")),
                    "stage2_attack_score": str(stage2_metrics.get("stage2_decision_attack_effectiveness_score", "")),
                    "stage2_fidelity_score": str(stage2_metrics.get("stage2_decision_fidelity_score", "")),
                    "stage2_constraint_score": str(stage2_metrics.get("stage2_decision_constraint_score", "")),
                    "stage2_train_best_epoch": str(stage2_metrics.get("train_selection_best_epoch", "")),
                    "stage2_train_best_score": str(stage2_metrics.get("train_selection_best_score", "")),
                    "stage3_remap_mode": str(stage3_metrics.get("remap_mode", "")),
                    "stage3_protocol_auto_fix": str(stage3_metrics.get("protocol_auto_fix", "")),
                    "stage3_remap_r2": str(stage3_metrics.get("remapper_eval_r2", "")),
                    "stage3_remap_port_acc": str(stage3_metrics.get("remapper_eval_port_acc", "")),
                    "stage3_adv_benign_rate": str(stage3_metrics.get("adv_benign_rate", "")),
                    "stage3_decision_score": str(stage3_metrics.get("stage3_decision_score", "")),
                    "stage3_deployability_score": str(stage3_metrics.get("stage3_decision_pcap_deployability_score", "")),
                    "stage3_remap_quality_score": str(stage3_metrics.get("stage3_decision_remap_quality_score", "")),
                    "stage3_remap_mod_source": str(stage3_metrics.get("remap_mod_source", "")),
                    "stage3_remap_collapse_ratio": str(stage3_metrics.get("remap_collapse_ratio", "")),
                    "stage3_pcap_alignment_coverage": str(stage3_metrics.get("paper_pcap_alignment_coverage", stage3_metrics.get("pcap_eval_avg_alignment", ""))),
                    "stage3_pcap_alignment_missing": str(stage3_metrics.get("pcap_eval_avg_missing", "")),
                    "stage3_pcap_target_l2_mean": str(stage3_metrics.get("pcap_target_l2_mean", "")),
                    "stage3_pcap_target_mae_mean": str(stage3_metrics.get("pcap_target_mae_mean", "")),
                    "stage3_pcap_valid_fatal_rate": str(stage3_metrics.get("pcap_valid_fatal_rate", "")),
                    "stage3_pcap_validfatal_at_0": str(stage3_metrics.get("pcap_validfatal_at_0", "")),
                    "stage3_pcap_skip_reason": str(stage3_metrics.get("pcap_skip_reason", "")),
                    "stage3_pcap_feature_statuses": str(stage3_metrics.get("pcap_feature_statuses", "")),
                    "stage3_pcap_semantic_dataset": str(stage3_metrics.get("pcap_semantic_dataset", "")),
                    "stage3_pcap_semantic_attack_label": str(stage3_metrics.get("pcap_semantic_attack_label", "")),
                    "stage3_pcap_semantic_categories": str(stage3_metrics.get("pcap_semantic_categories", "")),
                    "stage3_remap_best_epoch": str(stage3_metrics.get("remapper_train_best_epoch", "")),
                    "stage3_remap_best_score": str(stage3_metrics.get("remapper_train_best_score", "")),
                }
                rows_by_attack[row["attack_variant"]] = row

    rows = [rows_by_attack[name] for name in sorted(rows_by_attack)]
    fieldnames = list(rows[0].keys()) if rows else []
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    agg_rows = _aggregate_attack_rows(rows)
    agg_fieldnames = list(agg_rows[0].keys()) if agg_rows else []
    with summary_agg_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    stage2_rows = _make_paper_stage2_table_agg_rows(agg_rows)
    stage3_rows = _make_paper_stage3_table_agg_rows(agg_rows)
    if stage2_rows:
        with paper_stage2_table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stage2_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stage2_rows)
    if stage3_rows:
        with paper_stage3_table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stage3_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stage3_rows)
    _write_significance_summary(significance_path, rows)
    _write_family_summary(family_summary_path, agg_rows)
    rq_summary_rows = _rq_summary_rows(agg_rows)
    if rq_summary_rows:
        with rq_summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rq_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rq_summary_rows)
    rq_coverage_rows = _rq_coverage_rows(agg_rows)
    if rq_coverage_rows:
        with rq_coverage_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rq_coverage_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rq_coverage_rows)
    _write_markdown_report(report_path, agg_rows, family_summary_path, display_name, out_root)
    print(f"[{log_tag}] summary {summary_path}")
    print(f"[{log_tag}] aggregated summary {summary_agg_path}")
    print(f"[{log_tag}] paper stage2 table {paper_stage2_table_path}")
    print(f"[{log_tag}] paper stage3 table {paper_stage3_table_path}")
    print(f"[{log_tag}] significance {significance_path}")
    print(f"[{log_tag}] family summary {family_summary_path}")
    print(f"[{log_tag}] rq summary {rq_summary_path}")
    print(f"[{log_tag}] rq coverage {rq_coverage_path}")
    print(f"[{log_tag}] report {report_path}")


if __name__ == "__main__":
    main()
