from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _fmt(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.6f}"


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if value == value]
    return sum(finite) / len(finite) if finite else None


def build_feature_gap_summary(run_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    stage2 = _load_json(run_root / "stage2" / "metrics.json")
    stage3 = _load_json(run_root / "stage3" / "metrics.json")
    pcap_rows = _load_csv_rows(run_root / "stage3" / "pcap_eval.csv")
    source_rows = [row for row in pcap_rows if str(row.get("is_original", "")).strip() in {"1", "True", "true"}]
    adv_rows = [row for row in pcap_rows if str(row.get("is_original", "")).strip() in {"0", "False", "false"}]

    source_pmal = [_to_float(row.get("prob_malicious")) for row in source_rows]
    adv_pmal = [_to_float(row.get("prob_malicious")) for row in adv_rows]
    target_l2 = [_to_float(row.get("target_l2")) for row in adv_rows]
    align = [_to_float(row.get("alignment_coverage")) for row in pcap_rows]
    missing = [_to_float(row.get("alignment_missing")) for row in pcap_rows]

    source_pmal_mean = _mean([value for value in source_pmal if value is not None])
    adv_pmal_mean = _mean([value for value in adv_pmal if value is not None])
    rows = [
        {"metric": "stage2_asr_oracle", "value": stage2.get("asr_oracle", "")},
        {"metric": "stage2_adv_pmal_oracle", "value": stage2.get("adv_prob_malicious_mean_oracle", "")},
        {"metric": "stage2_norm_FFD", "value": stage2.get("norm_FFD", "")},
        {"metric": "stage2_norm_SWD", "value": stage2.get("norm_SWD", "")},
        {"metric": "stage3_pcap_attack_success_rate", "value": stage3.get("paper_pcap_attack_success_rate", "")},
        {"metric": "stage3_source_pmal_mean", "value": source_pmal_mean},
        {"metric": "stage3_adv_pmal_mean", "value": adv_pmal_mean},
        {
            "metric": "stage3_pmal_delta_adv_minus_source",
            "value": None if source_pmal_mean is None or adv_pmal_mean is None else adv_pmal_mean - source_pmal_mean,
        },
        {
            "metric": "stage3_target_l2_mean",
            "value": stage3.get("pcap_target_l2_mean", _mean([v for v in target_l2 if v is not None])),
        },
        {
            "metric": "stage3_alignment_coverage_mean",
            "value": stage3.get("pcap_eval_avg_alignment", _mean([v for v in align if v is not None])),
        },
        {"metric": "stage3_alignment_missing_mean", "value": _mean([v for v in missing if v is not None])},
        {"metric": "stage3_selected_alpha_mean", "value": stage3.get("pcap_selected_alpha_mean", "")},
        {"metric": "stage3_selected_response_l2_mean", "value": stage3.get("pcap_selected_response_l2_mean", "")},
        {"metric": "stage3_selected_field_sets", "value": stage3.get("pcap_selected_field_sets", "")},
        {"metric": "stage3_evidence_block_reason", "value": stage3.get("stage3_evidence_block_reason", "")},
    ]

    asr = _to_float(stage2.get("asr_oracle"))
    replay = _to_float(stage3.get("paper_pcap_attack_success_rate"))
    l2 = _to_float(stage3.get("pcap_target_l2_mean"))
    delta = None if source_pmal_mean is None or adv_pmal_mean is None else adv_pmal_mean - source_pmal_mean
    narrative = [
        "# Stage2-to-Stage3 Feature Gap Analysis",
        "",
        f"- Stage2 oracle ASR: `{_fmt(asr)}`; Stage2 oracle adversarial p_mal: `{_fmt(stage2.get('adv_prob_malicious_mean_oracle'))}`.",
        f"- Stage3 replay ASR: `{_fmt(replay)}`; source p_mal mean: `{_fmt(source_pmal_mean)}`; adv p_mal mean: `{_fmt(adv_pmal_mean)}`; delta: `{_fmt(delta)}`.",
        f"- Stage3 target L2 mean: `{_fmt(l2)}`; alignment coverage mean: `{_fmt(stage3.get('pcap_eval_avg_alignment', _mean([v for v in align if v is not None])))}`.",
    ]
    if asr is not None and asr >= 0.8 and (replay is None or replay < 0.5):
        narrative.append(
            "- Diagnosis: feature-space evasion is strong, but traffic-space replay remains weak; inspect carrier mismatch, target L2, and alignment status before treating this as algorithmic success."
        )
    elif l2 is not None and l2 >= 10.0:
        narrative.append(
            "- Diagnosis: the generated PCAP may evade, but the target-feature distance remains high, so remapping fidelity is still a risk boundary."
        )
    else:
        narrative.append(
            "- Diagnosis: no severe Stage2-to-Stage3 failure pattern is visible from the available aggregate metrics."
        )
    return rows, narrative


def write_outputs(run_root: Path, out_dir: Path) -> tuple[Path, Path]:
    rows, narrative = build_feature_gap_summary(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stage2_stage3_feature_gap_summary.csv"
    md_path = out_dir / "stage2_stage3_feature_gap_analysis.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"metric": row["metric"], "value": _fmt(row["value"])})
    md_path.write_text("\n".join(narrative) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage2 feature-space vs Stage3 nfstream/PCAP feature gap.")
    parser.add_argument("--run-root", required=True, help="Pipeline run root containing stage2/ and stage3/.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to <run-root>/analysis.")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "analysis"
    csv_path, md_path = write_outputs(run_root, out_dir)
    print(f"[FeatureGap] csv={csv_path}")
    print(f"[FeatureGap] report={md_path}")


if __name__ == "__main__":
    main()
