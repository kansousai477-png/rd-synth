"""Aggregate reviewer suite results for paper report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

RUN_DIR = Path("outputs/reviewer_suite/runs/2026-05-07T14-31-49Z_standard_nb15-2017-2018-iot23_seed_42")

KEYS_STAGE2 = ["oracle_attack_success_rate", "surrogate_attack_success_rate",
               "adv_prob_malicious_mean", "C2ST_AUC", "FFD", "SWD",
               "AdvToBen_L2", "CorrDelta", "Violation_Range"]
KEYS_STAGE3 = ["paper_pcap_attack_success_rate", "paper_pcap_concealment_proxy",
               "paper_pcap_detection_rate", "pcap_conditional_attack_success_rate",
               "pcap_adv_prob_malicious_mean", "pcap_orig_prob_malicious",
               "pcap_eval_avg_alignment", "stage3_decision_pcap_deployability_score",
               "stage3_decision_score", "stage3_full_evidence",
               "pcap_conditioned_feature_asr", "pcap_conditioned_adv_prob_malicious",
               "pcap_conditioned_source_prob_malicious",
               "remapper_eval_r2", "pcap_sanity_tcp_flag_invalid_rate",
               "pcap_sanity_nonmonotonic_rate", "pcap_validfatal_at_0"]

CATEGORIES = ["main", "ablation"]

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_get(d: dict, key: str, default=None):
    v = d.get(key)
    if v is None or (isinstance(v, float) and not __import__("math").isfinite(v)):
        return default
    return v

def fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


print("=" * 90)
print("RDSYNTH REVIEWER SUITE — AGGREGATED RESULTS")
print("=" * 90)

for dataset in ["nb15", "2017", "2018", "iot23"]:
    print(f"\n{'='*90}")
    print(f"DATASET: {dataset}")
    print(f"{'='*90}")

    for category in CATEGORIES:
        variants = ["full"] if category == "main" else ["backbone_gan", "random_remap", "w_o_stage1"]
        for variant in variants:
            if category == "main":
                base = RUN_DIR / dataset / "main" / "seed_42" / "global"
            else:
                base = RUN_DIR / dataset / "ablation" / variant / "seed_42" / "GLOBAL"

            s2_path = base / "stage2" / "metrics.json"
            s3_path = base / "stage3" / "metrics.json"

            s2 = load_json(s2_path) if s2_path.exists() else {}
            s3 = load_json(s3_path) if s3_path.exists() else {}

            label = f"{dataset}/{category}/{variant}"
            print(f"\n  --- {label} ---")

            # Stage2 core metrics
            print(f"  [Stage2] Oracle ASR={fmt(s2.get('oracle_attack_success_rate'))} "
                  f"Surrogate ASR={fmt(s2.get('surrogate_attack_success_rate'))} "
                  f"AdvPmal={fmt(s2.get('adv_prob_malicious_mean'))} "
                  f"FFD={fmt(s2.get('FFD'))} SWD={fmt(s2.get('SWD'))} "
                  f"C2ST-AUC={fmt(s2.get('C2ST_AUC'))}")

            # Baseline comparison
            baseline_keys = [k for k in s2 if k.startswith("baseline_") and "_oracle_asr" in k]
            if baseline_keys:
                best_baseline = max(
                    (float(s2[k]) for k in baseline_keys if isinstance(s2[k], (int, float))),
                    default=0.0
                )
                our_asr = float(s2.get("oracle_attack_success_rate", 0) or 0)
                print(f"  [Stage2] Best baseline oracle ASR={best_baseline:.4f}  Ours={our_asr:.4f}")

            # Stage3 core metrics
            print(f"  [Stage3] paper_pcap_ASR={fmt(s3.get('paper_pcap_attack_success_rate'))} "
                  f"concealment={fmt(s3.get('paper_pcap_concealment_proxy'))} "
                  f"detection_rate={fmt(s3.get('paper_pcap_detection_rate'))}")

            print(f"  [Stage3] SourcePmal={fmt(s3.get('pcap_orig_prob_malicious'))} "
                  f"AdvPmal={fmt(s3.get('pcap_adv_prob_malicious_mean'))} "
                  f"alignment={fmt(s3.get('pcap_eval_avg_alignment'))}")

            print(f"  [Stage3] FeatASR={fmt(s3.get('pcap_conditioned_feature_asr'))} "
                  f"FeatAdvPmal={fmt(s3.get('pcap_conditioned_adv_prob_malicious'))} "
                  f"FeatSrcPmal={fmt(s3.get('pcap_conditioned_source_prob_malicious'))}")

            print(f"  [Stage3] RemapR2={fmt(s3.get('remapper_eval_r2'))} "
                  f"DeployScore={fmt(s3.get('stage3_decision_pcap_deployability_score'))} "
                  f"FullEvidence={fmt(s3.get('stage3_full_evidence'))}")

            print(f"  [Stage3] TCP_flag_invalid={fmt(s3.get('pcap_sanity_tcp_flag_invalid_rate'))} "
                  f"Nonmono={fmt(s3.get('pcap_sanity_nonmonotonic_rate'))} "
                  f"ValidFatal@0={fmt(s3.get('pcap_validfatal_at_0'))}")

            # Baseline PCAP comparison
            pc_baseline_keys = [k for k in s3 if k.startswith("baseline_") and "paper_pcap_attack_success_rate" in k]
            our_pcap_asr = s3.get("paper_pcap_attack_success_rate", 0) or 0
            best_bl_pcap = 0.0
            bl_with_asr = []
            for k in pc_baseline_keys:
                v = s3.get(k)
                if isinstance(v, (int, float)):
                    bl_name = k.replace("baseline_", "").replace("_paper_pcap_attack_success_rate", "")
                    bl_with_asr.append((bl_name, float(v)))
                    if float(v) > best_bl_pcap:
                        best_bl_pcap = float(v)
            bl_with_asr.sort(key=lambda x: -x[1])
            print(f"  [Stage3] Our_pcap_ASR={fmt(our_pcap_asr)}  Best_baseline_pcap_ASR={fmt(best_bl_pcap)}")
            if bl_with_asr:
                top3 = bl_with_asr[:3]
                print(f"  [Stage3] Top baseline PCAP ASRs: {', '.join(f'{n}={v:.3f}' for n,v in top3)}")

# Final summary table
print(f"\n{'='*90}")
print("CROSS-DATASET SUMMARY TABLE")
print(f"{'='*90}")
print(f"{'Dataset':<10} {'Oracle ASR':>10} {'PCAP ASR':>10} {'Src pmal':>10} {'Adv pmal':>10} {'Deploy':>10} {'Evidence':>12}")
print("-" * 75)
for dataset in ["nb15", "2017", "2018", "iot23"]:
    base = RUN_DIR / dataset / "main" / "seed_42" / "global"
    s2 = load_json(base / "stage2" / "metrics.json") if (base / "stage2" / "metrics.json").exists() else {}
    s3 = load_json(base / "stage3" / "metrics.json") if (base / "stage3" / "metrics.json").exists() else {}
    print(f"{dataset:<10} {fmt(s2.get('oracle_attack_success_rate')):>10} "
          f"{fmt(s3.get('paper_pcap_attack_success_rate')):>10} "
          f"{fmt(s3.get('pcap_orig_prob_malicious')):>10} "
          f"{fmt(s3.get('pcap_adv_prob_malicious_mean')):>10} "
          f"{fmt(s3.get('stage3_decision_pcap_deployability_score')):>10} "
          f"{fmt(s3.get('stage3_full_evidence')):>12}")

# Ablation summary
print(f"\n{'='*90}")
print("ABLATION SUMMARY (unsw)")
print(f"{'='*90}")
print(f"{'Variant':<20} {'Oracle ASR':>10} {'PCAP ASR':>10} {'Deploy':>10} {'FullEvidence':>12}")
print("-" * 65)
for variant in ["full", "backbone_gan", "random_remap", "w_o_stage1"]:
    if variant == "full":
        base = RUN_DIR / "nb15" / "main" / "seed_42" / "global"
    else:
        base = RUN_DIR / "nb15" / "ablation" / variant / "seed_42" / "GLOBAL"
    s2 = load_json(base / "stage2" / "metrics.json") if (base / "stage2" / "metrics.json").exists() else {}
    s3 = load_json(base / "stage3" / "metrics.json") if (base / "stage3" / "metrics.json").exists() else {}
    print(f"{variant:<20} {fmt(s2.get('oracle_attack_success_rate')):>10} "
          f"{fmt(s3.get('paper_pcap_attack_success_rate')):>10} "
          f"{fmt(s3.get('stage3_decision_pcap_deployability_score')):>10} "
          f"{fmt(s3.get('stage3_full_evidence')):>12}")

print("\nDone.")
