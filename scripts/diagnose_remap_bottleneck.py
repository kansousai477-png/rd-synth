"""H1+H6: PCAP 重映射瓶颈诊断 — 三步逐维特征差距 + 信息瓶颈分析.

meeting 4.30 方案:
  V0 = source PCAP 原始特征向量 (original ASR)
  V1 = Stage2 对抗特征向量 (feature ASR, 69维)
  V2 = 修改后 PCAP nfstream 重提取特征 (real ASR, 69维)

信息瓶颈: 69维特征 → 14维修改参数 → PCAP → nfstream → 69维特征
关键问题: 14维修改参数能否充分表达 69维对抗偏移?

用法:
  .\scripts\python_in_venv.ps1 scripts\diagnose_remap_bottleneck.py --run-root outputs/paper_main
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.keys()}


# ---------------------------------------------------------------------------
# 信息瓶颈分析: 69维 → 14维修改参数
# ---------------------------------------------------------------------------


MOD_NAMES = [
    "mean_iat_ms", "std_iat_ms", "pad_bytes", "dst_port_new",
    "flag_ratio", "flow_scale", "payload_scale", "src_port_new",
    "tcp_init_win_fwd", "tcp_init_win_bwd", "syn_flag_ratio",
    "fin_flag_ratio", "rst_flag_ratio", "fwd_pkt_scale",
]

# Which CIC features each modification parameter influences (via PCAP structure)
MOD_INFLUENCE = {
    "mean_iat_ms": ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
                     "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
                     "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
                     "Flow Duration", "Flow Packets/s", "Flow Bytes/s",
                     "Active Mean", "Active Std", "Active Max", "Active Min",
                     "Idle Mean", "Idle Std", "Idle Max", "Idle Min"],
    "std_iat_ms": ["Flow IAT Std", "Fwd IAT Std", "Bwd IAT Std"],
    "pad_bytes": ["Packet Length Mean", "Packet Length Std", "Packet Length Variance",
                   "Total Length of Fwd Packet", "Total Length of Bwd Packet",
                   "Fwd Packet Length Mean", "Bwd Packet Length Mean",
                   "Average Packet Size"],
    "dst_port_new": ["Dst Port", "Destination Port"],
    "src_port_new": ["Src Port", "Source Port"],
    "flag_ratio": ["PSH Flag Count", "Fwd PSH Flags", "Bwd PSH Flags"],
    "flow_scale": ["Flow Duration", "Flow Bytes/s", "Flow Packets/s",
                   "Fwd Packets/s", "Bwd Packets/s"],
    "payload_scale": ["Packet Length Mean", "Packet Length Std",
                      "Total Length of Fwd Packet", "Total Length of Bwd Packet",
                      "Average Packet Size", "Flow Bytes/s"],
    "tcp_init_win_fwd": ["FWD Init Win Bytes", "Init_Win_bytes_forward"],
    "tcp_init_win_bwd": ["Bwd Init Win Bytes", "Init_Win_bytes_backward"],
    "syn_flag_ratio": ["SYN Flag Count", "syn_flag_number"],
    "fin_flag_ratio": ["FIN Flag Count", "fin_flag_number"],
    "rst_flag_ratio": ["RST Flag Count", "rst_flag_number"],
    "fwd_pkt_scale": ["Total Fwd Packet", "Fwd Packets/s", "Subflow Fwd Packets",
                      "Fwd Act Data Pkts", "Total Length of Fwd Packet",
                      "Fwd Packet Length Mean"],
}


def _feature_group(name_lower: str) -> str:
    temporal_kw = ["iat", "duration", "rate", "active", "idle", "time", "packets/s", "bytes/s"]
    spatial_kw = ["length", "size", "bytes", "header", "segment", "payload",
                  "packet len", "pkt len", "variance"]
    protocol_kw = ["flag", "port", "protocol", "syn", "fin", "rst", "psh",
                   "ack", "urg", "cwr", "ece", "window", "ttl", "init_win"]
    if any(kw in name_lower for kw in temporal_kw):
        return "temporal"
    if any(kw in name_lower for kw in protocol_kw):
        return "protocol"
    if any(kw in name_lower for kw in spatial_kw):
        return "spatial"
    return "other"


def analyze_information_bottleneck(
    feature_names: list[str],
    adv: np.ndarray,
    mal: np.ndarray,
    benign: np.ndarray,
) -> dict[str, Any]:
    """Analyze the 69→14 information bottleneck.

    Key question: what fraction of the adversarial perturbation variance
    in the 69-dim feature space can be explained by the 14 modification params?
    """
    n_features = len(feature_names)
    n_mod = len(MOD_NAMES)

    # Map each feature to which mod params influence it
    feature_influence: dict[str, list[str]] = {}
    for i, name in enumerate(feature_names):
        nl = name.lower().strip()
        influencers = []
        for mod, influenced_features in MOD_INFLUENCE.items():
            for inf in influenced_features:
                if inf.lower().strip() == nl or inf.lower().strip() in nl or nl in inf.lower().strip():
                    if mod not in influencers:
                        influencers.append(mod)
        feature_influence[name] = influencers

    # Count features with no direct mod parameter influence
    covered = sum(1 for v in feature_influence.values() if v)
    uncovered = [name for name, v in feature_influence.items() if not v]
    uncovered_by_group = {}
    for name in uncovered:
        grp = _feature_group(name.lower())
        uncovered_by_group.setdefault(grp, []).append(name)

    # Adversarial perturbation magnitude per feature (in original scale)
    adv_mean = adv.mean(axis=0)
    mal_mean = mal.mean(axis=0)
    benign_mean = benign.mean(axis=0)
    perturbation = adv_mean - mal_mean   # How much Stage2 shifts features

    # Per-feature group perturbation
    groups = {"temporal": [], "spatial": [], "protocol": [], "other": []}
    for i, name in enumerate(feature_names):
        grp = _feature_group(name.lower())
        groups[grp].append({
            "index": i,
            "feature": name,
            "perturbation": float(perturbation[i]),
            "perturbation_abs": float(abs(perturbation[i])),
            "influenced_by": feature_influence.get(name, []),
            "is_uncovered": name in uncovered,
        })

    group_stats = {}
    for grp, feats in groups.items():
        if not feats:
            continue
        pert_abs = [f["perturbation_abs"] for f in feats]
        group_stats[grp] = {
            "count": len(feats),
            "mean_abs_perturbation": float(np.mean(pert_abs)),
            "total_perturbation_l2": float(np.linalg.norm([f["perturbation"] for f in feats])),
            "uncovered_count": sum(1 for f in feats if f["is_uncovered"]),
            "top5_perturbed": sorted(feats, key=lambda f: f["perturbation_abs"], reverse=True)[:5],
        }

    return {
        "feature_count": n_features,
        "mod_param_count": n_mod,
        "compression_ratio": float(n_features) / float(n_mod),
        "covered_features": covered,
        "uncovered_features": len(uncovered),
        "uncovered_by_group": {k: len(v) for k, v in uncovered_by_group.items()},
        "uncovered_feature_names": uncovered,
        "group_stats": group_stats,
    }


# ---------------------------------------------------------------------------
# Remapper accuracy analysis
# ---------------------------------------------------------------------------


def analyze_remapper_accuracy(remap_eval: list[dict[str, str]]) -> dict[str, Any]:
    """Analyze per-mod-parameter remapper accuracy."""
    mods = []
    for row in remap_eval:
        mod_name = row.get("mod_name", "")
        mae = float(row.get("mae", 0))
        rmse = float(row.get("rmse", 0))
        target_mean = float(row.get("target_mean", 0))
        target_std = float(row.get("target_std", 0))
        nrmse = rmse / (target_std + 1e-8)
        mods.append({
            "mod_name": mod_name,
            "mae": mae, "rmse": rmse,
            "target_mean": target_mean, "target_std": target_std,
            "normalized_rmse": nrmse,
            "accuracy_rating": "good" if nrmse < 0.5 else ("fair" if nrmse < 1.0 else "poor"),
        })

    # Sort by normalized RMSE (worst first)
    mods.sort(key=lambda m: m["normalized_rmse"], reverse=True)

    good = [m for m in mods if m["accuracy_rating"] == "good"]
    fair = [m for m in mods if m["accuracy_rating"] == "fair"]
    poor = [m for m in mods if m["accuracy_rating"] == "poor"]

    return {
        "per_param": mods,
        "good_count": len(good), "fair_count": len(fair), "poor_count": len(poor),
        "good_params": [m["mod_name"] for m in good],
        "poor_params": [m["mod_name"] for m in poor],
        "worst3": [m["mod_name"] for m in mods[:3]],
    }


# ---------------------------------------------------------------------------
# PCAP evasion analysis
# ---------------------------------------------------------------------------


def analyze_pcap_evasion(pcap_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Analyze per-PCAP evasion success/failure patterns."""
    original = [r for r in pcap_rows if str(r.get("is_original", "0")) == "1"]
    adversarial = [r for r in pcap_rows if str(r.get("is_original", "0")) == "0"]

    per_source = []
    for adv_r in adversarial:
        src = adv_r.get("source_name", "")
        orig_r = next((r for r in original if r.get("source_name") == src), None)
        pmal_orig = float(orig_r.get("prob_malicious", 0) or 0) if orig_r else 0.0
        pmal_adv = float(adv_r.get("prob_malicious", 0) or 0)
        tgt_l2 = float(adv_r.get("target_l2", 0) or 0)
        tgt_mae = float(adv_r.get("target_mae", 0) or 0)

        per_source.append({
            "source_name": src,
            "pmal_original": pmal_orig,
            "pmal_adversarial": pmal_adv,
            "pmal_delta": pmal_adv - pmal_orig,
            "target_l2": tgt_l2,
            "target_mae": tgt_mae,
            "evasion_success": pmal_adv < 0.5,
            "eligible": pmal_orig >= 0.5,
        })

    eligible = [p for p in per_source if p["eligible"]]
    successes = [p for p in eligible if p["evasion_success"]]
    already_evasive = [p for p in per_source if not p["eligible"]]

    return {
        "total_sources": len(per_source),
        "eligible_count": len(eligible),
        "success_count": len(successes),
        "already_evasive_count": len(already_evasive),
        "evasion_rate_eligible": len(successes) / max(1, len(eligible)),
        "per_source": per_source,
        "mean_target_l2": float(np.mean([p["target_l2"] for p in per_source if p["target_l2"] > 0])),
        "mean_pmal_delta": float(np.mean([p["pmal_delta"] for p in eligible])),
    }


# ---------------------------------------------------------------------------
# Main diagnosis
# ---------------------------------------------------------------------------


def diagnose(run_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"run_root": str(run_root)}

    # 1. Load Stage2 data
    adv_npz = _load_npz(run_root / "stage2" / "adv_samples.npz")
    if not adv_npz:
        # Try attack_eval subdirectory
        attack_eval = run_root / "stage2" / "attack_eval"
        if attack_eval.exists():
            for attack_dir in sorted(attack_eval.iterdir()):
                if attack_dir.is_dir():
                    adv_npz = _load_npz(attack_dir / "adv_samples.npz")
                    if adv_npz:
                        result["attack_type"] = attack_dir.name
                        break

    if not adv_npz:
        result["error"] = "No adv_samples.npz found"
        return result

    feature_names = [str(n) for n in adv_npz.get("feature_names", [])]
    adv = adv_npz.get("adv")           # original scale
    mal = adv_npz.get("mal")           # original scale
    benign = adv_npz.get("benign")     # original scale

    result["feature_count"] = len(feature_names)
    result["adv_samples"] = int(adv.shape[0]) if adv is not None else 0
    result["feature_names"] = feature_names

    # 2. Information bottleneck analysis
    if adv is not None and mal is not None and benign is not None:
        result["info_bottleneck"] = analyze_information_bottleneck(
            feature_names, adv, mal, benign,
        )

    # 3. Remapper accuracy
    remap_eval = _load_csv(run_root / "stage3" / "stage3_remap_eval.csv")
    if remap_eval:
        result["remapper"] = analyze_remapper_accuracy(remap_eval)

    # 4. PCAP evasion analysis
    pcap_rows = _load_csv(run_root / "stage3" / "pcap_eval.csv")
    if pcap_rows:
        result["pcap_evasion"] = analyze_pcap_evasion(pcap_rows)

    # 5. Stage3 metrics
    s3_metrics = _load_json(run_root / "stage3" / "metrics.json")
    result["stage3_metrics"] = {
        k: s3_metrics.get(k) for k in [
            "paper_pcap_attack_success_rate", "paper_pcap_concealment_proxy",
            "paper_pcap_fidelity_target_l2", "paper_pcap_fidelity_target_mae",
            "pcap_target_l2_mean", "pcap_target_mae_mean",
            "pcap_eval_avg_alignment", "pcap_valid_fatal_rate",
            "pcap_replay_eligible_source_count", "pcap_source_already_evasive_count",
            "stage3_total_time_sec", "pcap_pcaps_per_sec", "pcap_packet_throughput_pps",
        ] if k in s3_metrics
    }

    # 6. Generate narrative
    result["narrative"] = _build_narrative(result)

    return result


def _build_narrative(r: dict[str, Any]) -> list[str]:
    lines = [
        "# PCAP 重映射瓶颈诊断报告",
        "",
        f"**运行目录**: `{r['run_root']}`",
        f"**特征维度**: {r.get('feature_count', '?')}",
        f"**Stage2 对抗样本数**: {r.get('adv_samples', '?')}",
        f"**攻击类型**: {r.get('attack_type', 'overall')}",
        "",
        "---",
        "",
    ]

    # ── PCAP Evasion Summary ──
    pe = r.get("pcap_evasion", {})
    if pe:
        lines += [
            "## 1. PCAP 绕过总览",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 源 PCAP 总数 | {pe['total_sources']} |",
            f"| 合格源 (pmal >= 0.5) | {pe['eligible_count']} |",
            f"| 成功绕过 | {pe['success_count']} |",
            f"| 已天然逃逸 (pmal < 0.5) | {pe['already_evasive_count']} |",
            f"| 合格源绕过率 | {pe['evasion_rate_eligible']:.0%} |",
            f"| 平均 target_l2 | {pe['mean_target_l2']:.2f} |",
            f"| 平均 pmal 降幅 | {pe['mean_pmal_delta']:+.4f} |",
            "",
        ]

        # Per-source table
        lines += [
            "| 源 PCAP | pmal_orig | pmal_adv | Δ | target_l2 | 成功 |",
            "|---------|-----------|----------|----|-----------|------|",
        ]
        for s in pe["per_source"]:
            lines.append(
                f"| {s['source_name'][:45]} | {s['pmal_original']:.4f} | "
                f"{s['pmal_adversarial']:.4f} | {s['pmal_delta']:+.4f} | "
                f"{s['target_l2']:.2f} | {s['evasion_success']} |"
            )
        lines.append("")

        # Key finding about already evasive
        if pe["already_evasive_count"] > pe["eligible_count"]:
            lines += [
                "**关键发现**: 大部分源 PCAP 对 CSV 训练的 Oracle 已经天然逃逸。",
                "这不是重映射算法的问题，而是 **Oracle 无法有效检测真实 PCAP** 的证据。",
                "需要同时使用 pcap_ids (PCAP 训练的分类器) 评估才能反映真实部署场景。",
                "",
            ]

    # ── Information Bottleneck ──
    ib = r.get("info_bottleneck", {})
    if ib:
        lines += [
            "## 2. 信息瓶颈分析 (69维 → 14维修改参数)",
            "",
            f"- 特征维度: {ib['feature_count']}",
            f"- 修改参数维度: {ib['mod_param_count']}",
            f"- 压缩比: {ib['compression_ratio']:.1f}:1",
            f"- 有直接修改参数覆盖的特征: {ib['covered_features']}/{ib['feature_count']}",
            f"- 无直接修改参数覆盖的特征: {ib['uncovered_features']}",
            "",
        ]

        gs = ib.get("group_stats", {})
        if gs:
            lines += [
                "### 分组对抗扰动强度",
                "",
                "| 组 | 特征数 | 平均|扰动| | 总扰动L2 | 无覆盖数 |",
                "|-----|--------|----------|----------|---------|",
            ]
            for grp, stats in sorted(gs.items()):
                lines.append(
                    f"| {grp} | {stats['count']} | {stats['mean_abs_perturbation']:.4f} | "
                    f"{stats['total_perturbation_l2']:.2f} | {stats['uncovered_count']} |"
                )
            lines.append("")

            # Top perturbed features per group
            for grp, stats in sorted(gs.items()):
                top = stats.get("top5_perturbed", [])[:3]
                if top:
                    lines.append(f"**{grp}** 扰动最大的特征: " +
                                 ", ".join(f"{t['feature']}({t['perturbation_abs']:.3f})" for t in top))

        # Uncovered features
        uncovered = ib.get("uncovered_feature_names", [])
        if uncovered:
            uncovered_by_grp = ib.get("uncovered_by_group", {})
            lines += [
                "",
                "### 无修改参数覆盖的特征",
                "",
            ]
            for grp, count in sorted(uncovered_by_grp.items()):
                grp_uncovered = [n for n in uncovered if _feature_group(n.lower()) == grp]
                lines.append(f"- **{grp}** ({count}个): {', '.join(grp_uncovered[:8])}")
            lines.append("")

    # ── Remapper Accuracy ──
    rem = r.get("remapper", {})
    if rem:
        lines += [
            "## 3. Remapper 逐参数精度",
            "",
            f"- 高精度参数 (NRMSE < 0.5): {rem['good_count']} — {', '.join(rem.get('good_params', [])[:6])}",
            f"- 低精度参数 (NRMSE >= 1.0): {rem['poor_count']} — {', '.join(rem.get('poor_params', []))}",
            "",
            "| 参数 | MAE | NRMSE | 评级 |",
            "|------|-----|-------|------|",
        ]
        for p in rem.get("per_param", [])[:8]:
            lines.append(
                f"| {p['mod_name']} | {p['mae']:.2f} | {p['normalized_rmse']:.2f} | {p['accuracy_rating']} |"
            )
        lines.append("")

    # ── Stage3 Metrics ──
    s3 = r.get("stage3_metrics", {})
    if s3:
        lines += [
            "## 4. Stage3 关键指标",
            "",
        ]
        for k, v in s3.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # ── Diagnosis & Recommendations ──
    lines += [
        "---",
        "## 5. 诊断结论与建议",
        "",
    ]

    pe = r.get("pcap_evasion", {})
    ib = r.get("info_bottleneck", {})
    rem = r.get("remapper", {})

    # Finding 1: already evasive
    if pe.get("already_evasive_count", 0) > pe.get("eligible_count", 1):
        lines += [
            "### 发现 1: Oracle 无法有效检测真实 PCAP",
            f"源 PCAP 中 {pe['already_evasive_count']}/{pe['total_sources']} 对 CSV Oracle 已天然逃逸。",
            "这表明 CSV 预提取特征和 PCAP 实时特征之间存在显著分布偏移。",
            "**建议**: 所有 Stage3 评估必须同时报告 Oracle 和 pcap_ids 双轨结果。",
            "论文中需要讨论这一 representation gap 对威胁模型有效性的影响。",
            "",
        ]

    # Finding 2: information bottleneck
    if ib:
        uncovered = ib.get("uncovered_features", 0)
        lines += [
            f"### 发现 2: {ib['feature_count']}→{ib['mod_param_count']} 信息瓶颈",
            f"{uncovered}/{ib['feature_count']} 个特征无直接修改参数覆盖，",
            "只能通过 PCAP 结构的副作用间接改变。",
            "**建议**: 考虑增加修改参数量（如 fwd/bwd 分离的 IAT、per-direction payload），",
            "或在论文中量化论证信息损失的可接受性。",
            "",
        ]

    # Finding 3: remapper accuracy
    if rem:
        poor = rem.get("poor_params", [])
        if poor:
            lines += [
                f"### 发现 3: Remapper 在 {', '.join(poor)} 上精度不足",
                "但这些参数（端口、TCP窗口）对 Oracle 分类的影响可能有限。",
                "dst_port 使用 vocab-aware 选择策略缓解了 MAE 大的问题。",
                "**建议**: 验证去除 TCP init_win 修改是否影响 ASR，简化修改参数集。",
                "",
            ]

    # Finding 4: target_l2 gap
    s3 = r.get("stage3_metrics", {})
    target_l2 = s3.get("pcap_target_l2_mean") or s3.get("paper_pcap_fidelity_target_l2")
    if target_l2 and target_l2 > 5:
        lines += [
            f"### 发现 4: target_l2 = {target_l2:.2f} 偏高",
            "修改后 PCAP 的特征与 Stage2 目标之间存在显著差距。",
            "可能原因: (1) 14个参数不足以表达69维偏移；(2) PCAP 物理约束限制可达到的特征空间；",
            "(3) nfstream 重提取引入噪声。",
            "**建议**: 开展逐 PCAP 的 per-dimension 特征对比（需要修改 stage3_pcap_eval 保存逐维差值）。",
            "",
        ]

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="PCAP 重映射瓶颈诊断 (H1+H6)")
    parser.add_argument("--run-root", required=True, help="Pipeline run root")
    parser.add_argument("--out-dir", default="", help="Output directory (default: <run-root>/diagnosis)")
    parser.add_argument("--json", action="store_true", help="Also output full JSON")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = diagnose(run_root)

    # Write report
    narrative = result.pop("narrative", ["# Error: no results generated"])
    md_path = out_dir / "remap_bottleneck_diagnosis.md"
    md_path.write_text("\n".join(narrative) + "\n", encoding="utf-8")
    print(f"[Diagnosis] Report: {md_path}")

    if args.json:
        json_path = out_dir / "remap_bottleneck_diagnosis.json"
        json_path.write_text(
            json.dumps({k: v for k, v in result.items() if k != "feature_names"},
                       indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Diagnosis] JSON: {json_path}")

    # Print report to console
    for line in narrative:
        print(line)


if __name__ == "__main__":
    main()
