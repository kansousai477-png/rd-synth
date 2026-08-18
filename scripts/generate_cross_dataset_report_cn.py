from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASET_TITLES: dict[str, str] = {
    "nb15": "CIC NB15",
    "2017": "CIC-IDS2017",
    "2018": "CIC-IDS2018",
    "iot23": "CIC-IoT-2023",
}

FAIR_BASELINE_EXCLUDES = {"main", "global_random", "identity"}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt(value: object, digits: int = 4) -> str:
    num = _to_float(value)
    if num is None:
        return "-"
    return f"{num:.{digits}f}"


def _find_stage1_metrics(stage1_dir: Path) -> dict:
    if not stage1_dir.exists():
        return {}
    for metrics_path in sorted(stage1_dir.glob("*/metrics.json")):
        if metrics_path.parent.name == "data_quality":
            continue
        return _load_json(metrics_path)
    return {}


def _select_top_stage2_baselines(rows: list[dict[str, str]], *, limit: int = 3) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("method", "") not in FAIR_BASELINE_EXCLUDES and row.get("family") in {"baseline"}
    ]
    candidates.sort(
        key=lambda row: (
            _to_float(row.get("asr_oracle")) or float("-inf"),
            -(_to_float(row.get("norm_ffd")) or float("inf")),
        ),
        reverse=True,
    )
    return candidates[:limit]


def _select_stage3_baselines(rows: list[dict[str, str]], *, limit: int = 3) -> list[dict[str, str]]:
    candidates = [row for row in rows if row.get("method", "") != "main"]
    candidates.sort(
        key=lambda row: (
            _to_float(row.get("pcap_attack_success_rate")) or float("-inf"),
            -(_to_float(row.get("pcap_adv_prob_malicious_mean")) or float("inf")),
        ),
        reverse=True,
    )
    return candidates[:limit]


def _load_dataset_bundle(root: Path, dataset: str) -> dict:
    dataset_root = root / dataset
    main_root = dataset_root / "main"
    ablation_root = dataset_root / "ablation_ablations"
    stage1 = _find_stage1_metrics(main_root / "stage1")
    stage2 = _load_json(main_root / "stage2" / "metrics.json")
    stage3 = _load_json(main_root / "stage3" / "metrics.json")
    stage2_table = _load_csv_rows(main_root / "pipeline" / "paper_stage2_table.csv")
    stage3_table = _load_csv_rows(main_root / "pipeline" / "paper_stage3_pcap_table.csv")
    ablation_rows = _load_csv_rows(ablation_root / "ablation_summary.csv")
    return {
        "title": DATASET_TITLES.get(dataset, dataset),
        "dataset": dataset,
        "root": dataset_root,
        "main_root": main_root,
        "ablation_root": ablation_root,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage2_table": stage2_table,
        "stage3_table": stage3_table,
        "ablation_rows": ablation_rows,
    }


def _ablation_rows_with_delta(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    full = next((row for row in rows if row.get("variant") == "full"), None)
    if full is None:
        return rows
    full_s2 = _to_float(full.get("stage2_decision_score"))
    full_s3 = _to_float(full.get("stage3_decision_score"))
    out = []
    for row in rows:
        record = dict(row)
        s2 = _to_float(row.get("stage2_decision_score"))
        s3 = _to_float(row.get("stage3_decision_score"))
        record["delta_stage2"] = "" if s2 is None or full_s2 is None else f"{s2 - full_s2:.4f}"
        record["delta_stage3"] = "" if s3 is None or full_s3 is None else f"{s3 - full_s3:.4f}"
        out.append(record)
    out.sort(key=lambda item: item.get("variant", ""))
    return out


def _write_report(out_path: Path, bundles: list[dict]) -> None:
    lines: list[str] = [
        "# RDSynth 跨数据集实验报告（中文版）",
        "",
        "## 1. 实验说明",
        "",
        "本轮实验以“数据集整体二分类”为单位，对四个数据集分别执行完整的 `Stage1 -> Stage2 -> Stage3` 流水线。",
        "Stage3 按当前实现边界只做离线 PCAP 重放：不做 online 重新发包，只从 `data/PCAPs/` 中自动筛选恶意 PCAP 进行重放与评估。",
        "主实验打开了 Stage2 baseline 和 Stage3 baseline 对比；消融实验关闭 baseline 以聚焦模块变化本身。",
        "",
        "## 2. 跨数据集总览",
        "",
        "| 数据集 | Stage1 决策分 | Stage1 一致率 | Stage1 基线一致率 | Stage2 决策分 | Stage2 ASR(oracle) | Stage3 决策分 | Stage3 重映射分 | Stage3 可部署分 | 选中 PCAP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for bundle in bundles:
        s1 = bundle["stage1"]
        s2 = bundle["stage2"]
        s3 = bundle["stage3"]
        lines.append(
            "| "
            + " | ".join(
                [
                    bundle["title"],
                    _fmt(s1.get("stage1_decision_score")),
                    _fmt(s1.get("agreement")),
                    _fmt(s1.get("baseline_agreement")),
                    _fmt(s2.get("stage2_decision_score")),
                    _fmt(s2.get("asr_oracle")),
                    _fmt(s3.get("stage3_decision_score")),
                    _fmt(s3.get("stage3_decision_remap_quality_score")),
                    _fmt(s3.get("stage3_decision_pcap_deployability_score")),
                    str(s3.get("pcap_selected_name", "-")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### 总体观察",
            "",
            "- `Stage1` 可以直接看 surrogate 提取是否明显优于同预算基线；`agreement - baseline_agreement` 越大，说明主动查询设计越有效。",
            "- `Stage2` 重点看 `stage2_decision_score` 与 `asr_oracle`；若 ASR 很高但决策分不高，通常意味着攻击有效但偏离良性分布较大。",
            "- `Stage3` 在本轮只代表“离线 PCAP 重放下的包级可实现性”，不是 online 发包闭环结果。",
            "",
            "## 3. 分数据集结果",
            "",
        ]
    )

    for idx, bundle in enumerate(bundles, start=1):
        title = bundle["title"]
        s1 = bundle["stage1"]
        s2 = bundle["stage2"]
        s3 = bundle["stage3"]
        stage2_rows = bundle["stage2_table"]
        stage3_rows = bundle["stage3_table"]
        ablation_rows = _ablation_rows_with_delta(bundle["ablation_rows"])
        main_stage2 = next((row for row in stage2_rows if row.get("method") == "main"), {})
        main_stage3 = next((row for row in stage3_rows if row.get("method") == "main"), {})
        top_stage2_baselines = _select_top_stage2_baselines(stage2_rows)
        top_stage3_baselines = _select_stage3_baselines(stage3_rows)

        agreement = _to_float(s1.get("agreement"))
        baseline_agreement = _to_float(s1.get("baseline_agreement"))
        agreement_gain = None
        if agreement is not None and baseline_agreement is not None:
            agreement_gain = agreement - baseline_agreement

        lines.extend(
            [
                f"### 3.{idx} {title}",
                "",
                f"- Stage1：决策分 `{_fmt(s1.get('stage1_decision_score'))}`，surrogate 与 oracle 一致率 `{_fmt(s1.get('agreement'))}`，相对 Stage1 基线提升 `{_fmt(agreement_gain)}`。",
                f"- Stage2：决策分 `{_fmt(s2.get('stage2_decision_score'))}`，ASR(oracle) `{_fmt(s2.get('asr_oracle'))}`，FFD `{_fmt(s2.get('norm_FFD'))}`，说明当前攻击{'更偏向强攻击' if (_to_float(s2.get('stage2_decision_attack_effectiveness_score')) or 0.0) > (_to_float(s2.get('stage2_decision_fidelity_score')) or 0.0) else '在攻击性和保真性之间相对均衡'}。",
                f"- Stage3：决策分 `{_fmt(s3.get('stage3_decision_score'))}`，重映射质量 `{_fmt(s3.get('stage3_decision_remap_quality_score'))}`，可部署性 `{_fmt(s3.get('stage3_decision_pcap_deployability_score'))}`，离线选中的 PCAP 为 `{s3.get('pcap_selected_name', '-')}`。",
                "",
                "#### Stage2 与 Baseline 对比",
                "",
                "| 方法 | 家族 | ASR(oracle) | 决策分 | FFD | SWD | 端到端时间(s) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                "| "
                + " | ".join(
                    [
                        "main",
                        "ours",
                        _fmt(main_stage2.get("asr_oracle")),
                        _fmt(main_stage2.get("decision_score")),
                        _fmt(main_stage2.get("norm_ffd")),
                        _fmt(main_stage2.get("norm_swd")),
                        _fmt(main_stage2.get("end_to_end_time_sec")),
                    ]
                )
                + " |",
            ]
        )
        for row in top_stage2_baselines:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("method", "")),
                        str(row.get("family", "")),
                        _fmt(row.get("asr_oracle")),
                        _fmt(row.get("decision_score")),
                        _fmt(row.get("norm_ffd")),
                        _fmt(row.get("norm_swd")),
                        _fmt(row.get("end_to_end_time_sec")),
                    ]
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "#### Stage3 离线 PCAP Baseline 对比",
                "",
                "| 方法 | 家族 | Stage3 决策分 | PCAP 攻击成功率 | 检测率 | Adv 恶意概率均值 | Target L2 | 对齐覆盖率 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| "
                + " | ".join(
                    [
                        "main",
                        "ours",
                        _fmt(main_stage3.get("decision_score")),
                        _fmt(main_stage3.get("pcap_attack_success_rate")),
                        _fmt(main_stage3.get("pcap_detection_rate")),
                        _fmt(main_stage3.get("pcap_adv_prob_malicious_mean")),
                        _fmt(main_stage3.get("pcap_target_l2_mean")),
                        _fmt(main_stage3.get("pcap_alignment_coverage")),
                    ]
                )
                + " |",
            ]
        )
        for row in top_stage3_baselines:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("method", "")),
                        str(row.get("family", "")),
                        _fmt(row.get("decision_score")),
                        _fmt(row.get("pcap_attack_success_rate")),
                        _fmt(row.get("pcap_detection_rate")),
                        _fmt(row.get("pcap_adv_prob_malicious_mean")),
                        _fmt(row.get("pcap_target_l2_mean")),
                        _fmt(row.get("pcap_alignment_coverage")),
                    ]
                )
                + " |"
            )

        if ablation_rows:
            lines.extend(
                [
                    "",
                    "#### 模块消融",
                    "",
                    "| 变体 | Stage2 决策分 | 相对 full | Stage3 决策分 | 相对 full | 重映射质量 | 可部署性 | pcap fatal rate |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in ablation_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("variant", "")),
                            _fmt(row.get("stage2_decision_score")),
                            str(row.get("delta_stage2") or "-"),
                            _fmt(row.get("stage3_decision_score")),
                            str(row.get("delta_stage3") or "-"),
                            _fmt(row.get("stage3_remap_quality_score")),
                            _fmt(row.get("stage3_deployability_score")),
                            _fmt(row.get("pcap_valid_fatal_rate")),
                        ]
                    )
                    + " |"
                )

        lines.extend(
            [
                "",
                "#### 结果解读",
                "",
            ]
        )
        if agreement_gain is not None and agreement_gain > 0:
            lines.append(f"- `Stage1` 在 {title} 上相对基线有正增益，说明主动查询策略是有价值的。")
        else:
            lines.append(f"- `Stage1` 在 {title} 上对基线优势有限，后续若要继续提升，优先检查查询池和 surrogate 容量。")

        if (_to_float(s3.get("stage3_decision_pcap_deployability_score")) or 0.0) < (
            _to_float(s3.get("stage3_decision_remap_quality_score")) or 0.0
        ):
            lines.append("- Stage3 呈现“重映射拟合强于最终可部署性”的典型形态，瓶颈更像是包级约束而不是 remapper 本身。")
        else:
            lines.append("- Stage3 的 remap 质量和可部署性比较接近，说明当前离线 PCAP 映射没有明显卡在协议修复阶段。")

        if str(s3.get("stage3_decision_score_scope", "")).strip() == "remap_only":
            reason = str(s3.get("stage3_decision_score_block_reason") or s3.get("pcap_feature_quality_block_reason") or "-")
            lines.append(f"- 当前 {title} 的 Stage3 只能按 `remap_only` 解读，不能当作完整包级部署结论；直接原因是 `{reason}`。")

        if ablation_rows:
            worst = min(
                ablation_rows,
                key=lambda row: _to_float(row.get("delta_stage3")) if _to_float(row.get("delta_stage3")) is not None else float("inf"),
            )
            lines.append(
                f"- 对 {title} 影响最大的消融是 `{worst.get('variant', '-')}`，它对 Stage3 的相对变化为 `{worst.get('delta_stage3') or '-'}`。"
            )

        lines.extend(
            [
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 结论",
            "",
            "- 四个数据集都已经完成 Stage1、Stage2、Stage3 主实验；Stage3 明确限定为 `data/PCAPs/` 下的离线重放，不包含 online 重发。",
            "- 主实验已经包含 baseline 对比：Stage2 用特征空间 baseline，Stage3 用可重放 traffic-space baseline。",
            "- 消融实验覆盖了 surrogate guidance、生成骨干、remap 策略、协议自动修复四类核心模块，足以回答当前实现中哪些模块对最终 Stage3 更关键。",
            "- 若后续要继续扩展，本轮最值得补的是多 seed 重复与真正的 online 发包闭环；在这之前，当前报告更适合作为“离线可实现性”证据，而不是最终部署结论。",
            "",
            "## 5. 产物路径",
            "",
            "- 主报告：`"
            + str(out_path).replace("\\", "/")
            + "`",
            "- 各数据集主实验目录：`<root>/<dataset>/main/`",
            "- 各数据集消融目录：`<root>/<dataset>/ablation_ablations/`",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Chinese Markdown report for dataset-level RDSynth experiments.")
    parser.add_argument("--root", default="outputs/reviewer_suite/dataset_suite", help="Experiment root directory.")
    parser.add_argument("--datasets", default="nb15,2017,2018,iot23", help="Comma-separated dataset list.")
    parser.add_argument(
        "--out",
        default="outputs/reviewer_suite/dataset_suite/EXPERIMENT_REPORT_CN.md",
        help="Output Markdown path.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()]
    bundles = [_load_dataset_bundle(root, dataset) for dataset in datasets]
    _write_report(Path(args.out).resolve(), bundles)


if __name__ == "__main__":
    main()
