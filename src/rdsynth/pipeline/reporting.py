"""Pipeline reporting: I/O, formatting, tables, and baseline leaderboard.

Large sub-domains live in:
  * reporting_baselines.py   — baseline leaderboard / summary records
  * reporting_paper_tables.py — paper-format stage2 / stage3 table records
All public names are re-exported here so existing callers are unaffected.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from rdsynth.pipeline.reporting_utils import fmt_value, maybe_float  # noqa: F401 — re-exported for sub-modules

# ── shared constants ─────────────────────────────────────────────────────────
_TRAFFIC_SPACE_BASELINES = {"gpmt_lite", "progen_lite", "amoeba_lite", "netdiffusion_lite"}
_CONTROL_BASELINES = {"global_random"}
_OUR_METHOD_NAME = "RDSynth"
_CONTROL_METHODS = {"identity", "global_random", "knn_benign"}
_STANDARD_ATTACK_METHODS = {"fgsm", "pgd"}


# ── name helpers ─────────────────────────────────────────────────────────────
def display_method_name(name: object) -> str:
    token = str(name or "").strip()
    if token.lower() in {"main", "ours", "rdsynth"}:
        return _OUR_METHOD_NAME
    return token


def display_family_name(name: object) -> str:
    token = str(name or "").strip()
    if token.lower() in {"ours", "rdsynth"}:
        return _OUR_METHOD_NAME
    return token


def baseline_credibility_level(name: object) -> str:
    token = str(name or "").strip().lower()
    if token in _CONTROL_METHODS:
        return "control"
    if token in _STANDARD_ATTACK_METHODS:
        return "standard"
    if token.endswith("_lite"):
        return "lite"
    if token:
        return "unclassified"
    return ""


# ── I/O helpers ──────────────────────────────────────────────────────────────
def write_config(cfg: dict, out_dir: Path, *, yaml_module: object) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "pipeline_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml_module.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── PCAP eval summary ───────────────────────────────────────────────────────
def pcap_eval_summary(eval_csv: Path, pcap_path: Path | None) -> dict:
    if not eval_csv.exists():
        return {}
    rows = []
    with open(eval_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    orig_row = None
    if pcap_path is not None:
        for row in rows:
            if row.get("pcap") == str(pcap_path):
                orig_row = row
                break
    if orig_row is None:
        for row in rows:
            name = Path(row.get("pcap", "")).name
            if not name.startswith("adv_"):
                orig_row = row
                break

    adv_rows = []
    for row in rows:
        name = Path(row.get("pcap", "")).name
        if name.startswith("adv_"):
            adv_rows.append(row)

    def _as_float(row: dict, key: str) -> float | None:
        if row is None:
            return None
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return None

    out = {}
    orig_pmal = _as_float(orig_row, "prob_malicious")
    if orig_pmal is not None:
        out["pcap_orig_prob_malicious"] = orig_pmal
    if adv_rows:
        pmal = [float(r.get("prob_malicious", 0.0)) for r in adv_rows]
        preds = [int(float(r.get("pred_label", 0))) for r in adv_rows]
        out["pcap_adv_prob_malicious_mean"] = float(sum(pmal) / len(pmal))
        out["pcap_adv_prob_malicious_min"] = float(min(pmal))
        out["pcap_adv_prob_malicious_max"] = float(max(pmal))
        out["pcap_adv_pred_malicious_rate"] = float(sum(preds) / len(preds))
    return out


# ── table printing ───────────────────────────────────────────────────────────
def print_table(rows: list[tuple[str, str, str]]) -> str:
    headers = ("Stage", "Metric", "Value")
    all_rows = [headers] + rows
    widths = [0, 0, 0]
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        f"{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  {headers[2]:<{widths[2]}}",
        f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}",
    ]
    for stage, metric, value in rows:
        lines.append(f"{stage:<{widths[0]}}  {metric:<{widths[1]}}  {value:<{widths[2]}}")
    return "\n".join(lines)


# ── wide-record / CSV helpers ───────────────────────────────────────────────
def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "metric"


def make_wide_record(
    rows: list[tuple[str, str, str]],
    cfg: dict,
    oracle_name: str,
) -> dict[str, str]:
    record: dict[str, str] = {
        "project_out_dir": str(cfg["project"]["out_dir"]),
        "oracle_name": oracle_name,
        "dataset": str(cfg.get("data", {}).get("dataset", "")),
    }
    for stage, metric, value in rows:
        key = f"{_slugify(stage)}__{_slugify(metric)}"
        record[key] = value
    return record


def write_dict_csv(
    path: Path,
    records: list[dict[str, str]],
    fieldnames: list[str] | None = None,
    append_extra_fields: bool = True,
) -> None:
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for record in records:
            for key in record.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    elif append_extra_fields:
        extras = []
        seen = set(fieldnames)
        for record in records:
            for key in record.keys():
                if key not in seen:
                    seen.add(key)
                    extras.append(key)
        fieldnames = [*fieldnames, *extras]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore" if not append_extra_fields else "raise",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


# ── field ordering ───────────────────────────────────────────────────────────
def _sort_metric_keys(keys: list[str]) -> list[str]:
    stage_order = {
        "data__": 0,
        "stage1__": 1,
        "stage2__": 2,
        "stage2_bl__": 3,
        "stage3__": 4,
        "stage3_pcap__": 5,
        "stage3_pcap_bl__": 6,
        "stage2_paper__": 7,
        "stage2_bl_paper__": 8,
        "stage3_paper__": 9,
        "stage3_bl_paper__": 10,
    }

    def _group_rank(key: str) -> tuple[int, str]:
        for prefix, rank in stage_order.items():
            if key.startswith(prefix):
                return rank, key
        return 99, key

    return sorted(keys, key=_group_rank)


def wide_fieldnames(record: dict[str, str]) -> list[str]:
    base = ["project_out_dir", "dataset", "oracle_name"]
    metric_keys = [key for key in record.keys() if key not in set(base)]
    return [*base, *_sort_metric_keys(metric_keys)]


def overview_fieldnames(record: dict[str, str]) -> list[str]:
    preferred = [
        "project_out_dir",
        "dataset",
        "oracle_name",
        "data__rows",
        "data__features",
        "data__label_positive_rate",
        "stage1__oracle_eval_acc",
        "stage1__oracle_eval_f1",
        "stage1__agreement",
        "stage1__surrogate_val_acc",
        "stage1__surrogate_val_f1",
        "stage2__asr_surrogate",
        "stage2__asr_oracle",
        "stage2__norm_ffd",
        "stage2__norm_swd",
        "stage2__norm_advtomal_l2",
        "stage2__adv_prob_malicious_mean",
        "stage2__adv_prob_malicious_mean_oracle",
        "stage3__adv_benign_rate",
        "stage3__adv_prob_malicious_mean",
        "stage3__remap_use_direct",
        "stage3_pcap__pcap_modified",
        "stage3_pcap__pcap_skip_reason",
        "stage3_pcap__pcap_evasion_valid",
        "stage3_pcap__pcap_orig_prob_malicious",
        "stage3_paper__paper_pcap_alignment_coverage",
    ]
    return [key for key in preferred if key in record]


def select_overview_rows(rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    preferred_metrics = {
        ("Data", "rows"),
        ("Data", "features"),
        ("Data", "label_positive_rate"),
        ("Stage1", "oracle_eval_acc"),
        ("Stage1", "oracle_eval_f1"),
        ("Stage1", "agreement"),
        ("Stage1", "surrogate_val_acc"),
        ("Stage2", "asr_surrogate"),
        ("Stage2", "asr_oracle"),
        ("Stage2", "norm_FFD"),
        ("Stage2", "norm_SWD"),
        ("Stage2", "norm_AdvToMal_L2"),
        ("Stage3", "adv_benign_rate"),
        ("Stage3", "adv_prob_malicious_mean"),
        ("Stage3", "remap_use_direct"),
        ("Stage3/PCAP", "pcap_modified"),
        ("Stage3/PCAP", "pcap_skip_reason"),
        ("Stage3/PCAP", "pcap_evasion_valid"),
        ("Stage3/PCAP", "pcap_orig_prob_malicious"),
    }
    selected = [row for row in rows if (row[0], row[1]) in preferred_metrics]
    return selected if selected else rows


# ── re-exports from sub-modules (at bottom to avoid circular imports) ────────
from rdsynth.pipeline.reporting_baselines import (  # noqa: E402, F401
    baseline_fieldnames,
    baseline_leaderboard_rows,
    collect_baseline_summary_records,
    leaderboard_fieldnames,
    make_leaderboard_records,
    print_baseline_table,
)
from rdsynth.pipeline.reporting_paper_tables import (  # noqa: E402, F401
    _append_metric_rows,
    collect_paper_summary_rows,
    make_stage2_paper_table_records,
    make_stage3_pcap_table_records,
    stage2_paper_fieldnames,
    stage3_evidence_summary,
    stage3_pcap_fieldnames,
)
