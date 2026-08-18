from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rdsynth.pipeline.stage3_ops import Stage3Settings
from rdsynth.pipeline.stage3_pcap_semantics import (
    categories_for_attack,
    categories_for_attacks,
    filter_candidates_by_categories,
)


@dataclass(frozen=True)
class Stage3PcapSelection:
    pcap_path_cfg: str
    pcap_scan_dir_cfg: str
    scan_dir: Path | None
    scan_glob: str
    scan_limit: int
    scan_min_prob: float
    scan_compare_existing: bool
    scan_pmal_weight: float
    scan_target_fit_weight: float
    scan_target_mod_fit_weight: float
    semantic_filter: bool
    semantic_dataset: str
    semantic_attack_label: str
    semantic_categories: list[str]
    selected_path: Path | None
    selected_source: str
    selected_prob: float | None
    evasion_valid: bool | None
    candidate_paths: list[Path]
    source_selection_mode: str
    source_sample_n: int
    source_sample_seed: int


def build_pcap_selection(settings: Stage3Settings, metrics_payload: dict[str, Any]) -> Stage3PcapSelection:
    pcap_path_cfg = settings.pcap_path
    pcap_path = Path(pcap_path_cfg) if pcap_path_cfg else None
    selected_path = pcap_path if pcap_path is not None and pcap_path.exists() else None
    selected_source = "config" if selected_path is not None else ""
    scan_dir = Path(settings.pcap_scan_dir) if settings.pcap_scan_dir else None

    metrics_payload["pcap_scan_min_prob"] = settings.pcap_scan_min_prob
    metrics_payload["pcap_scan_limit"] = settings.pcap_scan_limit
    metrics_payload["pcap_source_selection_mode"] = settings.pcap_source_selection_mode
    metrics_payload["pcap_source_sample_n"] = int(settings.pcap_source_sample_n)
    semantic_categories: list[str] = []
    if settings.pcap_semantic_filter:
        semantic_labels = [label for label in settings.pcap_attack_labels if str(label).strip()]
        if semantic_labels:
            semantic_categories = categories_for_attacks(settings.pcap_dataset, semantic_labels)
        else:
            semantic_categories = categories_for_attack(settings.pcap_dataset, settings.pcap_attack_label)
    metrics_payload["pcap_semantic_filter"] = bool(settings.pcap_semantic_filter)
    metrics_payload["pcap_semantic_dataset"] = settings.pcap_dataset
    metrics_payload["pcap_semantic_attack_label"] = settings.pcap_attack_label
    metrics_payload["pcap_semantic_attack_labels"] = list(settings.pcap_attack_labels)
    metrics_payload["pcap_semantic_categories"] = list(semantic_categories)
    if scan_dir is not None:
        metrics_payload["pcap_scan_dir"] = str(scan_dir)

    return Stage3PcapSelection(
        pcap_path_cfg=pcap_path_cfg,
        pcap_scan_dir_cfg=settings.pcap_scan_dir,
        scan_dir=scan_dir,
        scan_glob=settings.pcap_scan_glob,
        scan_limit=settings.pcap_scan_limit,
        scan_min_prob=settings.pcap_scan_min_prob,
        scan_compare_existing=settings.pcap_scan_compare_existing,
        scan_pmal_weight=settings.pcap_scan_pmal_weight,
        scan_target_fit_weight=settings.pcap_scan_target_fit_weight,
        scan_target_mod_fit_weight=settings.pcap_scan_target_mod_fit_weight,
        semantic_filter=settings.pcap_semantic_filter,
        semantic_dataset=settings.pcap_dataset,
        semantic_attack_label=settings.pcap_attack_label,
        semantic_categories=semantic_categories,
        selected_path=selected_path,
        selected_source=selected_source,
        selected_prob=None,
        evasion_valid=None,
        candidate_paths=[selected_path] if selected_path is not None else [],
        source_selection_mode=settings.pcap_source_selection_mode,
        source_sample_n=int(settings.pcap_source_sample_n),
        source_sample_seed=int(settings.pcap_source_sample_seed),
    )


def resolve_selected_pcap(
    selection: Stage3PcapSelection,
    *,
    pcap_features: Any,
    pcap_eval_model: Any,
    target_pre: np.ndarray | None,
    target_mod: np.ndarray | None,
    metrics_payload: dict[str, Any],
    mandatory_paths: list[str] | None = None,
) -> Stage3PcapSelection:
    selected_path = selection.selected_path
    selected_source = selection.selected_source
    selected_prob = selection.selected_prob
    evasion_valid = selection.evasion_valid
    candidate_paths = list(selection.candidate_paths)

    if pcap_eval_model is not None:
        if selected_path is not None:
            selected_prob, _, _, _ = pcap_features.pcap_prob(selected_path)
        needs_scan = (
            selected_path is None
            or selection.scan_compare_existing
            or selected_prob is None
            or selected_prob < selection.scan_min_prob
        )
        if needs_scan and selection.scan_dir is not None and selection.scan_dir.exists():
            ranked = pcap_features.rank_pcaps(
                selection.scan_dir,
                selection.scan_glob,
                selection.scan_limit,
                target_pre=target_pre,
                target_mod=target_mod,
                pmal_weight=selection.scan_pmal_weight,
                target_fit_weight=selection.scan_target_fit_weight,
                target_mod_fit_weight=selection.scan_target_mod_fit_weight,
                semantic_categories=selection.semantic_categories,
                mandatory_paths=mandatory_paths,
            )
            metrics_payload["pcap_scan_count"] = len(ranked)
            skipped_rows = [row for row in ranked if row.get("skip_reason")]
            valid_ranked = [
                row for row in ranked if not row.get("skip_reason") and row.get("prob_malicious") is not None
            ]
            hard_ranked = [
                row
                for row in valid_ranked
                if (
                    row.get("pred_label") == 1
                    or (
                        row.get("prob_malicious") is not None
                        and float(row.get("prob_malicious") or 0.0) >= float(selection.scan_min_prob)
                    )
                )
            ]
            metrics_payload["pcap_scan_skipped_count"] = len(skipped_rows)
            metrics_payload["pcap_source_hard_candidate_count"] = int(len(hard_ranked))
            if selection.semantic_categories:
                metrics_payload["pcap_scan_semantic_categories"] = list(selection.semantic_categories)
            if ranked:
                candidate_paths = []
                for row in hard_ranked or valid_ranked:
                    try:
                        candidate_path = Path(str(row["path"]))
                    except Exception as exc:
                        print(f"[Stage3/Sel][Warn] invalid pcap path in row: {exc}")
                        continue
                    candidate_paths.append(candidate_path)
                metrics_payload["pcap_scan_top_candidates"] = [
                    {
                        "name": row.get("name", ""),
                        "prob_malicious": row.get("prob_malicious"),
                        "pred_label": row.get("pred_label"),
                        "selection_score": row.get("selection_score"),
                        "target_feature_l2": row.get("target_feature_l2"),
                        "target_mod_l2": row.get("target_mod_l2"),
                        "status": row.get("status"),
                        "skip_reason": row.get("skip_reason"),
                        "pcap_size_bytes": row.get("pcap_size_bytes"),
                    }
                    for row in ranked[: min(5, len(ranked))]
                ]
                if selection.source_selection_mode in {"random", "random_hard", "all", "top_hard"}:
                    rng_seed = selection.source_sample_seed if selection.source_sample_seed > 0 else 0
                    rng = np.random.default_rng(rng_seed)
                    ranked_rows = list(valid_ranked)
                    if selection.source_selection_mode in {"random_hard", "top_hard"}:
                        hard_rows = [
                            row
                            for row in ranked_rows
                            if (
                                row.get("pred_label") == 1
                                or (
                                    row.get("prob_malicious") is not None
                                    and float(row.get("prob_malicious") or 0.0) >= float(selection.scan_min_prob)
                                )
                            )
                        ]
                        metrics_payload["pcap_source_hard_candidate_count"] = int(len(hard_rows))
                        if hard_rows:
                            ranked_rows = hard_rows
                            metrics_payload["pcap_source_hard_filter_applied"] = True
                        else:
                            metrics_payload["pcap_source_hard_filter_applied"] = False
                    hard_candidate_paths: list[Path] = []
                    for row in ranked_rows:
                        try:
                            hard_candidate_paths.append(Path(str(row["path"])))
                        except Exception as exc:
                            print(f"[Stage3/Sel][Warn] invalid hard candidate path: {exc}")
                            continue
                    if selection.source_selection_mode == "all" or int(selection.source_sample_n) <= 0:
                        candidate_paths = list(hard_candidate_paths)
                    elif selection.source_selection_mode == "top_hard":
                        take_n = min(len(hard_candidate_paths), max(1, int(selection.source_sample_n)))
                        candidate_paths = list(hard_candidate_paths[:take_n])
                    else:
                        take_n = min(len(hard_candidate_paths), max(1, int(selection.source_sample_n)))
                        sampled_idx = sorted(rng.choice(len(hard_candidate_paths), size=take_n, replace=False).tolist())
                        candidate_paths = [hard_candidate_paths[idx] for idx in sampled_idx]
                    metrics_payload["pcap_source_sampled_names"] = [path.name for path in candidate_paths]
                    if candidate_paths:
                        selected_path = candidate_paths[0]
                        if selection.source_selection_mode == "all":
                            selected_source = "scan_all"
                        elif selection.source_selection_mode == "top_hard":
                            selected_source = "scan_top_hard"
                        else:
                            selected_source = (
                                "scan_random_hard"
                                if selection.source_selection_mode == "random_hard"
                                else "scan_random"
                            )
                        try:
                            selected_prob, _, _, _ = pcap_features.pcap_prob(selected_path)
                        except Exception as exc:
                            print(f"[Stage3/Sel][Warn] pcap_prob failed for {selected_path}: {exc}")
                            selected_prob = None
                else:
                    if not valid_ranked:
                        metrics_payload["pcap_skip_reason"] = "pcap_scan_no_valid_candidates"
                        selected_path = None
                        evasion_valid = False
                    elif selection.scan_min_prob > 0.0 and not hard_ranked:
                        # ── Fallback: no PCAP reached the strict pmal threshold ──
                        # Use the highest-pmal valid candidate instead of failing.
                        # AI-classified PCAP directories may contain mismatched malware
                        # families that score lower on this dataset's NIDS.
                        best_valid = valid_ranked[0]
                        fallback_path = Path(str(best_valid["path"]))
                        fallback_pmal = float(best_valid.get("prob_malicious", 0))
                        print(
                            f"[Stage3/Sel] no PCAP reached pmal≥{selection.scan_min_prob:.2f}; "
                            f"falling back to best candidate {fallback_path.name} (pmal={fallback_pmal:.4f})"
                        )
                        metrics_payload["pcap_scan_fallback_applied"] = True
                        metrics_payload["pcap_scan_fallback_reason"] = (
                            f"no source reached pmal≥{selection.scan_min_prob:.2f}; "
                            f"using {fallback_path.name} (pmal={fallback_pmal:.4f})"
                        )
                        selected_path = fallback_path
                        selected_source = "scan_fallback"
                        try:
                            selected_prob, _, _, _ = pcap_features.pcap_prob(selected_path)
                        except Exception:
                            selected_prob = fallback_pmal
                    else:
                        best = (hard_ranked or valid_ranked)[0]
                        best_path = Path(str(best["path"]))
                        best_prob = float(best["prob_malicious"]) if best.get("prob_malicious") is not None else None
                        best_score = best.get("selection_score")
                        existing_score = None
                        if selected_path is not None:
                            current = pcap_features.score_pcap_candidate(
                                selected_path,
                                target_pre=target_pre,
                                target_mod=target_mod,
                                pmal_weight=selection.scan_pmal_weight,
                                target_fit_weight=selection.scan_target_fit_weight,
                                target_mod_fit_weight=selection.scan_target_mod_fit_weight,
                            )
                            existing_score = current.get("selection_score")
                        if best_path is not None and (
                            existing_score is None
                            or (best_score is not None and float(best_score) > float(existing_score))
                        ):
                            selected_path = best_path
                            selected_prob = best_prob
                            selected_source = "scan"
    else:
        if selected_path is None and selection.scan_dir is not None and selection.scan_dir.exists():
            candidates = sorted(selection.scan_dir.rglob(selection.scan_glob))
            if selection.semantic_categories:
                candidates = sorted(filter_candidates_by_categories(candidates, selection.semantic_categories))
            if candidates:
                selected_path = candidates[0]
                selected_source = "scan"

    if selected_path is not None:
        metrics_payload["pcap_selected_path"] = str(selected_path)
        metrics_payload["pcap_selected_name"] = selected_path.name
        metrics_payload["pcap_selected_source"] = selected_source
        if selected_prob is not None:
            metrics_payload["pcap_selected_prob_malicious"] = float(selected_prob)
            evasion_valid = selected_prob >= selection.scan_min_prob
            metrics_payload["pcap_evasion_valid"] = bool(evasion_valid)
            if not evasion_valid:
                metrics_payload["pcap_skip_reason"] = "source_already_evasive"
    elif selection.pcap_path_cfg or selection.pcap_scan_dir_cfg:
        metrics_payload.setdefault("pcap_skip_reason", "pcap_not_found")

    return replace(
        selection,
        selected_path=selected_path,
        selected_source=selected_source,
        selected_prob=selected_prob,
        evasion_valid=evasion_valid,
        candidate_paths=candidate_paths,
        source_selection_mode=selection.source_selection_mode,
        source_sample_n=selection.source_sample_n,
        source_sample_seed=selection.source_sample_seed,
    )
