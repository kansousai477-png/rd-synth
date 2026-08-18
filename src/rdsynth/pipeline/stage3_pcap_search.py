from __future__ import annotations

import csv
import hashlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rdsynth.pipeline.stage3_ops import Stage3Settings
from rdsynth.stages.stage3_remap import MOD_NAMES, apply_mod_using_scapy, clip_modifications


@dataclass(frozen=True)
class Stage3PcapSearchResult:
    mods: np.ndarray
    pcap_written_count: int
    pcap_apply_time_sec: float
    pcap_packet_count: int
    pcap_pcaps_per_sec: float
    pcap_packet_throughput_pps: float
    pcap_selected_alpha_mean: float | None
    pcap_selected_alphas: list[float] | None
    pcap_selected_field_sets: list[str] | None
    pcap_selected_deployability_score_mean: float | None
    pcap_selected_response_l2_mean: float | None
    pcap_selected_pmal_mean: float | None
    pcap_selected_target_l2_mean: float | None
    pcap_selected_target_mae_mean: float | None
    pcap_selected_alignment_coverage_mean: float | None
    pcap_search_rounds_used_mean: float | None
    pcap_search_trace_path: str | None
    pcap_kept_original_count: int
    pcap_modified: bool
    pcap_out_dir: str

    def metrics_payload(self) -> dict[str, Any]:
        payload = {
            "pcap_written_count": int(self.pcap_written_count),
            "pcap_apply_time_sec": float(self.pcap_apply_time_sec),
            "pcap_packet_count": int(self.pcap_packet_count),
            "pcap_pcaps_per_sec": float(self.pcap_pcaps_per_sec),
            "pcap_packet_throughput_pps": float(self.pcap_packet_throughput_pps),
            "pcap_kept_original_count": int(self.pcap_kept_original_count),
            "pcap_modified": bool(self.pcap_modified),
            "pcap_out_dir": self.pcap_out_dir,
        }
        optional_values = {
            "pcap_selected_alpha_mean": self.pcap_selected_alpha_mean,
            "pcap_selected_alphas": self.pcap_selected_alphas,
            "pcap_selected_field_sets": self.pcap_selected_field_sets,
            "pcap_selected_deployability_score_mean": self.pcap_selected_deployability_score_mean,
            "pcap_selected_response_l2_mean": self.pcap_selected_response_l2_mean,
            "pcap_selected_pmal_mean": self.pcap_selected_pmal_mean,
            "pcap_selected_target_l2_mean": self.pcap_selected_target_l2_mean,
            "pcap_selected_target_mae_mean": self.pcap_selected_target_mae_mean,
            "pcap_selected_alignment_coverage_mean": self.pcap_selected_alignment_coverage_mean,
            "pcap_search_rounds_used_mean": self.pcap_search_rounds_used_mean,
            "pcap_search_trace_path": self.pcap_search_trace_path,
        }
        for key, value in optional_values.items():
            if value is not None:
                payload[key] = value
        return payload


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _temp_root() -> Path:
    root = Path(tempfile.gettempdir()) / "rdsynth_stage3"
    root.mkdir(parents=True, exist_ok=True)
    return root


def temporary_probe_workspace(prefix: str):
    return tempfile.TemporaryDirectory(prefix=prefix, dir=str(_temp_root()))


def _candidate_probe_filename(sample_idx: int, alpha: float, field_set: list[str], round_idx: int | None = None) -> str:
    field_digest = hashlib.sha1(",".join(field_set).encode("utf-8")).hexdigest()[:8]
    alpha_slug = str(alpha).replace(".", "_")
    round_slug = "" if round_idx is None else f"_r{round_idx}"
    return f"cand_{sample_idx}_{alpha_slug}{round_slug}_{field_digest}.pcap"


def _write_search_trace(path: Path, rows: list[dict[str, object]]) -> str | None:
    if not rows:
        return None
    headers = [
        "sample_idx",
        "round_idx",
        "trial_idx",
        "alpha",
        "field_set",
        "malicious_prob",
        "target_l2",
        "target_mae",
        "response_l2",
        "alignment_coverage",
        "deploy_score",
        "accepted_as_best",
        "kept_source",
        "probe_pcap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})
    return str(path)


def _candidate_mods(
    mod_row: np.ndarray,
    pcap_target_mod: np.ndarray | None,
    search_alphas: list[float],
    *,
    bidirectional: bool = False,
) -> list[tuple[float, np.ndarray]]:
    if pcap_target_mod is None:
        return [(1.0, mod_row.astype(np.float32))]
    port_idx = MOD_NAMES.index("dst_port_new")
    candidates: list[tuple[float, np.ndarray]] = []
    alpha_values = [float(alpha) for alpha in search_alphas]
    if bidirectional:
        for alpha in list(alpha_values):
            if alpha > 0.0:
                alpha_values.append(-0.5 * alpha)
        alpha_values.append(0.0)
    seen_alphas: set[float] = set()
    for alpha in alpha_values:
        alpha = float(alpha)
        alpha_key = round(alpha, 6)
        if alpha_key in seen_alphas:
            continue
        seen_alphas.add(alpha_key)
        candidate = mod_row.astype(np.float32).copy()
        for mod_idx in range(len(MOD_NAMES)):
            if mod_idx == port_idx:
                continue
            candidate[mod_idx] = float(
                pcap_target_mod[mod_idx] + alpha * (candidate[mod_idx] - pcap_target_mod[mod_idx])
            )
        candidate = clip_modifications(candidate.reshape(1, -1))[0]
        candidates.append((float(alpha), candidate.astype(np.float32)))
    return candidates


def _candidate_field_sets(
    pcap_apply_fields: list[str],
    search_field_subsets: bool,
) -> list[list[str]]:
    base = list(dict.fromkeys(pcap_apply_fields))
    if not search_field_subsets or len(base) <= 1:
        return [base]
    timing = [f for f in base if f in {"mean_iat_ms", "std_iat_ms", "flow_scale"}]
    packet_shape = [f for f in base if f in {"pad_bytes", "flag_ratio", "dst_port_new", "payload_scale"}]
    candidates = [
        base,
        timing,
        packet_shape,
        [f for f in base if f in {"pad_bytes", "payload_scale", "flow_scale"}],
        [f for f in base if f in {"mean_iat_ms", "std_iat_ms", "pad_bytes", "payload_scale", "flow_scale"}],
        [f for f in base if f in {"dst_port_new"}],
        [f for f in base if f in {"flag_ratio"}],
        [f for f in base if f in {"payload_scale"}],
    ]
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for item in candidates:
        item = [f for f in item if f in base]
        key = tuple(item)
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out or [base]


def search_and_write_pcaps(
    *,
    pkts: object,
    mods: np.ndarray,
    adv: np.ndarray | None,
    pcap_out_dir: Path,
    settings: Stage3Settings,
    seed: int,
    protocol_auto_fix: bool,
    pcap_eval_model: Any,
    search_alphas: list[float],
    pcap_target_mod: np.ndarray | None,
    orig_pmal_for_selection: float | None,
    orig_feat_pre_mean: np.ndarray | None,
    out_dir: Path,
    target_metric_fn: Callable[
        [Path, np.ndarray | None],
        tuple[float | None, float | None, float | None, float | None, dict[str, object] | None],
    ],
    wrpcap_fn,
) -> Stage3PcapSearchResult:
    search_trace_rows: list[dict[str, object]] = []

    def eval_target_metrics(
        pcap_file: Path,
        target_pre: np.ndarray | None,
        field_set: list[str],
    ) -> tuple[float | None, float | None, float | None, float | None, dict[str, object] | None]:
        try:
            return target_metric_fn(pcap_file, target_pre, field_set)
        except TypeError:
            return target_metric_fn(pcap_file, target_pre)

    def candidate_probe_score(candidate_mod: np.ndarray, field_set: list[str]) -> float:
        if pcap_target_mod is None:
            return float(-len(field_set))
        diff = np.asarray(candidate_mod, dtype=np.float64) - np.asarray(pcap_target_mod, dtype=np.float64)
        l2_error = float(np.sqrt(np.mean(diff**2)))
        return -l2_error - 0.001 * float(len(field_set))

    def candidate_deploy_score(
        candidate_mod: np.ndarray,
        pmal: float | None,
        target_l2: float | None = None,
        target_mae: float | None = None,
        response_l2: float | None = None,
        alignment_cov: float | None = None,
    ) -> float:
        if pmal is None:
            return float("-inf")
        if target_l2 is None or target_mae is None:
            target_l2 = float("nan")
            target_mae = float("nan")
        if pcap_target_mod is not None and not (np.isfinite(target_l2) and np.isfinite(target_mae)):
            diff = np.asarray(candidate_mod, dtype=np.float64) - np.asarray(pcap_target_mod, dtype=np.float64)
            target_l2 = float(np.sqrt(np.mean(diff**2)))
            target_mae = float(np.mean(np.abs(diff)))
        evasion_rate = max(0.0, min(1.0, 1.0 - float(pmal)))
        cov = float(alignment_cov) if alignment_cov is not None and np.isfinite(float(alignment_cov)) else 0.0
        base_score = 0.7 * evasion_rate + 0.3 * cov
        response_score = (
            float(np.clip(response_l2 / 2.0, 0.0, 1.0)) if response_l2 is not None and np.isfinite(response_l2) else 0.0
        )
        if orig_pmal_for_selection is None or not np.isfinite(float(orig_pmal_for_selection)):
            return 0.75 * base_score + 0.25 * response_score

        orig_pmal = float(orig_pmal_for_selection)
        improvement_score = _clip01((orig_pmal - float(pmal)) / max(orig_pmal, 1.0e-3))
        worsening_penalty = _clip01((float(pmal) - orig_pmal) / max(orig_pmal, 1.0e-3))
        return 0.55 * base_score + 0.30 * improvement_score + 0.15 * response_score - 0.35 * worsening_penalty

    def select_packets(
        mod_row: np.ndarray,
        sample_idx: int,
        target_pre: np.ndarray | None,
    ) -> tuple[np.ndarray, object, float, list[str], float | None, int, bool]:
        candidates = _candidate_mods(
            mod_row,
            pcap_target_mod,
            search_alphas,
            bidirectional=bool(getattr(settings, "pcap_search_bidirectional", True)),
        )
        field_sets = _candidate_field_sets(settings.pcap_apply_fields, settings.pcap_search_field_subsets)
        max_rounds = max(1, int(getattr(settings, "pcap_search_rounds", 1)))
        if pcap_eval_model is None or (len(candidates) == 1 and max_rounds == 1):
            alpha, best_mod = candidates[0]
            best_fields = field_sets[0]
            best_pkts = apply_mod_using_scapy(
                pkts,
                best_mod,
                seed=seed + sample_idx,
                apply_fields=best_fields,
                tcp_fixup=settings.pcap_tcp_fixup,
                dst_port_policy=settings.pcap_dst_port_policy,
                dst_port_allowlist=settings.pcap_dst_port_allowlist,
                protocol_auto_fix=protocol_auto_fix,
            )
            if orig_pmal_for_selection is not None and pcap_eval_model is not None:
                with temporary_probe_workspace("keep_") as tmp_dir:
                    tmp_pcap = Path(tmp_dir) / f"cand_keep_{sample_idx}.pcap"
                    wrpcap_fn(str(tmp_pcap), best_pkts)
                    cand_pmal, _, _, response_l2, _ = eval_target_metrics(tmp_pcap, target_pre, best_fields)
                search_trace_rows.append(
                    {
                        "sample_idx": int(sample_idx),
                        "round_idx": 0,
                        "trial_idx": 0,
                        "alpha": float(alpha),
                        "field_set": "+".join(best_fields),
                        "malicious_prob": cand_pmal if cand_pmal is not None else "",
                        "response_l2": response_l2 if response_l2 is not None else "",
                        "accepted_as_best": int(
                            cand_pmal is not None and float(cand_pmal) <= float(orig_pmal_for_selection)
                        ),
                        "kept_source": int(cand_pmal is not None and float(cand_pmal) > float(orig_pmal_for_selection)),
                        "probe_pcap": str(tmp_pcap),
                    }
                )
                if cand_pmal is not None and float(cand_pmal) > float(orig_pmal_for_selection):
                    return best_mod, pkts, -1.0, best_fields, None, 1, True
                return best_mod, best_pkts, float(alpha), best_fields, response_l2, 1, False
            return best_mod, best_pkts, float(alpha), best_fields, None, 1, False

        best_alpha = float(candidates[0][0])
        best_mod = candidates[0][1]
        best_pkts = None
        best_pmal = None
        best_score = float("-inf")
        best_fields = field_sets[0]
        best_response_l2 = None
        rounds_used = 0
        current_mod = mod_row.astype(np.float32)
        with temporary_probe_workspace("search_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            for round_idx in range(max_rounds):
                round_scale = 0.5**round_idx
                round_alphas = [float(alpha) * round_scale for alpha in search_alphas]
                candidates = _candidate_mods(
                    current_mod,
                    pcap_target_mod,
                    round_alphas,
                    bidirectional=bool(getattr(settings, "pcap_search_bidirectional", True)),
                )
                candidate_trials: list[tuple[float, np.ndarray, list[str], float]] = []
                for alpha, candidate_mod in candidates:
                    for field_set in field_sets:
                        candidate_trials.append(
                            (
                                candidate_probe_score(candidate_mod, list(field_set)),
                                candidate_mod,
                                list(field_set),
                                float(alpha),
                            )
                        )
                if settings.pcap_search_probe_topk > 0 and len(candidate_trials) > settings.pcap_search_probe_topk:
                    candidate_trials.sort(
                        key=lambda item: (float(item[0]), -len(item[2]), -float(item[3])), reverse=True
                    )
                    top_trials = candidate_trials[: settings.pcap_search_probe_topk]
                    mandatory_trials = [
                        item
                        for item in candidate_trials
                        if np.isclose(float(item[3]), 1.0) or np.isclose(float(item[3]), -0.5)
                    ]
                    trial_keys: set[tuple[float, tuple[str, ...]]] = set()
                    merged_trials = []
                    for item in [*mandatory_trials, *top_trials]:
                        key = (round(float(item[3]), 6), tuple(item[2]))
                        if key in trial_keys:
                            continue
                        trial_keys.add(key)
                        merged_trials.append(item)
                        if len(merged_trials) >= max(settings.pcap_search_probe_topk, len(mandatory_trials)):
                            break
                    candidate_trials = merged_trials
                round_improved = False
                for trial_idx, (_, candidate_mod, field_set, alpha) in enumerate(candidate_trials):
                    candidate_pkts = apply_mod_using_scapy(
                        pkts,
                        candidate_mod,
                        seed=seed + sample_idx + round_idx,
                        apply_fields=field_set,
                        tcp_fixup=settings.pcap_tcp_fixup,
                        dst_port_policy=settings.pcap_dst_port_policy,
                        dst_port_allowlist=settings.pcap_dst_port_allowlist,
                        protocol_auto_fix=protocol_auto_fix,
                    )
                    tmp_pcap = tmp_dir_path / _candidate_probe_filename(sample_idx, alpha, field_set, round_idx)
                    wrpcap_fn(str(tmp_pcap), candidate_pkts)
                    pmal, target_l2, target_mae, response_l2, meta = eval_target_metrics(
                        tmp_pcap, target_pre, field_set
                    )
                    if pmal is None:
                        continue
                    alignment_cov = None
                    if isinstance(meta, dict):
                        alignment = meta.get("alignment")
                        if isinstance(alignment, dict) and "coverage" in alignment:
                            alignment_cov = float(alignment["coverage"])
                    deploy_score = candidate_deploy_score(
                        candidate_mod,
                        pmal,
                        target_l2=target_l2,
                        target_mae=target_mae,
                        response_l2=response_l2,
                        alignment_cov=alignment_cov,
                    )
                    accepted_as_best = bool(
                        deploy_score > best_score
                        or (np.isclose(deploy_score, best_score) and (best_pmal is None or float(pmal) < best_pmal))
                    )
                    search_trace_rows.append(
                        {
                            "sample_idx": int(sample_idx),
                            "round_idx": int(round_idx),
                            "trial_idx": int(trial_idx),
                            "alpha": float(alpha),
                            "field_set": "+".join(field_set),
                            "malicious_prob": float(pmal),
                            "target_l2": target_l2 if target_l2 is not None else "",
                            "target_mae": target_mae if target_mae is not None else "",
                            "response_l2": response_l2 if response_l2 is not None else "",
                            "alignment_coverage": alignment_cov if alignment_cov is not None else "",
                            "deploy_score": float(deploy_score),
                            "accepted_as_best": int(accepted_as_best),
                            "kept_source": 0,
                            "probe_pcap": str(tmp_pcap),
                        }
                    )
                    if accepted_as_best:
                        best_pmal = float(pmal)
                        best_score = float(deploy_score)
                        best_alpha = float(alpha)
                        best_mod = candidate_mod.astype(np.float32)
                        best_pkts = candidate_pkts
                        best_fields = list(field_set)
                        best_response_l2 = (
                            float(response_l2) if response_l2 is not None and np.isfinite(response_l2) else None
                        )
                        current_mod = best_mod
                        rounds_used = round_idx + 1
                        round_improved = True
                if not round_improved:
                    break
        if (
            orig_pmal_for_selection is not None
            and best_pmal is not None
            and float(best_pmal) > float(orig_pmal_for_selection)
        ):
            return best_mod, pkts, -1.0, best_fields, None, max(1, rounds_used), True
        if best_pkts is None:
            best_pkts = apply_mod_using_scapy(
                pkts,
                best_mod,
                seed=seed + sample_idx,
                apply_fields=best_fields,
                tcp_fixup=settings.pcap_tcp_fixup,
                dst_port_policy=settings.pcap_dst_port_policy,
                dst_port_allowlist=settings.pcap_dst_port_allowlist,
                protocol_auto_fix=protocol_auto_fix,
            )
        return best_mod, best_pkts, best_alpha, best_fields, best_response_l2, max(1, rounds_used), False

    total_written_packets = 0
    selected_alphas: list[float] = []
    selected_field_sets: list[str] = []
    selected_search_scores: list[float] = []
    selected_response_scores: list[float] = []
    selected_pmals: list[float] = []
    selected_target_l2s: list[float] = []
    selected_target_maes: list[float] = []
    selected_alignment_covs: list[float] = []
    selected_rounds_used: list[int] = []
    kept_original = 0
    pcap_apply_start = time.perf_counter()
    pcap_out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(settings.pcap_apply_n, mods.shape[0])):
        target_pre = adv[i] if adv is not None and i < adv.shape[0] else None
        best_mod, new_pkts, best_alpha, best_fields, cached_response_l2, rounds_used, kept_source = select_packets(
            mods[i], i, target_pre
        )
        total_written_packets += int(len(new_pkts))
        mods[i] = best_mod
        selected_alphas.append(float(best_alpha))
        selected_field_sets.append("+".join(best_fields))
        selected_rounds_used.append(int(rounds_used))
        if pcap_target_mod is not None and best_alpha >= 0.0:
            diff = np.asarray(best_mod, dtype=np.float64) - np.asarray(pcap_target_mod, dtype=np.float64)
            l2_fidelity = float(np.sqrt(np.mean(diff**2)))
            selected_search_scores.append(-l2_fidelity)
        if kept_source:
            kept_original += 1
        out_pcap = pcap_out_dir / f"adv_{i:04d}.pcap"
        wrpcap_fn(str(out_pcap), new_pkts)
        if orig_feat_pre_mean is not None and not kept_source:
            response_l2 = cached_response_l2
            pmal, target_l2, target_mae, response_l2_eval, meta = eval_target_metrics(out_pcap, target_pre, best_fields)
            if response_l2 is None:
                response_l2 = response_l2_eval
            if pmal is not None and np.isfinite(pmal):
                selected_pmals.append(float(pmal))
            if target_l2 is not None and np.isfinite(target_l2):
                selected_target_l2s.append(float(target_l2))
            if target_mae is not None and np.isfinite(target_mae):
                selected_target_maes.append(float(target_mae))
            if isinstance(meta, dict):
                alignment = meta.get("alignment")
                if isinstance(alignment, dict) and alignment.get("coverage") is not None:
                    selected_alignment_covs.append(float(alignment["coverage"]))
            if response_l2 is not None and np.isfinite(response_l2):
                selected_response_scores.append(float(response_l2))

    pcap_apply_time_sec = time.perf_counter() - pcap_apply_start
    written_count = int(min(settings.pcap_apply_n, mods.shape[0]))
    trace_path = _write_search_trace(pcap_out_dir / "pcap_search_trace.csv", search_trace_rows)
    return Stage3PcapSearchResult(
        mods=mods,
        pcap_written_count=written_count,
        pcap_apply_time_sec=float(pcap_apply_time_sec),
        pcap_packet_count=int(total_written_packets),
        pcap_pcaps_per_sec=float(written_count / pcap_apply_time_sec) if pcap_apply_time_sec > 0.0 else float("nan"),
        pcap_packet_throughput_pps=float(total_written_packets / pcap_apply_time_sec)
        if pcap_apply_time_sec > 0.0
        else float("nan"),
        pcap_selected_alpha_mean=float(np.mean(selected_alphas)) if selected_alphas else None,
        pcap_selected_alphas=[float(alpha) for alpha in selected_alphas] if selected_alphas else None,
        pcap_selected_field_sets=selected_field_sets or None,
        pcap_selected_deployability_score_mean=float(np.mean(selected_search_scores))
        if selected_search_scores
        else None,
        pcap_selected_response_l2_mean=float(np.mean(selected_response_scores)) if selected_response_scores else None,
        pcap_selected_pmal_mean=float(np.mean(selected_pmals)) if selected_pmals else None,
        pcap_selected_target_l2_mean=float(np.mean(selected_target_l2s)) if selected_target_l2s else None,
        pcap_selected_target_mae_mean=float(np.mean(selected_target_maes)) if selected_target_maes else None,
        pcap_selected_alignment_coverage_mean=float(np.mean(selected_alignment_covs))
        if selected_alignment_covs
        else None,
        pcap_search_rounds_used_mean=float(np.mean(selected_rounds_used)) if selected_rounds_used else None,
        pcap_search_trace_path=trace_path,
        pcap_kept_original_count=int(kept_original),
        pcap_modified=True,
        pcap_out_dir=str(pcap_out_dir),
    )
