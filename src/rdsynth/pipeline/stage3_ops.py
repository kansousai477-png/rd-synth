from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rdsynth.stages.stage3_remap import MOD_NAMES, apply_mod_using_scapy, clip_modifications
from rdsynth.utils.checkpoints import OracleRestoreData, load_stage1_artifacts


@dataclass(frozen=True)
class Stage3Environment:
    scapy_available: bool
    nfstream_available: bool
    cicflowmeter_available: bool


@dataclass(frozen=True)
class Stage3ModelSelection:
    pcap_eval_model: Any
    pcap_eval_model_name: str


@dataclass(frozen=True)
class Stage3Settings:
    remap_train_source: str
    remap_train_objective: str
    loss: str
    huber_beta: float
    grad_clip: float
    target_clip_sigma: float
    weight_decay: float
    remap_mode: str
    remap_min_r2: float
    remap_blend_alpha: float
    remap_collapse_ratio_threshold: float
    epochs: int
    batch_size: int
    lr: float
    protocol_auto_fix: bool
    feature_aliases_path: str
    feature_backend: str
    ids_name: str
    oracle_name: str
    pcap_eval_use_ids: bool
    pcap_eval_use_oracle: bool
    pcap_align_min_coverage: float
    pcap_eval_batch_size: int
    pcap_cache_enable: bool
    pcap_cache_dir: str
    pcap_feature_fail_closed: bool
    pcap_feature_fail_on_partial_alignment: bool
    pcap_search_alphas: list[float]
    pcap_path: str
    pcap_scan_dir: str
    pcap_scan_limit: int
    pcap_scan_min_prob: float
    pcap_scan_glob: str
    pcap_scan_compare_existing: bool
    pcap_scan_pmal_weight: float
    pcap_scan_target_fit_weight: float
    pcap_scan_target_mod_fit_weight: float
    pcap_scan_max_bytes: int
    pcap_feature_max_flows_per_pcap: int | None
    pcap_semantic_filter: bool
    pcap_dataset: str
    pcap_attack_label: str
    pcap_attack_labels: list[str]
    pcap_source_selection_mode: str
    pcap_source_sample_n: int
    pcap_source_sample_seed: int
    pcap_target_source: str
    adv_samples_path: str
    copy_adv_samples: bool
    pcap_apply_fields: list[str]
    pcap_search_bidirectional: bool
    pcap_search_field_subsets: bool
    pcap_search_probe_topk: int
    pcap_search_rounds: int
    pcap_tcp_fixup: bool
    pcap_dst_port_policy: str
    pcap_dst_port_allowlist: list[int]
    pcap_out_dir: str
    pcap_apply_n: int
    pcap_multi_step_max_rounds: int  # 0=disabled, N=up to N additional remap rounds
    pcap_eval: bool
    pcap_compare_baselines: bool
    pcap_baseline_jobs: int
    save_intermediate_results: bool
    pcap_ids_benign_path: str
    pcap_ids_benign_paths: list[str]
    pcap_mandatory_paths: list[str]
    pcap_ids_benign_dir: str
    pcap_ids_benign_glob: str
    pcap_ids_benign_max_pcaps: int
    pcap_ids_benign_max_flows_per_pcap: int | None
    pcap_ids_training_max_pcaps: int
    pcap_ids_malicious_max_flows_per_pcap: int | None
    pcap_ids_feature_backend: str
    pcap_ids_model_type: str
    pcap_ids_hidden_dims: list[int]
    pcap_ids_epochs: int
    pcap_ids_batch_size: int
    pcap_ids_lr: float
    cicflowmeter_cmd: str
    cicflowmeter_timeout: int
    cicflowmeter_available: bool
    pcap_ids_max_batches_per_epoch: int

    @classmethod
    def from_cfg(cls, stage3_cfg: Mapping[str, Any], stage2_cfg: Mapping[str, Any]) -> "Stage3Settings":
        search_alphas_cfg = stage3_cfg.get("pcap_search_alphas", [0.25, 0.5, 1.0, 1.5])
        if isinstance(search_alphas_cfg, str):
            search_alphas = [float(s.strip()) for s in search_alphas_cfg.split(",") if s.strip()]
        else:
            search_alphas = [float(v) for v in list(search_alphas_cfg or [1.0])]
        if not search_alphas:
            search_alphas = [1.0]

        pcap_apply_fields_cfg = stage3_cfg.get("pcap_apply_fields")
        if pcap_apply_fields_cfg is None:
            pcap_apply_fields = [
                "mean_iat_ms",
                "std_iat_ms",
                "pad_bytes",
                "dst_port_new",
                "flag_ratio",
                "flow_scale",
                "payload_scale",
            ]
        elif isinstance(pcap_apply_fields_cfg, str):
            pcap_apply_fields = [s.strip() for s in pcap_apply_fields_cfg.split(",") if s.strip()]
        else:
            pcap_apply_fields = [str(v).strip() for v in list(pcap_apply_fields_cfg) if str(v).strip()]
        pcap_apply_fields = [field for field in pcap_apply_fields if field in MOD_NAMES]

        dst_port_allowlist_cfg = stage3_cfg.get("pcap_dst_port_allowlist", [])
        if isinstance(dst_port_allowlist_cfg, str):
            dst_port_allowlist = [int(s.strip()) for s in dst_port_allowlist_cfg.split(",") if s.strip().isdigit()]
        else:
            dst_port_allowlist = []
            for item in list(dst_port_allowlist_cfg or []):
                try:
                    dst_port_allowlist.append(int(item))
                except (TypeError, ValueError):
                    continue

        return cls(
            remap_train_source=str(stage3_cfg.get("remap_train_source", "all")).lower(),
            remap_train_objective=str(stage3_cfg.get("remap_train_objective", "identity")).lower(),
            loss=str(stage3_cfg.get("loss", "huber")),
            huber_beta=float(stage3_cfg.get("huber_beta", 1.0)),
            grad_clip=float(stage3_cfg.get("grad_clip", 1.0)),
            target_clip_sigma=float(stage3_cfg.get("target_clip_sigma", 5.0)),
            weight_decay=float(stage3_cfg.get("weight_decay", 1.0e-4)),
            remap_mode=str(stage3_cfg.get("remap_mode", "auto")).lower(),
            remap_min_r2=float(stage3_cfg.get("remap_min_r2", 0.0)),
            remap_blend_alpha=float(stage3_cfg.get("remap_blend_alpha", 0.70)),
            remap_collapse_ratio_threshold=float(stage3_cfg.get("remap_collapse_ratio_threshold", 0.25)),
            epochs=int(stage3_cfg["epochs"]),
            batch_size=int(stage3_cfg["batch_size"]),
            lr=float(stage3_cfg["lr"]),
            protocol_auto_fix=bool(stage3_cfg.get("protocol_auto_fix", True)),
            feature_aliases_path=str(stage3_cfg.get("feature_aliases_path", "")),
            feature_backend=str(stage3_cfg.get("feature_backend", "auto")).lower(),
            ids_name=str(stage3_cfg.get("ids_name") or stage3_cfg.get("oracle_name") or "pcap_ids"),
            oracle_name=str(stage3_cfg.get("oracle_name") or stage2_cfg.get("oracle_name", "mlp_small")),
            pcap_eval_use_ids=bool(stage3_cfg.get("pcap_eval_use_ids", False)),
            pcap_eval_use_oracle=bool(stage3_cfg.get("pcap_eval_use_oracle", True)),
            pcap_align_min_coverage=float(stage3_cfg.get("pcap_align_min_coverage", 0.85)),
            pcap_eval_batch_size=int(stage3_cfg.get("pcap_eval_batch_size", 512)),
            pcap_cache_enable=bool(stage3_cfg.get("pcap_cache_enable", True)),
            pcap_cache_dir=str(stage3_cfg.get("pcap_cache_dir", os.path.join(".cache", "rdsynth_stage3_pcap"))),
            pcap_feature_fail_closed=bool(stage3_cfg.get("pcap_feature_fail_closed", False)),
            pcap_feature_fail_on_partial_alignment=bool(
                stage3_cfg.get("pcap_feature_fail_on_partial_alignment", False)
            ),
            pcap_search_alphas=search_alphas,
            pcap_path=str(stage3_cfg.get("pcap_path", "")),
            pcap_scan_dir=str(stage3_cfg.get("pcap_scan_dir", "")),
            pcap_scan_limit=int(stage3_cfg.get("pcap_scan_limit", 0)),
            pcap_scan_min_prob=float(stage3_cfg.get("pcap_scan_min_prob", 0.5)),
            pcap_scan_glob=str(stage3_cfg.get("pcap_scan_glob", "*.pcap")),
            pcap_scan_compare_existing=bool(stage3_cfg.get("pcap_scan_compare_existing", True)),
            pcap_scan_pmal_weight=float(stage3_cfg.get("pcap_scan_pmal_weight", 0.45)),
            pcap_scan_target_fit_weight=float(stage3_cfg.get("pcap_scan_target_fit_weight", 0.35)),
            pcap_scan_target_mod_fit_weight=float(stage3_cfg.get("pcap_scan_target_mod_fit_weight", 0.20)),
            pcap_scan_max_bytes=int(stage3_cfg.get("pcap_scan_max_bytes", 0) or 0),
            pcap_feature_max_flows_per_pcap=(
                int(stage3_cfg["pcap_feature_max_flows_per_pcap"])
                if stage3_cfg.get("pcap_feature_max_flows_per_pcap") not in (None, "")
                else None
            ),
            pcap_semantic_filter=bool(stage3_cfg.get("pcap_semantic_filter", True)),
            pcap_dataset=str(stage3_cfg.get("pcap_dataset", "")),
            pcap_attack_label=str(stage3_cfg.get("pcap_attack_label", "")),
            pcap_attack_labels=[str(v) for v in list(stage3_cfg.get("pcap_attack_labels") or [])],
            pcap_source_selection_mode=str(stage3_cfg.get("pcap_source_selection_mode", "best")).lower(),
            pcap_source_sample_n=int(stage3_cfg.get("pcap_source_sample_n", 1)),
            pcap_source_sample_seed=int(stage3_cfg.get("pcap_source_sample_seed", 0)),
            pcap_target_source=str(stage3_cfg.get("pcap_target_source", "stage2_saved_samples")).lower(),
            adv_samples_path=str(stage3_cfg.get("adv_samples_path", "")),
            copy_adv_samples=bool(stage3_cfg.get("copy_adv_samples", True)),
            pcap_apply_fields=pcap_apply_fields,
            pcap_search_bidirectional=bool(stage3_cfg.get("pcap_search_bidirectional", True)),
            pcap_search_field_subsets=bool(stage3_cfg.get("pcap_search_field_subsets", True)),
            pcap_search_probe_topk=int(stage3_cfg.get("pcap_search_probe_topk", 4)),
            pcap_search_rounds=max(1, int(stage3_cfg.get("pcap_search_rounds", 1))),
            pcap_tcp_fixup=bool(stage3_cfg.get("pcap_tcp_fixup", True)),
            pcap_dst_port_policy=str(stage3_cfg.get("pcap_dst_port_policy", "keep")).lower(),
            pcap_dst_port_allowlist=dst_port_allowlist,
            pcap_out_dir=str(stage3_cfg.get("pcap_out_dir", "")),
            pcap_apply_n=int(stage3_cfg.get("pcap_apply_n", 1)),
            pcap_multi_step_max_rounds=int(stage3_cfg.get("pcap_multi_step_max_rounds", 3)),
            pcap_eval=bool(stage3_cfg.get("pcap_eval", True)),
            pcap_compare_baselines=bool(stage3_cfg.get("pcap_compare_baselines", True)),
            pcap_baseline_jobs=int(stage3_cfg.get("pcap_baseline_jobs", 1)),
            save_intermediate_results=bool(stage3_cfg.get("save_intermediate_results", True)),
            pcap_ids_benign_path=str(stage3_cfg.get("pcap_ids_benign_path", "data/PCAPs/benign/benign.pcap")),
            pcap_ids_benign_paths=[str(v) for v in list(stage3_cfg.get("pcap_ids_benign_paths") or [])],
            pcap_mandatory_paths=[str(v) for v in list(stage3_cfg.get("pcap_mandatory_paths") or [])],
            pcap_ids_benign_dir=str(stage3_cfg.get("pcap_ids_benign_dir", "")),
            pcap_ids_benign_glob=str(stage3_cfg.get("pcap_ids_benign_glob", "*.pcap")),
            pcap_ids_benign_max_pcaps=int(stage3_cfg.get("pcap_ids_benign_max_pcaps", 1)),
            pcap_ids_benign_max_flows_per_pcap=(
                int(stage3_cfg["pcap_ids_benign_max_flows_per_pcap"])
                if stage3_cfg.get("pcap_ids_benign_max_flows_per_pcap") not in (None, "")
                else None
            ),
            pcap_ids_training_max_pcaps=int(stage3_cfg.get("pcap_ids_training_max_pcaps", 5)),
            pcap_ids_malicious_max_flows_per_pcap=(
                int(stage3_cfg["pcap_ids_malicious_max_flows_per_pcap"])
                if stage3_cfg.get("pcap_ids_malicious_max_flows_per_pcap") not in (None, "")
                else None
            ),
            pcap_ids_feature_backend=str(stage3_cfg.get("pcap_ids_feature_backend", "")),
            pcap_ids_model_type=str(stage3_cfg.get("pcap_ids_model_type", "extra_trees")),
            pcap_ids_hidden_dims=[int(v) for v in list(stage3_cfg.get("pcap_ids_hidden_dims", [128, 128]))],
            pcap_ids_epochs=int(stage3_cfg.get("pcap_ids_epochs", 5)),
            pcap_ids_batch_size=int(stage3_cfg.get("pcap_ids_batch_size", 256)),
            pcap_ids_lr=float(stage3_cfg.get("pcap_ids_lr", 1.0e-3)),
            pcap_ids_max_batches_per_epoch=int(stage3_cfg.get("pcap_ids_max_batches_per_epoch", 50)),
            cicflowmeter_cmd=str(stage3_cfg.get("cicflowmeter_cmd", "tools/CICFlowMeter/CICFlowMeter-4.0")),
            cicflowmeter_timeout=int(stage3_cfg.get("cicflowmeter_timeout", 300)),
            cicflowmeter_available=_check_java_available(),
        )


def _check_java_available() -> bool:
    import shutil

    java_exe = shutil.which("java")
    if java_exe is None:
        return False
    import subprocess

    try:
        result = subprocess.run([java_exe, "-version"], capture_output=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


def detect_stage3_environment() -> Stage3Environment:
    return Stage3Environment(
        scapy_available=importlib.util.find_spec("scapy") is not None,
        nfstream_available=importlib.util.find_spec("nfstream") is not None,
        cicflowmeter_available=_check_java_available(),
    )


def load_stage3_artifacts(
    *,
    cfg: Mapping[str, Any],
    oracle_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    device: Any,
    seed: int,
) -> Any:
    return load_stage1_artifacts(
        cfg=cfg,
        oracle_name=oracle_name,
        feature_dim=x_train.shape[1],
        n_classes=int(np.max(y_train)) + 1,
        surrogate_hidden_dims=cfg["stage1"]["sur_hidden"],
        feature_names=feature_names,
        device=device,
        require_checkpoint=False,
        oracle_restore_data=OracleRestoreData(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=seed,
        ),
    )


def resolve_pcap_eval_model(
    *,
    ids: Any = None,
    oracle: Any,
    surrogate: Any,
    prefer_ids: bool = False,
    prefer_oracle: bool = False,
) -> Stage3ModelSelection:
    if prefer_ids and ids is not None:
        return Stage3ModelSelection(pcap_eval_model=ids, pcap_eval_model_name="ids")
    if prefer_oracle and oracle is not None:
        return Stage3ModelSelection(pcap_eval_model=oracle, pcap_eval_model_name="oracle")
    if surrogate is not None:
        return Stage3ModelSelection(pcap_eval_model=surrogate, pcap_eval_model_name="surrogate")
    return Stage3ModelSelection(pcap_eval_model=None, pcap_eval_model_name="none")


def select_remap_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    remap_source: str,
) -> np.ndarray:
    if remap_source == "benign":
        return x_train[y_train == 0]
    if remap_source == "malicious":
        return x_train[y_train == 1]
    return x_train


def validate_remap_mode(remap_mode: str) -> None:
    if remap_mode not in {"auto", "direct", "learned", "random"}:
        raise ValueError(f"Unsupported stage3.remap_mode: {remap_mode}")


def aligned_feature_diff(
    diff: np.ndarray,
    align_meta: Mapping[str, Any] | None,
    feature_names: list[str],
) -> np.ndarray:
    if align_meta and align_meta.get("missing_features"):
        missing = set(align_meta.get("missing_features", []))
        align_mask = np.array([name not in missing for name in feature_names], dtype=bool)
        if align_mask.shape[0] == diff.shape[0] and np.any(align_mask):
            return diff[align_mask]
    return diff


def pcap_output_dir(settings: Stage3Settings, out_dir: Path) -> Path:
    return Path(settings.pcap_out_dir) if settings.pcap_out_dir else (out_dir / "pcap")


def blend_modifications(
    learned_mods: np.ndarray,
    direct_mods: np.ndarray,
    alpha: float,
    mod_names: list[str],
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    learned_mods = np.asarray(learned_mods, dtype=np.float32)
    direct_mods = np.asarray(direct_mods, dtype=np.float32)
    if learned_mods.shape != direct_mods.shape:
        return clip_modifications(direct_mods.copy())

    port_idx = mod_names.index("dst_port_new")
    cont_idx = [i for i in range(len(mod_names)) if i != port_idx]
    out = direct_mods.copy()
    out[:, cont_idx] = alpha * learned_mods[:, cont_idx] + (1.0 - alpha) * direct_mods[:, cont_idx]
    target_port = alpha * learned_mods[:, port_idx] + (1.0 - alpha) * direct_mods[:, port_idx]
    learned_gap = np.abs(learned_mods[:, port_idx] - target_port)
    direct_gap = np.abs(direct_mods[:, port_idx] - target_port)
    out[:, port_idx] = np.where(learned_gap <= direct_gap, learned_mods[:, port_idx], direct_mods[:, port_idx])
    return clip_modifications(out)


def effective_blend_alpha(
    learned_mods: np.ndarray,
    direct_mods: np.ndarray,
    mod_names: list[str],
    *,
    requested_alpha: float,
    collapse_ratio_threshold: float,
) -> tuple[float, dict[str, float | str]]:
    alpha = float(np.clip(requested_alpha, 0.0, 1.0))
    port_idx = mod_names.index("dst_port_new")
    cont_idx = [i for i in range(len(mod_names)) if i != port_idx]
    learned_std = float(np.mean(np.std(np.asarray(learned_mods)[:, cont_idx], axis=0))) if cont_idx else 0.0
    direct_std = float(np.mean(np.std(np.asarray(direct_mods)[:, cont_idx], axis=0))) if cont_idx else 0.0
    collapse_ratio = learned_std / (direct_std + 1.0e-6)
    info: dict[str, float | str] = {
        "pred_std_mean": learned_std,
        "direct_std_mean": direct_std,
        "collapse_ratio": collapse_ratio,
        "blend_alpha_requested": alpha,
        "blend_reason": "configured",
    }
    if collapse_ratio < collapse_ratio_threshold:
        alpha = min(alpha, 0.35)
        info["blend_reason"] = "collapse_guard"
    info["blend_alpha_effective"] = alpha
    return alpha, info


def apply_scapy_modification(
    pkts: object,
    mod_row: np.ndarray,
    *,
    seed: int,
    settings: Stage3Settings,
    protocol_auto_fix: bool,
) -> object:
    return apply_mod_using_scapy(
        pkts,
        mod_row,
        seed=seed,
        apply_fields=settings.pcap_apply_fields,
        tcp_fixup=settings.pcap_tcp_fixup,
        dst_port_policy=settings.pcap_dst_port_policy,
        dst_port_allowlist=settings.pcap_dst_port_allowlist,
        protocol_auto_fix=protocol_auto_fix,
    )


def write_modified_pcaps(
    pkts: object,
    mods: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
    count: int,
    settings: Stage3Settings,
    protocol_auto_fix: bool,
    wrpcap_fn,
) -> tuple[int, int, float, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    packet_total = 0
    written_paths: list[Path] = []
    start = time.perf_counter()
    for i in range(min(count, mods.shape[0])):
        new_pkts = apply_scapy_modification(
            pkts,
            mods[i],
            seed=seed + i,
            settings=settings,
            protocol_auto_fix=protocol_auto_fix,
        )
        packet_total += int(len(new_pkts))
        out_pcap = output_dir / f"adv_{i:04d}.pcap"
        wrpcap_fn(str(out_pcap), new_pkts)
        written += 1
        written_paths.append(out_pcap)
    elapsed = time.perf_counter() - start
    return written, packet_total, elapsed, written_paths
