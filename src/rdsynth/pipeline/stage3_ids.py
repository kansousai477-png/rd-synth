from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from rdsynth.stages.oracle import OracleBundle, train_oracle_from_config
from rdsynth.stages.stage3_features import (
    extract_pcap_features_cicflowmeter,
    extract_pcap_features_nfstream,
    extract_pcap_features_scapy,
)


@dataclass(frozen=True)
class Stage3IdsTrainingResult:
    ids_bundle: OracleBundle | None
    metrics: dict[str, object]


def _sample_rows(rows: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    if max_rows is None or rows.shape[0] <= max_rows:
        return np.asarray(rows, dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(rows.shape[0], size=max_rows, replace=False)
    indices.sort()
    return np.asarray(rows[indices], dtype=np.float32)


def _resolve_pcap_paths(
    *,
    single_path: str,
    multiple_paths: list[str],
    directory: str,
    pattern: str,
    max_pcaps: int,
) -> list[Path]:
    paths: list[Path] = []
    if single_path:
        path = Path(single_path)
        if path.exists() and path.is_file():
            paths.append(path.resolve())
    for item in multiple_paths:
        path = Path(str(item))
        if path.exists() and path.is_file():
            paths.append(path.resolve())
    if directory:
        root = Path(directory)
        if root.exists():
            paths.extend(sorted(path.resolve() for path in root.rglob(pattern) if path.is_file()))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        unique.append(path)
    if max_pcaps > 0:
        return unique[:max_pcaps]
    return unique


def _extract_features_for_paths(
    *,
    paths: list[Path],
    feature_names: list[str],
    fill_values: np.ndarray,
    alias_map: dict[str, object],
    feature_backend: str,
    nfstream_available: bool,
    scapy_available: bool,
    cicflowmeter_available: bool = False,
    cicflowmeter_cmd: str = "tools/CICFlowMeter/CICFlowMeter-4.0",
    cicflowmeter_timeout: int = 300,
    max_flows_per_pcap: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    used_paths: list[str] = []
    for index, path in enumerate(paths):
        if feature_backend == "scapy":
            feat, _ = extract_pcap_features_scapy(
                str(path),
                feature_names,
                fill_values,
                return_meta=True,
            )
        elif feature_backend == "cicflowmeter":
            feat, _ = extract_pcap_features_cicflowmeter(
                str(path),
                feature_names,
                fill_values,
                alias_map=alias_map,
                return_meta=True,
                max_flows=max_flows_per_pcap,
                cicflowmeter_cmd=cicflowmeter_cmd,
                timeout=cicflowmeter_timeout,
            )
        elif feature_backend == "auto":
            if cicflowmeter_available:
                feat, _ = extract_pcap_features_cicflowmeter(
                    str(path),
                    feature_names,
                    fill_values,
                    alias_map=alias_map,
                    return_meta=True,
                    max_flows=max_flows_per_pcap,
                    cicflowmeter_cmd=cicflowmeter_cmd,
                    timeout=cicflowmeter_timeout,
                )
            elif nfstream_available:
                feat, _ = extract_pcap_features_nfstream(
                    str(path),
                    feature_names,
                    fill_values,
                    alias_map=alias_map,
                    return_meta=True,
                    max_flows=max_flows_per_pcap,
                )
            elif scapy_available:
                feat, _ = extract_pcap_features_scapy(
                    str(path),
                    feature_names,
                    fill_values,
                    return_meta=True,
                )
            else:
                raise RuntimeError("No feature extraction backend available for Stage3 IDS training.")
        else:
            if nfstream_available:
                feat, _ = extract_pcap_features_nfstream(
                    str(path),
                    feature_names,
                    fill_values,
                    alias_map=alias_map,
                    return_meta=True,
                    max_flows=max_flows_per_pcap,
                )
            elif scapy_available:
                feat, _ = extract_pcap_features_scapy(
                    str(path),
                    feature_names,
                    fill_values,
                    return_meta=True,
                )
            else:
                raise RuntimeError("Neither nfstream nor scapy is available for Stage3 IDS training.")
        feat_np = np.asarray(feat, dtype=np.float32)
        if feat_np.ndim != 2 or feat_np.shape[0] == 0 or feat_np.shape[1] != len(feature_names):
            continue
        feat_np = _sample_rows(feat_np, max_flows_per_pcap, seed + index)
        if feat_np.shape[0] == 0:
            continue
        rows.append(feat_np)
        used_paths.append(str(path))
    if not rows:
        return np.zeros((0, len(feature_names)), dtype=np.float32), []
    return np.concatenate(rows, axis=0), used_paths


def build_stage3_ids_model_cfg(settings: Any) -> dict[str, Any]:
    model_type = str(getattr(settings, "pcap_ids_model_type", "extra_trees") or "extra_trees").strip().lower()
    cfg: dict[str, Any] = {
        "type": model_type,
        "class_weight": "balanced",
    }
    if model_type in {"mlp", "cnn", "rnn", "gru", "lstm", "transformer"}:
        cfg.update(
            {
                "hidden_dims": list(getattr(settings, "pcap_ids_hidden_dims", [128, 128])),
                "epochs": int(getattr(settings, "pcap_ids_epochs", 5)),
                "batch_size": int(getattr(settings, "pcap_ids_batch_size", 256)),
                "lr": float(getattr(settings, "pcap_ids_lr", 1.0e-3)),
                "sample_strategy": "balanced",
                "max_batches_per_epoch": int(getattr(settings, "pcap_ids_max_batches_per_epoch", 50)),
            }
        )
    return cfg


def train_stage3_ids(
    *,
    malicious_pcap: Path,
    malicious_pcaps: list[Path] | None = None,
    settings: Any,
    feature_names: list[str],
    raw_feature_mean: np.ndarray,
    alias_map: dict[str, object],
    preprocessor: Any,
    device: Any,
    seed: int,
) -> Stage3IdsTrainingResult:
    benign_paths = _resolve_pcap_paths(
        single_path=str(getattr(settings, "pcap_ids_benign_path", "") or ""),
        multiple_paths=list(getattr(settings, "pcap_ids_benign_paths", []) or []),
        directory=str(getattr(settings, "pcap_ids_benign_dir", "") or ""),
        pattern=str(getattr(settings, "pcap_ids_benign_glob", "*.pcap") or "*.pcap"),
        max_pcaps=int(getattr(settings, "pcap_ids_benign_max_pcaps", 1) or 1),
    )
    nfstream_available = importlib.util.find_spec("nfstream") is not None
    scapy_available = importlib.util.find_spec("scapy") is not None
    cicflowmeter_available = getattr(settings, "cicflowmeter_available", False) or False
    cicflowmeter_cmd = str(
        getattr(settings, "cicflowmeter_cmd", "tools/CICFlowMeter/CICFlowMeter-4.0")
        or "tools/CICFlowMeter/CICFlowMeter-4.0"
    )
    cicflowmeter_timeout = int(getattr(settings, "cicflowmeter_timeout", 300) or 300)
    feature_backend = str(
        getattr(settings, "pcap_ids_feature_backend", "") or getattr(settings, "feature_backend", "auto")
    ).lower()
    if feature_backend not in {"nfstream", "scapy", "cicflowmeter", "auto"}:
        feature_backend = "auto"
    if feature_backend == "auto":
        if cicflowmeter_available:
            feature_backend = "cicflowmeter"
        elif nfstream_available:
            feature_backend = "nfstream"
        elif scapy_available:
            feature_backend = "scapy"
        else:
            raise RuntimeError(
                "No PCAP feature extraction backend available: "
                "CICFlowMeter (requires Java), nfstream, and scapy are all unavailable."
            )

    resolved_malicious_pcaps = list(malicious_pcaps or [malicious_pcap])
    malicious_rows, malicious_used = _extract_features_for_paths(
        paths=resolved_malicious_pcaps,
        feature_names=feature_names,
        fill_values=raw_feature_mean,
        alias_map=alias_map,
        feature_backend=feature_backend,
        nfstream_available=nfstream_available,
        scapy_available=scapy_available,
        cicflowmeter_available=cicflowmeter_available,
        cicflowmeter_cmd=cicflowmeter_cmd,
        cicflowmeter_timeout=cicflowmeter_timeout,
        max_flows_per_pcap=getattr(settings, "pcap_ids_malicious_max_flows_per_pcap", None),
        seed=seed,
    )
    benign_rows, benign_used = _extract_features_for_paths(
        paths=benign_paths,
        feature_names=feature_names,
        fill_values=raw_feature_mean,
        alias_map=alias_map,
        feature_backend=feature_backend,
        nfstream_available=nfstream_available,
        scapy_available=scapy_available,
        cicflowmeter_available=cicflowmeter_available,
        cicflowmeter_cmd=cicflowmeter_cmd,
        cicflowmeter_timeout=cicflowmeter_timeout,
        max_flows_per_pcap=getattr(settings, "pcap_ids_benign_max_flows_per_pcap", None),
        seed=seed + 97,
    )
    metrics: dict[str, object] = {
        "pcap_ids_enabled": True,
        "pcap_ids_source_pcap": str(malicious_pcap),
        "pcap_ids_malicious_pcaps_configured": [str(path) for path in resolved_malicious_pcaps],
        "pcap_ids_feature_backend": feature_backend,
        "pcap_ids_malicious_pcaps_used": malicious_used,
        "pcap_ids_benign_pcaps_used": benign_used,
        "pcap_ids_malicious_rows": int(malicious_rows.shape[0]),
        "pcap_ids_benign_rows": int(benign_rows.shape[0]),
    }
    if malicious_rows.shape[0] == 0 or benign_rows.shape[0] == 0:
        metrics["pcap_ids_error"] = "insufficient_pcap_training_rows"
        return Stage3IdsTrainingResult(ids_bundle=None, metrics=metrics)

    x_raw = np.concatenate([benign_rows, malicious_rows], axis=0)
    y = np.concatenate(
        [
            np.zeros((benign_rows.shape[0],), dtype=np.int64),
            np.ones((malicious_rows.shape[0],), dtype=np.int64),
        ],
        axis=0,
    )
    x = np.asarray(preprocessor.transform(x_raw), dtype=np.float32)
    stratify = y if len(np.unique(y)) > 1 and x.shape[0] >= 4 else None
    test_size = 0.25 if x.shape[0] >= 8 else 0.5
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=int(seed),
            stratify=stratify,
        )
    except ValueError:
        x_train, x_val, y_train, y_val = x, x, y, y

    ids_cfg = build_stage3_ids_model_cfg(settings)
    ids_name = str(getattr(settings, "ids_name", "pcap_ids") or "pcap_ids")
    ids_bundle, val_acc = train_oracle_from_config(
        ids_name,
        ids_cfg,
        x_train,
        y_train,
        x_val,
        y_val,
        device=device,
        seed=seed,
    )
    metrics.update(
        {
            "pcap_ids_train_rows": int(x_train.shape[0]),
            "pcap_ids_val_rows": int(x_val.shape[0]),
            "pcap_ids_val_acc": float(val_acc),
            "pcap_ids_model_type": str(ids_bundle.model_type),
            "pcap_ids_name": ids_name,
        }
    )
    return Stage3IdsTrainingResult(ids_bundle=ids_bundle, metrics=metrics)
