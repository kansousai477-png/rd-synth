from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from rdsynth.data.csv_datasets import load_csv_dataset, resolve_dataset_profile
from rdsynth.data.nb15 import DatasetBundle, prepare_splits
from rdsynth.stages.stage3_features import (
    extract_pcap_features_cicflowmeter,
    extract_pcap_features_nfstream,
    extract_pcap_features_scapy,
)
from rdsynth.utils.config import maybe_int, optional_section, require_section
from rdsynth.utils.feature_align import build_statistical_feature_aliases, load_feature_aliases


def _check_java_available() -> bool:
    import shutil
    import subprocess

    java_exe = shutil.which("java")
    if java_exe is None:
        return False
    try:
        result = subprocess.run([java_exe, "-version"], capture_output=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


@dataclass
class DataContext:
    features: pd.DataFrame
    labels: np.ndarray
    raw_labels: np.ndarray
    bundle: DatasetBundle


def _data_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return require_section(cfg, "data")


def _require_fraction(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{name}' must be a number.") from exc
    if parsed < 0.0 or parsed >= 1.0:
        raise ValueError(f"Config key '{name}' must be in [0.0, 1.0).")
    return parsed


_DATA_CONTEXT_CACHE_VERSION = 2


def _data_cache_dir(cfg: Dict[str, Any]) -> Path:
    data_cfg = _data_cfg(cfg)
    cache_dir = data_cfg.get("cache_dir")
    if cache_dir:
        return Path(str(cache_dir)).resolve()
    project_runtime = (cfg.get("project") or {}).get("runtime") or {}
    cwd = project_runtime.get("cwd")
    base = Path(str(cwd)).resolve() if cwd else Path.cwd().resolve()
    return base / ".cache" / "rdsynth_data_context"


def _data_cache_key(cfg: Dict[str, Any], seed: int) -> str:
    data_cfg = _data_cfg(cfg)
    payload = {
        "version": _DATA_CONTEXT_CACHE_VERSION,
        "seed": int(seed),
        "data": data_cfg,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _data_cache_path(cfg: Dict[str, Any], seed: int) -> Path:
    return _data_cache_dir(cfg) / f"{_data_cache_key(cfg, seed)}.pkl"


def data_cache_path(cfg: Dict[str, Any], seed: int) -> Path:
    return _data_cache_path(cfg, seed)


def _data_artifact_dir(cfg: Dict[str, Any], seed: int) -> Path:
    return _data_cache_dir(cfg) / f"{_data_cache_key(cfg, seed)}_artifacts"


def data_artifact_dir(cfg: Dict[str, Any], seed: int) -> Path:
    return _data_artifact_dir(cfg, seed)


def _persist_data_artifacts(cfg: Dict[str, Any], seed: int, context: DataContext) -> None:
    artifact_dir = _data_artifact_dir(cfg, seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        artifact_dir / "raw_dataset.npz",
        features=context.features.to_numpy(dtype=np.float64, copy=False),
        labels=np.asarray(context.labels),
    )
    split_payload: dict[str, Any] = {}
    for name in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test"):
        value = getattr(context.bundle, name, None)
        if value is not None:
            split_payload[name] = np.asarray(value)
    np.savez_compressed(artifact_dir / "split_arrays.npz", **split_payload)
    preprocess_state = {
        "feature_names": list(getattr(context.bundle, "feature_names", list(context.features.columns))),
        "raw_label_count": int(np.asarray(context.raw_labels).shape[0]),
        "scaler": getattr(context.bundle, "scaler", None),
        "power_transformer": getattr(context.bundle, "power_transformer", None),
        "scaler_type": getattr(context.bundle, "scaler_type", ""),
        "impute_strategy": getattr(context.bundle, "impute_strategy", ""),
        "impute_values": getattr(context.bundle, "impute_values", None),
        "winsorize_lower": getattr(context.bundle, "winsorize_lower", None),
        "winsorize_upper": getattr(context.bundle, "winsorize_upper", None),
        "log1p_mask": getattr(context.bundle, "log1p_mask", None),
        "log1p_shift": getattr(context.bundle, "log1p_shift", None),
        "feature_transform_plan": getattr(context.bundle, "feature_transform_plan", None),
    }
    with (artifact_dir / "preprocess_state.pkl").open("wb") as handle:
        pickle.dump(preprocess_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "cache_version": _DATA_CONTEXT_CACHE_VERSION,
        "feature_names": list(context.features.columns),
        "rows": int(context.features.shape[0]),
        "feature_count": int(context.features.shape[1]),
        "label_count": int(np.asarray(context.labels).shape[0]),
        "raw_test_label_count": int(
            np.asarray(getattr(context.bundle, "raw_y_test", np.zeros((0,), dtype=object))).shape[0]
        ),
        "split_rows": {
            "train": int(np.asarray(getattr(context.bundle, "x_train", np.zeros((0, 0)))).shape[0]),
            "val": int(np.asarray(getattr(context.bundle, "x_val", np.zeros((0, 0)))).shape[0]),
            "test": int(np.asarray(getattr(context.bundle, "x_test", np.zeros((0, 0)))).shape[0]),
        },
        "artifacts": {
            "raw_dataset": "raw_dataset.npz",
            "split_arrays": "split_arrays.npz",
            "preprocess_state": "preprocess_state.pkl",
        },
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _optional_positive_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("PCAP augmentation limits must be positive when provided.")
    return parsed


def _resolve_pcap_malicious_paths(data_cfg: Dict[str, Any]) -> list[Path]:
    explicit_paths = data_cfg.get("pcap_malicious_paths") or []
    paths: list[Path] = []
    for item in explicit_paths:
        path = Path(str(item))
        if path.exists() and path.is_file():
            paths.append(path.resolve())
    single_path = str(data_cfg.get("pcap_malicious_path", "") or "").strip()
    if single_path:
        path = Path(single_path)
        if path.exists() and path.is_file():
            paths.append(path.resolve())
    directory = str(data_cfg.get("pcap_malicious_dir", "") or "").strip()
    if directory:
        glob_pattern = str(data_cfg.get("pcap_malicious_glob", "*.pcap") or "*.pcap")
        root = Path(directory)
        if root.exists():
            paths.extend(sorted(path.resolve() for path in root.rglob(glob_pattern) if path.is_file()))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        unique.append(path)
    max_pcaps = _optional_positive_int(data_cfg.get("pcap_malicious_max_pcaps"))
    return unique[:max_pcaps] if max_pcaps is not None else unique


def _extract_pcap_malicious_features(
    *,
    feature_names: list[str],
    fill_values: np.ndarray,
    data_cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    pcap_paths = _resolve_pcap_malicious_paths(data_cfg)
    if not pcap_paths:
        return pd.DataFrame(columns=feature_names), np.zeros((0,), dtype=np.int64)

    dataset_name = str(data_cfg.get("dataset", ""))
    alias_map = build_statistical_feature_aliases(
        feature_names,
        dataset_name=dataset_name,
        base_alias_map=load_feature_aliases(str(data_cfg.get("pcap_feature_aliases_path", "") or "")),
    )
    backend = str(data_cfg.get("pcap_malicious_backend", "auto") or "auto").strip().lower()
    nfstream_available = importlib.util.find_spec("nfstream") is not None
    scapy_available = importlib.util.find_spec("scapy") is not None
    cicflowmeter_cmd = str(
        data_cfg.get("cicflowmeter_cmd", "tools/CICFlowMeter/CICFlowMeter-4.0") or "tools/CICFlowMeter/CICFlowMeter-4.0"
    )
    cicflowmeter_timeout = int(data_cfg.get("cicflowmeter_timeout", 300) or 300)
    cicflowmeter_available = _check_java_available()
    strict_ingest = bool(data_cfg.get("strict_ingest", False))
    max_flows_per_pcap = _optional_positive_int(data_cfg.get("pcap_malicious_max_flows_per_pcap"))
    # Resolve "auto" backend
    if backend == "auto":
        if cicflowmeter_available:
            backend = "cicflowmeter"
        elif nfstream_available:
            backend = "nfstream"
        elif scapy_available:
            backend = "scapy"
        else:
            backend = "none"

    rows: list[np.ndarray] = []
    for pcap_path in pcap_paths:
        try:
            if backend == "scapy":
                feat, _ = extract_pcap_features_scapy(
                    str(pcap_path),
                    feature_names,
                    fill_values,
                    return_meta=True,
                )
            elif backend == "cicflowmeter":
                feat, _ = extract_pcap_features_cicflowmeter(
                    str(pcap_path),
                    feature_names,
                    fill_values,
                    alias_map=alias_map,
                    return_meta=True,
                    cicflowmeter_cmd=cicflowmeter_cmd,
                    timeout=cicflowmeter_timeout,
                )
            elif backend == "nfstream":
                feat, _ = extract_pcap_features_nfstream(
                    str(pcap_path),
                    feature_names,
                    fill_values,
                    alias_map=alias_map,
                    return_meta=True,
                )
            elif backend == "none":
                raise RuntimeError(
                    "No PCAP feature extraction backend available (Java/CICFlowMeter, nfstream, or scapy required)."
                )
            else:
                if nfstream_available:
                    feat, _ = extract_pcap_features_nfstream(
                        str(pcap_path),
                        feature_names,
                        fill_values,
                        alias_map=alias_map,
                        return_meta=True,
                    )
                elif scapy_available:
                    feat, _ = extract_pcap_features_scapy(
                        str(pcap_path),
                        feature_names,
                        fill_values,
                        return_meta=True,
                    )
                else:
                    raise RuntimeError("Neither nfstream nor scapy is available for PCAP malicious augmentation.")
        except Exception as exc:
            if strict_ingest:
                raise
            print(f"[Data][Warn] PCAP feature extraction failed for {pcap_path}: {exc}")
            continue
        feat = np.asarray(feat, dtype=np.float64)
        if feat.ndim != 2 or feat.shape[1] != len(feature_names) or feat.shape[0] == 0:
            if strict_ingest:
                raise ValueError(f"PCAP malicious augmentation produced invalid shape for {pcap_path}: {feat.shape}")
            continue
        if max_flows_per_pcap is not None and feat.shape[0] > max_flows_per_pcap:
            feat = feat[:max_flows_per_pcap]
        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=feature_names), np.zeros((0,), dtype=np.int64)

    merged = np.concatenate(rows, axis=0)
    frame = pd.DataFrame(merged, columns=feature_names)
    labels = np.ones((frame.shape[0],), dtype=np.int64)
    return frame, labels


def _augment_with_malicious_pcaps(
    features: pd.DataFrame,
    labels: np.ndarray,
    data_cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    if not any(data_cfg.get(key) for key in ("pcap_malicious_path", "pcap_malicious_paths", "pcap_malicious_dir")):
        return features, labels
    fill_values = np.nanmean(features.to_numpy(dtype=np.float64, copy=False), axis=0)
    fill_values = np.nan_to_num(fill_values, nan=0.0, posinf=0.0, neginf=0.0)
    pcap_features, pcap_labels = _extract_pcap_malicious_features(
        feature_names=list(features.columns),
        fill_values=fill_values,
        data_cfg=data_cfg,
    )
    if pcap_features.empty:
        return features, labels
    merged_features = pd.concat([features, pcap_features], axis=0, ignore_index=True)
    merged_labels = np.concatenate([np.asarray(labels, dtype=np.int64), pcap_labels], axis=0)
    return merged_features, merged_labels


def load_data_context(cfg: Dict[str, Any], seed: int) -> DataContext:
    cache_path = _data_cache_path(cfg, seed)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, DataContext):
                artifact_dir = _data_artifact_dir(cfg, seed)
                if not (artifact_dir / "metadata.json").exists():
                    _persist_data_artifacts(cfg, seed, cached)
                return cached
        except Exception as exc:
            print(f"[Data][Warn] pickle cache corrupted or incompatible, rebuilding: {exc}")
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass

    data_cfg = _data_cfg(cfg)
    preprocess = optional_section(data_cfg, "preprocess")
    test_frac = _require_fraction(data_cfg.get("test_frac"), "data.test_frac")
    val_frac = _require_fraction(data_cfg.get("val_frac"), "data.val_frac")
    if test_frac + val_frac >= 1.0:
        raise ValueError("Config values 'data.test_frac' + 'data.val_frac' must be less than 1.0.")
    profile = resolve_dataset_profile(data_cfg)
    loaded_csv = load_csv_dataset(
        csv_path=profile.csv_path,
        csv_dir=profile.csv_dir,
        csv_glob=profile.csv_glob,
        label_col=profile.label_col,
        label_source=profile.label_source,
        task=str(data_cfg.get("task", "binary")),
        benign_labels=profile.benign_labels,
        drop_cols=profile.drop_cols,
        merge_strategy=str(data_cfg.get("merge_strategy", "intersection")),
        drop_zero_variance=bool(data_cfg.get("drop_zero_variance", False)),
        drop_near_constant=bool(data_cfg.get("drop_near_constant", False)),
        near_constant_var=float(data_cfg.get("near_constant_var", 1.0e-12)),
        max_rows=maybe_int(data_cfg.get("max_rows"), "data.max_rows", positive_only=True),
        include_labels=data_cfg.get("include_labels"),
        max_rows_per_label=maybe_int(data_cfg.get("max_rows_per_label"), "data.max_rows_per_label", positive_only=True),
        csv_chunk_size=int(data_cfg.get("csv_chunk_size", 200000)),
        seed=seed,
        strict_ingest=bool(data_cfg.get("strict_ingest", False)),
        encoding_errors=str(data_cfg.get("encoding_errors", "replace")),
        return_raw_labels=True,
    )
    if not isinstance(loaded_csv, tuple) or len(loaded_csv) not in {2, 3}:
        raise ValueError("load_csv_dataset must return (features, labels) or (features, labels, raw_labels).")
    if len(loaded_csv) == 3:
        features, labels, raw_labels = loaded_csv
    else:
        features, labels = loaded_csv
        raw_labels = np.asarray(labels, dtype=object)
    pre_aug_rows = int(features.shape[0])
    features, labels = _augment_with_malicious_pcaps(features, labels, data_cfg)
    if int(features.shape[0]) > pre_aug_rows:
        aug_rows = int(features.shape[0]) - pre_aug_rows
        raw_labels = np.concatenate(
            [
                np.asarray(raw_labels, dtype=object),
                np.asarray(["pcap_malicious_aug"] * aug_rows, dtype=object),
            ],
            axis=0,
        )
    bundle = prepare_splits(
        features,
        labels,
        raw_labels,
        test_frac=test_frac,
        val_frac=val_frac,
        seed=seed,
        scaler_type=preprocess.get("scaler", "standard"),
        power_transform=preprocess.get("power_transform", False),
        impute_strategy=preprocess.get("impute", "zero"),
        winsorize_quantile=preprocess.get("winsorize_quantile"),
        log1p_cols=data_cfg.get("log1p_cols"),
        type_aware_transform=preprocess.get("type_aware", True),
        discrete_scaling=preprocess.get("discrete_scaling", "minmax"),
        schema_overrides=preprocess.get("schema_overrides"),
    )
    context = DataContext(
        features=features, labels=labels, raw_labels=np.asarray(raw_labels, dtype=object), bundle=bundle
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(context, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(cache_path)
    _persist_data_artifacts(cfg, seed, context)
    return context
