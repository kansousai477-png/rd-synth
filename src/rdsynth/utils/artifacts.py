from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(src_path: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    dst = out_dir / "config.yaml"
    shutil.copyfile(src_path, dst)


def save_metrics(metrics: Dict[str, Any], out_dir: Path) -> None:
    save_json(metrics, out_dir / "metrics.json")


def _coerce_csv_value(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (int, float, bool, str)):
        return str(val)
    if hasattr(val, "tolist"):
        return json.dumps(val.tolist(), ensure_ascii=True)
    try:
        return str(float(val))
    except (TypeError, ValueError):
        return json.dumps(val, ensure_ascii=True)


def save_metrics_csv(metrics: Dict[str, Any], out_dir: Path, filename: str = "metrics.csv") -> None:
    path = out_dir / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics.keys()):
            writer.writerow([key, _coerce_csv_value(metrics[key])])


def save_records_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        fieldnames = ordered
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _coerce_csv_value(value) for key, value in row.items()})
    return path


def save_training_log_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return save_records_csv(path, rows)


def save_array_csv(path: Path, array: Any, *, header: Iterable[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(list(header))
        for row in arr:
            writer.writerow([_coerce_csv_value(value) for value in row])
    return path


def _sanitize_state_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_state_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_state_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_sha256_file(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_path = Path(f"{path}.sha256")
    sha_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return sha_path


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised via mocked import failure
        raise RuntimeError(
            "torch is required to save checkpoint artifacts. Use the project virtual environment "
            "(for example `venv\\Scripts\\python.exe` on Windows or `./venv/bin/python` on POSIX)."
        ) from exc
    return torch


def save_state(obj: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch = _require_torch()
    torch.save(_sanitize_state_value(obj), out_path)
    write_sha256_file(out_path)


_json_compatible = _sanitize_state_value  # identical logic, shared implementation


def save_json(payload: Dict[str, Any], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_compatible(payload), f, indent=2, sort_keys=True)


def _config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _load_config_mapping(config_path: str | Path) -> Mapping[str, Any]:
    resolved = Path(config_path).resolve()
    with open(resolved, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Config at {resolved} must deserialize to a mapping.")
    return payload


def _metadata_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_attack_type(data_cfg: Mapping[str, Any], project_cfg: Mapping[str, Any]) -> str:
    for key in ("attack_type", "attack", "attack_label", "attack_name"):
        value = _metadata_string(data_cfg.get(key))
        if value:
            return value
    include_labels = data_cfg.get("include_labels")
    if isinstance(include_labels, Sequence) and not isinstance(include_labels, (str, bytes)):
        labels = [_metadata_string(item) for item in include_labels if _metadata_string(item)]
        if len(labels) >= 2:
            return labels[-1]
    return _metadata_string(project_cfg.get("attack_type"))


def _coerce_target_model(cfg: Mapping[str, Any], stage_name: str) -> str:
    if stage_name == "stage1":
        stage1_cfg = cfg.get("stage1")
        if isinstance(stage1_cfg, Mapping):
            oracle_names = stage1_cfg.get("oracle_names")
            if isinstance(oracle_names, Sequence) and not isinstance(oracle_names, (str, bytes)) and oracle_names:
                return _metadata_string(oracle_names[0])
    for section_name in ("stage3", "stage2"):
        section = cfg.get(section_name)
        if isinstance(section, Mapping):
            oracle_name = _metadata_string(section.get("oracle_name"))
            if oracle_name:
                return oracle_name
    oracle_models = cfg.get("oracle_models")
    if isinstance(oracle_models, Sequence) and not isinstance(oracle_models, (str, bytes)):
        for item in oracle_models:
            if isinstance(item, Mapping):
                name = _metadata_string(item.get("name"))
                if name:
                    return name
    return ""


def _coerce_variant(cfg: Mapping[str, Any]) -> str:
    project_cfg = cfg.get("project")
    if isinstance(project_cfg, Mapping):
        for key in ("variant", "ablation_variant", "run_variant"):
            value = _metadata_string(project_cfg.get(key))
            if value:
                return value
    experiment_cfg = cfg.get("experiment")
    if isinstance(experiment_cfg, Mapping):
        for key in ("variant", "name"):
            value = _metadata_string(experiment_cfg.get(key))
            if value:
                return value
    return ""


def _coerce_run_id(project_cfg: Mapping[str, Any], config_hash: str) -> str:
    runtime_cfg = project_cfg.get("runtime")
    if isinstance(runtime_cfg, Mapping):
        for key in ("run_id", "invocation_id"):
            value = _metadata_string(runtime_cfg.get(key))
            if value:
                return value
    explicit = _metadata_string(project_cfg.get("run_id"))
    if explicit:
        return explicit
    out_dir = _metadata_string(project_cfg.get("out_dir"))
    if out_dir:
        return Path(out_dir).name
    return config_hash[:12]


def infer_rq_for_stage(stage_name: str) -> str:
    rq_map = {
        "stage1": "RQ1",
        "stage2": "RQ2/RQ3/RQ4",
        "stage3": "RQ5",
        "data_prep": "Stress",
        "pipeline": "RQ1-RQ5",
    }
    return rq_map.get(stage_name, "")


def build_artifact_metadata(
    *,
    config_path: str | Path,
    stage_name: str,
    status: str = "success",
    failure_reason: str = "",
    created_at: str | None = None,
    cfg: Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    config_mapping = cfg if cfg is not None else _load_config_mapping(resolved_config)
    project_cfg = config_mapping.get("project") if isinstance(config_mapping.get("project"), Mapping) else {}
    data_cfg = config_mapping.get("data") if isinstance(config_mapping.get("data"), Mapping) else {}
    runtime_cfg = project_cfg.get("runtime") if isinstance(project_cfg.get("runtime"), Mapping) else {}
    config_hash = _metadata_string(runtime_cfg.get("config_sha256")) or _config_hash(resolved_config)

    git_commit = ""
    for key in ("git_commit", "commit", "commit_hash"):
        git_commit = _metadata_string(runtime_cfg.get(key))
        if git_commit:
            break
    if not git_commit:
        git_commit = _metadata_string(project_cfg.get("git_commit")) or _metadata_string(project_cfg.get("commit"))
    if not git_commit:
        git_commit = _metadata_string(os.environ.get("GIT_COMMIT")) or _metadata_string(os.environ.get("CI_COMMIT_SHA"))

    metadata: Dict[str, Any] = {
        "run_id": _coerce_run_id(project_cfg, config_hash),
        "git_commit": git_commit,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolved_config),
        "config_hash": config_hash,
        "dataset": _metadata_string(data_cfg.get("dataset")),
        "attack_type": _coerce_attack_type(data_cfg, project_cfg),
        "target_model": _coerce_target_model(config_mapping, stage_name),
        "variant": _coerce_variant(config_mapping),
        "seed": project_cfg.get("seed"),
        "stage": stage_name,
        "rq": _metadata_string(project_cfg.get("rq")) or infer_rq_for_stage(stage_name),
        "status": _metadata_string(status) or "success",
        "failure_reason": _metadata_string(failure_reason),
    }
    for key, value in (extra_fields or {}).items():
        metadata[str(key)] = value
    return metadata


def sanitize_failure_label(value: Any) -> str:
    text = _metadata_string(value)
    if not text:
        return "unknown"
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("._")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def write_failure_record(
    *,
    project_cfg: Mapping[str, Any],
    config_path: str | Path,
    stage_name: str,
    error: BaseException,
    cfg: Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> Path | None:
    runtime_cfg = project_cfg.get("runtime")
    if not isinstance(runtime_cfg, Mapping):
        return None
    failed_out_dir = _metadata_string(runtime_cfg.get("failed_out_dir"))
    if not failed_out_dir:
        return None

    failed_dir = ensure_dir(Path(failed_out_dir))
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = build_artifact_metadata(
        config_path=config_path,
        stage_name=stage_name,
        status="failed",
        failure_reason=str(error),
        created_at=created_at,
        cfg=cfg,
        extra_fields=extra_fields,
    )
    payload = {
        "metadata": metadata,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stage": stage_name,
    }
    stamp = created_at.replace(":", "-")
    name = f"{sanitize_failure_label(stage_name)}_{sanitize_failure_label(metadata.get('run_id'))}_{stamp}.json"
    path = failed_dir / name
    save_json(payload, path)
    return path


def summarize_array(array: Any) -> Dict[str, Any]:
    arr = np.asarray(array)
    summary: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size == 0:
        return summary

    if np.issubdtype(arr.dtype, np.number):
        arr64 = np.asarray(arr, dtype=np.float64)
        finite = np.isfinite(arr64)
        finite_vals = arr64[finite]
        summary["finite_rate"] = float(np.mean(finite))
        summary["nan_rate"] = float(np.mean(np.isnan(arr64)))
        if finite_vals.size:
            summary["min"] = float(np.min(finite_vals))
            summary["max"] = float(np.max(finite_vals))
            summary["mean"] = float(np.mean(finite_vals))
            summary["std"] = float(np.std(finite_vals))
    return summary


def save_stage_manifest(
    *,
    stage_name: str,
    out_dir: Path,
    config_path: str | Path,
    inputs: Dict[str, Any] | None = None,
    outputs: Dict[str, Any] | None = None,
    arrays: Dict[str, Any] | None = None,
    metrics: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    filename: str = "manifest.json",
) -> Path:
    manifest = {
        "stage": stage_name,
        "config_path": str(config_path),
        "metadata": metadata or build_artifact_metadata(config_path=config_path, stage_name=stage_name),
        "inputs": inputs or {},
        "outputs": outputs or {},
        "arrays": {name: summarize_array(value) for name, value in (arrays or {}).items()},
        "metrics": metrics or {},
    }
    path = out_dir / filename
    save_json(manifest, path)
    return path
