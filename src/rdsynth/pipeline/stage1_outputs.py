from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rdsynth.pipeline.stage1_training import Stage1ModelState
from rdsynth.pipeline.stage_contracts import StageManifestSpec, build_stage_output_files, save_stage_manifest_spec
from rdsynth.utils.artifacts import save_metrics, save_metrics_csv, save_state


def attach_stage1_training_metrics(metrics: dict[str, Any], state: Stage1ModelState) -> None:
    if state.oracle_train_time_sec is None or state.surrogate_train_time_sec is None:
        return
    query_runtime_sec = float(state.surrogate_query_runtime_sec or state.surrogate_train_time_sec)
    query_count = int(state.surrogate_query_count or 0)
    metrics["oracle_train_time_sec"] = float(state.oracle_train_time_sec)
    metrics["surrogate_train_time_sec"] = float(state.surrogate_train_time_sec)
    metrics["surrogate_query_runtime_sec"] = query_runtime_sec
    metrics["surrogate_query_count"] = query_count
    metrics["surrogate_query_qps"] = float(query_count / query_runtime_sec) if query_runtime_sec > 0.0 else float("nan")
    metrics["stage1_total_train_time_sec"] = float(state.oracle_train_time_sec) + float(state.surrogate_train_time_sec)
    round_log = list(state.surrogate_round_log or [])
    if round_log:
        metrics["surrogate_extraction_rounds"] = int(len(round_log))
        metrics["surrogate_extraction_round_query_counts"] = [
            int(row.get("query_count_delta", 0.0)) for row in round_log
        ]
        metrics["surrogate_extraction_round_total_queries"] = [
            int(row.get("query_count_total", 0.0)) for row in round_log
        ]
        metrics["surrogate_extraction_round_runtime_sec"] = [
            float(row.get("runtime_sec_delta", float("nan"))) for row in round_log
        ]


def save_stage1_run_outputs(
    *,
    model_dir: Path,
    mirror_dir: Path | None = None,
    stage1_path: Path,
    config_path: Path,
    stage1_cfg: Mapping[str, Any],
    bundle: Any,
    oracle_name: str,
    state: Stage1ModelState,
    metrics: dict[str, Any],
    eval_snapshot: dict[str, Any],
) -> None:
    save_metrics(metrics, model_dir)
    save_metrics_csv(metrics, model_dir)

    save_eval_snapshot = bool(stage1_cfg.get("save_eval_snapshot", True))
    if save_eval_snapshot:
        np.savez_compressed(model_dir / "eval_snapshot.npz", **eval_snapshot)

    outputs = build_stage_output_files(
        primary_artifact_key="checkpoint",
        primary_artifact_name="stage1.pt",
        extra_outputs={
            "eval_snapshot": "eval_snapshot.npz" if save_eval_snapshot else None,
        },
    )
    save_stage_manifest_spec(
        StageManifestSpec(
            stage_name="stage1",
            out_dir=model_dir,
            config_path=config_path,
            inputs={
                "oracle_name": oracle_name,
                "oracle_type": state.oracle_type,
                "feature_dim": int(bundle.x_train.shape[1]),
                "train_rows": int(bundle.x_train.shape[0]),
                "val_rows": int(bundle.x_val.shape[0]),
            },
            outputs=outputs,
            arrays=eval_snapshot,
            metrics={
                "agreement": float(metrics["agreement"]),
                "surrogate_val_acc": float(metrics["surrogate_val_acc"]),
                "surrogate_val_f1": float(metrics["surrogate_val_f1"]),
                "stage1_total_train_time_sec": float(metrics.get("stage1_total_train_time_sec", float("nan"))),
                "surrogate_query_qps": float(metrics.get("surrogate_query_qps", float("nan"))),
            },
        )
    )

    if state.checkpoint_payload is not None:
        save_state(state.checkpoint_payload, stage1_path)
        _mirror_stage1_outputs(model_dir=model_dir, mirror_dir=mirror_dir)
        print(f"[Stage1] {state.status} and saved to {model_dir}")
    else:
        _mirror_stage1_outputs(model_dir=model_dir, mirror_dir=mirror_dir)
        print(f"[Stage1] loaded from {model_dir}")


def _mirror_stage1_outputs(
    *,
    model_dir: Path,
    mirror_dir: Path | None,
) -> None:
    if mirror_dir is None:
        return
    if mirror_dir.resolve() == model_dir.resolve():
        return
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "metrics.json",
        "metrics.csv",
        "config.yaml",
        "manifest.json",
        "stage1.pt",
        "stage1.pt.sha256",
        "eval_snapshot.npz",
    ):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, mirror_dir / name)
