from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from rdsynth.pipeline.data import load_data_context
from rdsynth.pipeline.runtime import load_stage_runtime
from rdsynth.pipeline.stage1_eval import evaluate_stage1_models
from rdsynth.pipeline.stage1_matrix import Stage1MatrixSummaryRow, write_stage1_agreement_matrix
from rdsynth.pipeline.stage1_outputs import attach_stage1_training_metrics, save_stage1_run_outputs
from rdsynth.pipeline.stage1_quality import run_stage1_data_quality
from rdsynth.pipeline.stage1_training import load_or_train_stage1_models
from rdsynth.stages.oracle import OracleWrapper
from rdsynth.utils.artifacts import (
    ensure_dir,
    save_config,
)
from rdsynth.utils.checkpoints import resolve_stage1_artifact_root
from rdsynth.utils.config import maybe_int


@dataclass(frozen=True)
class Stage1Settings:
    extraction_mode: str
    eval_max_rows: int | None
    eval_batch_size: int | None
    compute_matrix: bool
    matrix_max_rows: int | None
    matrix_batch_size: int | None
    compare_baseline: bool
    baseline_steps: int
    force_retrain: bool
    balance_eval: bool
    query_strategy: str
    query_pool: int
    query_mix_ratio: float
    query_real_ratio: float
    query_balance: bool
    query_label_noise: float
    extraction_rounds: int
    real_warmup_steps: int
    calibration_bins: int
    z_dim: int
    gen_hidden: Any
    sur_hidden: Any
    steps: int
    batch_size: int
    lr_s: float
    lr_g: float
    log_every: int
    query_budget: int | None
    consistency_weight: float
    consistency_noise: float
    use_forward_diff: bool
    n_G: int
    n_S: int
    fd_m: int
    fd_epsilon: float

    @classmethod
    def from_cfg(cls, stage1_cfg: Mapping[str, Any]) -> "Stage1Settings":
        return cls(
            extraction_mode=str(stage1_cfg.get("extraction_mode", "active")),
            eval_max_rows=maybe_int(stage1_cfg.get("eval_max_rows")),
            eval_batch_size=maybe_int(stage1_cfg.get("eval_batch_size")),
            compute_matrix=bool(stage1_cfg.get("compute_matrix", True)),
            matrix_max_rows=maybe_int(stage1_cfg.get("matrix_max_rows")),
            matrix_batch_size=maybe_int(stage1_cfg.get("matrix_batch_size")),
            compare_baseline=bool(stage1_cfg.get("compare_baseline", False)),
            baseline_steps=int(stage1_cfg.get("baseline_steps", stage1_cfg["steps"])),
            force_retrain=bool(stage1_cfg.get("force_retrain", False)),
            balance_eval=bool(stage1_cfg.get("balance_eval", False)),
            query_strategy=str(stage1_cfg.get("query_strategy", "random")),
            query_pool=int(stage1_cfg.get("query_pool", 1)),
            query_mix_ratio=float(stage1_cfg.get("query_mix_ratio", 0.5)),
            query_real_ratio=float(stage1_cfg.get("query_real_ratio", 0.0)),
            query_balance=bool(stage1_cfg.get("query_balance", False)),
            query_label_noise=float(stage1_cfg.get("query_label_noise", 0.0)),
            extraction_rounds=max(1, int(stage1_cfg.get("extraction_rounds", 1))),
            real_warmup_steps=int(stage1_cfg.get("real_warmup_steps", 0)),
            calibration_bins=int(stage1_cfg.get("calibration_bins", 10)),
            z_dim=int(stage1_cfg["z_dim"]),
            gen_hidden=stage1_cfg["gen_hidden"],
            sur_hidden=stage1_cfg["sur_hidden"],
            steps=int(stage1_cfg["steps"]),
            batch_size=int(stage1_cfg["batch_size"]),
            lr_s=float(stage1_cfg["lr_s"]),
            lr_g=float(stage1_cfg["lr_g"]),
            log_every=int(stage1_cfg["log_every"]),
            query_budget=maybe_int(stage1_cfg.get("query_budget")),
            consistency_weight=float(stage1_cfg.get("consistency_weight", 0.0)),
            consistency_noise=float(stage1_cfg.get("consistency_noise", 0.0)),
            use_forward_diff=bool(stage1_cfg.get("use_forward_diff", True)),
            n_G=int(stage1_cfg.get("n_G", 1)),
            n_S=int(stage1_cfg.get("n_S", 1)),
            fd_m=int(stage1_cfg.get("fd_m", 3)),
            fd_epsilon=float(stage1_cfg.get("fd_epsilon", 0.01)),
        )


@dataclass
class Stage1RunResult:
    name: str
    oracle_type: str
    oracle: OracleWrapper
    surrogate: nn.Module
    metrics: dict[str, Any]
    summary_row: Stage1MatrixSummaryRow
    baseline_row: Stage1MatrixSummaryRow


def run_stage1(config_path: str | Path) -> None:
    runtime = load_stage_runtime(config_path, "stage1")
    cfg = runtime.cfg
    stage1_cfg = runtime.stage_cfg
    settings = Stage1Settings.from_cfg(stage1_cfg)

    seed = runtime.seed
    device = runtime.device
    out_dir = runtime.out_dir

    data_ctx = load_data_context(cfg, seed)
    bundle = data_ctx.bundle

    run_stage1_data_quality(
        stage1_cfg=stage1_cfg,
        out_dir=out_dir,
        features=data_ctx.features,
        labels=data_ctx.labels,
        seed=seed,
    )

    oracle_cfgs = _select_oracle_configs(cfg, stage1_cfg)
    save_config(str(runtime.config_path), out_dir)

    results: list[Stage1RunResult] = []
    oracle_pool: dict[str, OracleWrapper] = {}
    surrogate_pool: dict[str, nn.Module] = {}

    for oracle_cfg in oracle_cfgs:
        result = _run_single_oracle(
            cfg=cfg,
            config_path=runtime.config_path,
            oracle_cfg=oracle_cfg,
            bundle=bundle,
            settings=settings,
            stage1_cfg=stage1_cfg,
            out_dir=out_dir,
            seed=seed,
            device=device,
        )
        results.append(result)
        oracle_pool[result.name] = result.oracle
        surrogate_pool[result.name] = result.surrogate

    _print_stage1_summary(results, settings.compare_baseline)

    if not settings.compute_matrix:
        print("\n[Stage1] mutual extraction matrix skipped (compute_matrix=false)")
        return

    write_stage1_agreement_matrix(
        bundle=bundle,
        oracle_cfgs=oracle_cfgs,
        oracle_pool=oracle_pool,
        surrogate_pool=surrogate_pool,
        out_dir=out_dir,
        matrix_max_rows=settings.matrix_max_rows,
        matrix_batch_size=settings.matrix_batch_size,
        seed=seed,
        device=device,
        summary_rows=[result.summary_row for result in results],
    )


def _run_single_oracle(
    cfg: Mapping[str, Any],
    config_path: Path,
    oracle_cfg: Mapping[str, Any],
    bundle: Any,
    settings: Stage1Settings,
    stage1_cfg: Mapping[str, Any],
    out_dir: Path,
    seed: int,
    device: torch.device,
) -> Stage1RunResult:
    name = str(oracle_cfg["name"])
    print(f"\n[Stage1] processing oracle={name}")
    local_model_dir = ensure_dir(out_dir / name)
    artifact_dir = ensure_dir(resolve_stage1_artifact_root(cfg, name))
    model_dir = artifact_dir
    stage1_path = artifact_dir / "stage1.pt"
    saved_config_path = artifact_dir / "config.yaml"
    saved_fingerprint_path = artifact_dir / "stage1_fingerprint.json"

    current_fingerprint = _stage1_config_fingerprint(cfg, oracle_cfg, stage1_cfg)
    saved_fingerprint = _json_digest(saved_fingerprint_path) if saved_fingerprint_path.exists() else None
    if saved_fingerprint is not None:
        config_changed = saved_fingerprint != _stable_json_digest(current_fingerprint)
    else:
        current_hash = _config_digest(config_path)
        saved_hash = _config_digest(saved_config_path) if saved_config_path.exists() else None
        config_changed = saved_hash is not None and saved_hash != current_hash

    state = load_or_train_stage1_models(
        cfg=cfg,
        config_path=config_path,
        oracle_cfg=oracle_cfg,
        bundle=bundle,
        model_dir=model_dir,
        stage1_path=stage1_path,
        force_retrain=settings.force_retrain,
        config_changed=config_changed,
        settings=settings,
        device=device,
        seed=seed,
    )

    eval_result = evaluate_stage1_models(
        config=cfg,
        bundle=bundle,
        oracle=state.oracle,
        surrogate=state.surrogate,
        oracle_type=state.oracle_type,
        n_classes=int(state.n_classes),
        val_acc=float(state.val_acc),
        settings=settings,
        seed=seed,
        device=device,
    )
    metrics = eval_result.metrics
    eval_snapshot = eval_result.eval_snapshot
    attach_stage1_training_metrics(metrics, state)
    save_stage1_run_outputs(
        model_dir=model_dir,
        mirror_dir=None if local_model_dir.resolve() == model_dir.resolve() else local_model_dir,
        stage1_path=stage1_path,
        config_path=config_path,
        stage1_cfg=stage1_cfg,
        bundle=bundle,
        oracle_name=name,
        state=state,
        metrics=metrics,
        eval_snapshot=eval_snapshot,
    )
    _write_stage1_config_fingerprint(saved_fingerprint_path, current_fingerprint)

    return Stage1RunResult(
        name=name,
        oracle_type=state.oracle_type,
        oracle=state.oracle,
        surrogate=state.surrogate,
        metrics=metrics,
        summary_row=Stage1MatrixSummaryRow(
            name=name,
            oracle_type=state.oracle_type,
            agreement=float(metrics["agreement"]),
            acc=float(metrics["surrogate_val_acc"]),
            f1=float(metrics["surrogate_val_f1"]),
        ),
        baseline_row=Stage1MatrixSummaryRow(
            name=name,
            oracle_type=state.oracle_type,
            agreement=float(metrics["baseline_agreement"]),
            acc=float(metrics["baseline_surrogate_val_acc"]),
            f1=float(metrics["baseline_surrogate_val_f1"]),
        ),
    )


def _select_oracle_configs(cfg: Mapping[str, Any], stage1_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    oracle_cfgs = cfg.get("oracle_models")
    if not isinstance(oracle_cfgs, list) or not oracle_cfgs:
        raise ValueError("oracle_models not found in config.")

    oracle_filter = os.environ.get("RDSYNTH_ORACLES")
    if oracle_filter:
        selected = {name.strip() for name in oracle_filter.split(",") if name.strip()}
    else:
        selected = {name for name in (stage1_cfg.get("oracle_names", []) or []) if name}
    if not selected:
        return [dict(cfg) for cfg in oracle_cfgs]

    filtered = [dict(oracle_cfg) for oracle_cfg in oracle_cfgs if oracle_cfg.get("name") in selected]
    if not filtered:
        raise ValueError(f"No oracle_models matched filter: {sorted(selected)}")
    return filtered


def _print_stage1_summary(results: list[Stage1RunResult], compare_baseline: bool) -> None:
    print("\n[Stage1] summary (oracle vs surrogate):")
    print("name\ttype\tagreement\tacc\tf1\toracle_acc\toracle_f1\tsur_ece\tsur_brier")
    for result in results:
        print(
            f"{result.name}\t{result.oracle_type}\t{result.metrics['agreement']:.4f}"
            f"\t{result.metrics['surrogate_val_acc']:.4f}\t{result.metrics['surrogate_val_f1']:.4f}"
            f"\t{result.metrics['oracle_eval_acc']:.4f}\t{result.metrics['oracle_eval_f1']:.4f}"
            f"\t{result.metrics['surrogate_ece']:.4f}\t{result.metrics['surrogate_brier']:.4f}"
        )

    if compare_baseline:
        print("\n[Stage1] baseline summary (random generator):")
        print("name\ttype\tagreement\tacc\tf1")
        for result in results:
            print(
                f"{result.name}\t{result.oracle_type}\t{result.metrics['baseline_agreement']:.4f}"
                f"\t{result.metrics['baseline_surrogate_val_acc']:.4f}"
                f"\t{result.metrics['baseline_surrogate_val_f1']:.4f}"
            )


def _config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage1_config_fingerprint(
    cfg: Mapping[str, Any],
    oracle_cfg: Mapping[str, Any],
    stage1_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    project_cfg = cfg.get("project", {})
    if not isinstance(project_cfg, Mapping):
        project_cfg = {}
    relevant_stage1 = dict(stage1_cfg)
    relevant_stage1.pop("force_retrain", None)
    return {
        "schema_version": 1,
        "project": {
            "seed": project_cfg.get("seed"),
            "attack_type": project_cfg.get("attack_type"),
            "eval_attack_label": project_cfg.get("eval_attack_label"),
        },
        "data": cfg.get("data", {}),
        "stage1": relevant_stage1,
        "oracle": dict(oracle_cfg),
    }


def _stable_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _stable_json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json_text(payload).encode("utf-8")).hexdigest()


def _json_digest(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _stable_json_digest(payload)


def _write_stage1_config_fingerprint(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
