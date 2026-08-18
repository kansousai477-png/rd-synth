from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from rdsynth.pipeline.stage1_eval import TORCH_ORACLE_TYPES
from rdsynth.pipeline.stage_contracts import VersionedArtifactSpec, build_versioned_artifact_payload
from rdsynth.stages.oracle import OracleBundle, OracleWrapper, serialize_safe_oracle_model, train_oracle_from_config
from rdsynth.stages.stage1_surrogate import train_surrogate_blackbox
from rdsynth.utils.artifacts import save_config
from rdsynth.utils.checkpoints import OracleRestoreData, load_stage1_artifacts


@dataclass(frozen=True)
class Stage1ModelState:
    oracle: OracleWrapper
    surrogate: torch.nn.Module
    oracle_type: str
    n_classes: int
    val_acc: float
    status: str
    checkpoint_payload: dict[str, Any] | None
    oracle_train_time_sec: float | None = None
    surrogate_train_time_sec: float | None = None
    surrogate_query_count: int | None = None
    surrogate_query_runtime_sec: float | None = None
    surrogate_round_log: list[dict[str, float]] | None = None


def load_or_train_stage1_models(
    *,
    cfg: Mapping[str, Any],
    config_path: Path,
    oracle_cfg: Mapping[str, Any],
    bundle: Any,
    model_dir: Path,
    stage1_path: Path,
    force_retrain: bool,
    config_changed: bool,
    settings: Any,
    device: torch.device,
    seed: int,
) -> Stage1ModelState:
    name = str(oracle_cfg["name"])
    n_classes = len(torch.unique(torch.as_tensor(bundle.y_train)))
    query_label_noise = float(getattr(settings, "query_label_noise", 0.0))
    should_train = not stage1_path.exists() or force_retrain or config_changed

    if not should_train:
        artifacts = load_stage1_artifacts(
            cfg=cfg,
            oracle_name=name,
            feature_dim=bundle.x_train.shape[1],
            n_classes=n_classes,
            surrogate_hidden_dims=settings.sur_hidden,
            feature_names=list(bundle.feature_names),
            device=device,
            require_checkpoint=True,
            oracle_restore_data=OracleRestoreData(
                x_train=bundle.x_train,
                y_train=bundle.y_train,
                x_val=bundle.x_val,
                y_val=bundle.y_val,
                seed=seed,
            ),
        )
        if artifacts.oracle is None or artifacts.oracle_type is None:
            raise RuntimeError(f"Stage1 checkpoint for '{name}' is incomplete.")
        return Stage1ModelState(
            oracle=artifacts.oracle,
            surrogate=artifacts.surrogate,
            oracle_type=artifacts.oracle_type,
            n_classes=n_classes,
            val_acc=float("nan"),
            status="loaded",
            checkpoint_payload=None,
        )

    oracle_train_start = time.perf_counter()
    oracle_bundle, val_acc = train_oracle_from_config(
        name=name,
        cfg=dict(oracle_cfg),
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_val=bundle.x_val,
        y_val=bundle.y_val,
        device=device,
        seed=seed,
    )
    oracle_train_time_sec = time.perf_counter() - oracle_train_start
    oracle = OracleWrapper(oracle_bundle.model, oracle_bundle.model_type, device)
    real_x = torch.tensor(bundle.x_train, dtype=torch.float32) if settings.query_real_ratio > 0.0 else None
    extraction_mode = str(getattr(settings, "extraction_mode", "active") or "active").strip().lower()
    use_active_extraction = extraction_mode != "baseline_only"
    surrogate_train_start = time.perf_counter()
    surrogate_bundle = train_surrogate_blackbox(
        oracle=oracle,
        feature_dim=bundle.x_train.shape[1],
        n_classes=oracle_bundle.n_classes,
        z_dim=settings.z_dim,
        gen_hidden=settings.gen_hidden,
        sur_hidden=settings.sur_hidden,
        steps=settings.steps,
        batch_size=settings.batch_size,
        lr_s=settings.lr_s,
        lr_g=settings.lr_g,
        device=device,
        log_every=settings.log_every,
        query_budget=settings.query_budget,
        consistency_weight=settings.consistency_weight,
        consistency_noise=settings.consistency_noise,
        update_generator=use_active_extraction,
        use_forward_diff=settings.use_forward_diff if use_active_extraction else False,
        n_G=settings.n_G,
        n_S=settings.n_S,
        fd_m=settings.fd_m,
        fd_epsilon=settings.fd_epsilon,
        query_strategy=settings.query_strategy,
        query_pool=settings.query_pool,
        query_mix_ratio=settings.query_mix_ratio,
        real_x=real_x,
        query_real_ratio=settings.query_real_ratio,
        query_balance=settings.query_balance,
        query_label_noise=query_label_noise,
        real_warmup_steps=settings.real_warmup_steps,
        extraction_rounds=int(getattr(settings, "extraction_rounds", 1)),
    )
    surrogate_train_time_sec = time.perf_counter() - surrogate_train_start
    save_config(str(config_path), model_dir)

    oracle_state_format, oracle_state = _serialize_oracle_checkpoint(oracle_bundle)

    checkpoint_payload = build_versioned_artifact_payload(
        VersionedArtifactSpec(
            fields={
                "oracle_name": name,
                "surrogate_state": surrogate_bundle.surrogate.state_dict(),
                "generator_state": surrogate_bundle.generator.state_dict(),
                "oracle_type": oracle_bundle.model_type,
                "oracle_state_format": oracle_state_format,
                "feature_dim": bundle.x_train.shape[1],
                "n_classes": oracle_bundle.n_classes,
                "oracle_train_time_sec": float(oracle_train_time_sec),
                "surrogate_train_time_sec": float(surrogate_train_time_sec),
                "surrogate_query_count": int(getattr(surrogate_bundle, "query_count", 0)),
                "surrogate_query_runtime_sec": float(
                    getattr(surrogate_bundle, "runtime_sec", surrogate_train_time_sec)
                ),
                "surrogate_round_log": list(getattr(surrogate_bundle, "round_log", []) or []),
                "feature_names": list(bundle.feature_names),
            },
            optional_fields={"oracle_state": oracle_state},
        )
    )
    status = "trained" if not stage1_path.exists() else "retrained"
    return Stage1ModelState(
        oracle=oracle,
        surrogate=surrogate_bundle.surrogate,
        oracle_type=oracle_bundle.model_type,
        n_classes=oracle_bundle.n_classes,
        val_acc=float(val_acc),
        oracle_train_time_sec=float(oracle_train_time_sec),
        surrogate_train_time_sec=float(surrogate_train_time_sec),
        surrogate_query_count=int(getattr(surrogate_bundle, "query_count", 0)),
        surrogate_query_runtime_sec=float(getattr(surrogate_bundle, "runtime_sec", surrogate_train_time_sec)),
        surrogate_round_log=list(getattr(surrogate_bundle, "round_log", []) or []),
        status=status,
        checkpoint_payload=checkpoint_payload,
    )


def _serialize_oracle_checkpoint(oracle_bundle: OracleBundle) -> tuple[str, Any | None]:
    if oracle_bundle.model_type in TORCH_ORACLE_TYPES:
        return "torch_state_dict", oracle_bundle.model.state_dict()
    safe_payload = serialize_safe_oracle_model(oracle_bundle.model, oracle_bundle.model_type)
    if safe_payload is not None:
        return "safe_linear", safe_payload
    return "retrain_from_config", None
