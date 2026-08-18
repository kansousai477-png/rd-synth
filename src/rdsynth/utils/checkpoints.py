from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from rdsynth.models.mlp import MLP
from rdsynth.stages.oracle import (
    OracleWrapper,
    build_oracle_model,
    restore_safe_oracle_model,
    train_oracle_from_config,
)

TORCH_ORACLE_TYPES = {"mlp", "cnn", "rnn", "lstm", "gru", "transformer"}


@dataclass(frozen=True)
class OracleRestoreData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    seed: int = 0


def load_torch_state(path: Path, map_location=None, *, allow_unsafe: bool = False) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if allow_unsafe:
            return torch.load(path, map_location=map_location)
        raise RuntimeError(
            "Safe checkpoint loading is unavailable in this PyTorch build. "
            "Set project.allow_unsafe_checkpoint_load=true only for trusted artifacts."
        ) from exc
    except Exception as exc:
        if allow_unsafe:
            print(f"[Checkpoint][Warn] safe load failed for {path}; falling back to unsafe torch.load.")
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
        raise RuntimeError(
            f"Safe checkpoint load failed for {path}. "
            "This usually means the artifact contains legacy pickled Python objects. "
            "Set project.allow_unsafe_checkpoint_load=true or RDSYNTH_ALLOW_UNSAFE_CHECKPOINT_LOAD=1 "
            "only if you trust the source."
        ) from exc


def infer_mlp_out_dim(state_dict: Dict[str, torch.Tensor]) -> int | None:
    if "output.weight" in state_dict:
        return int(state_dict["output.weight"].shape[0])
    max_idx = -1
    last_weight = None
    for k, v in state_dict.items():
        if k.startswith("net.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                idx = int(parts[1])
                if idx > max_idx:
                    max_idx = idx
                    last_weight = v
    if last_weight is not None:
        return int(last_weight.shape[0])
    return None


def remap_mlp_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("feature_net.") or k.startswith("output.") for k in state_dict):
        return state_dict
    linear_ids = []
    for k in state_dict:
        if k.startswith("net.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                linear_ids.append(int(parts[1]))
    if not linear_ids:
        raise RuntimeError("Unrecognized MLP state_dict format.")
    last_idx = max(linear_ids)
    remap = {}
    for k, v in state_dict.items():
        if not k.startswith("net."):
            continue
        parts = k.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        idx = int(parts[1])
        param = ".".join(parts[2:])
        if idx < last_idx:
            remap[f"feature_net.{idx}.{param}"] = v
        elif idx == last_idx:
            remap[f"output.{param}"] = v
    return remap


def load_mlp_state(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], label: str) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    remap = remap_mlp_state_dict(state_dict)
    model.load_state_dict(remap, strict=False)
    print(f"[Checkpoint][Warn] {label} checkpoint is legacy format; applied key remap.")


def build_surrogate_from_state(
    state_dict: Dict[str, torch.Tensor],
    feature_dim: int,
    hidden_dims: Any,
    device: torch.device,
    label: str = "surrogate",
) -> Tuple[MLP, int]:
    out_dim = infer_mlp_out_dim(state_dict) or 2
    model = MLP(feature_dim, hidden_dims, out_dim).to(device)
    load_mlp_state(model, state_dict, label)
    model.eval()
    return model, out_dim


@dataclass
class Stage1Artifacts:
    surrogate: MLP
    oracle: OracleWrapper | None
    oracle_type: str | None
    checkpoint_path: Path


def stage1_artifact_root(project_out_dir: str | Path, oracle_name: str) -> Path:
    return Path(project_out_dir) / "stage1" / oracle_name


def stage1_checkpoint_path(project_out_dir: str | Path, oracle_name: str) -> Path:
    return stage1_artifact_root(project_out_dir, oracle_name) / "stage1.pt"


def resolve_stage1_artifact_root(cfg: Mapping[str, Any], oracle_name: str) -> Path:
    project_cfg = cfg.get("project")
    if not isinstance(project_cfg, Mapping):
        raise ValueError("Config section 'project' not found.")
    shared_root = str(project_cfg.get("stage1_shared_root", "") or "").strip()
    if shared_root:
        return Path(shared_root) / oracle_name
    out_dir = project_cfg.get("out_dir")
    if out_dir is None:
        raise ValueError("Config key 'project.out_dir' not found.")
    return stage1_artifact_root(out_dir, oracle_name)


def resolve_stage1_metrics_path(cfg: Mapping[str, Any], oracle_name: str) -> Path:
    return resolve_stage1_artifact_root(cfg, oracle_name) / "metrics.json"


def load_stage1_artifacts(
    cfg: Mapping[str, Any],
    oracle_name: str,
    feature_dim: int,
    n_classes: int,
    surrogate_hidden_dims: Any,
    device: torch.device,
    feature_names: list[str] | None = None,
    require_checkpoint: bool = True,
    oracle_restore_data: OracleRestoreData | None = None,
) -> Stage1Artifacts:
    project_cfg = cfg.get("project")
    if not isinstance(project_cfg, Mapping):
        raise ValueError("Config section 'project' not found.")
    checkpoint_path = resolve_stage1_artifact_root(cfg, oracle_name) / "stage1.pt"
    if not checkpoint_path.exists():
        if require_checkpoint:
            raise FileNotFoundError(
                f"Stage1 checkpoint not found at {checkpoint_path}. "
                "Run scripts/run_stage1.py or the pipeline before later stages."
            )
        surrogate = MLP(feature_dim, surrogate_hidden_dims, n_classes).to(device)
        surrogate.eval()
        return Stage1Artifacts(
            surrogate=surrogate,
            oracle=None,
            oracle_type=None,
            checkpoint_path=checkpoint_path,
        )

    env_allow_unsafe = os.environ.get("RDSYNTH_ALLOW_UNSAFE_CHECKPOINT_LOAD", "").strip() == "1"
    stage1_state = load_torch_state(
        checkpoint_path,
        map_location=device,
        allow_unsafe=bool(project_cfg.get("allow_unsafe_checkpoint_load", False)) or env_allow_unsafe,
    )
    saved_feature_dim = stage1_state.get("feature_dim")
    if saved_feature_dim is not None and int(saved_feature_dim) != int(feature_dim):
        raise ValueError(
            f"Stage1 checkpoint feature_dim mismatch: checkpoint={saved_feature_dim}, current={feature_dim}."
        )
    saved_oracle_name = stage1_state.get("oracle_name")
    if saved_oracle_name is not None and str(saved_oracle_name) != str(oracle_name):
        raise ValueError(
            f"Stage1 checkpoint oracle_name mismatch: checkpoint={saved_oracle_name}, requested={oracle_name}."
        )
    saved_feature_names = stage1_state.get("feature_names")
    if feature_names is not None and saved_feature_names is not None:
        current_names = [str(name) for name in feature_names]
        restored_names = [str(name) for name in saved_feature_names]
        if restored_names != current_names:
            raise ValueError("Stage1 checkpoint feature_names mismatch with current dataset.")
    surrogate, _ = build_surrogate_from_state(
        stage1_state["surrogate_state"],
        feature_dim,
        surrogate_hidden_dims,
        device,
        label="surrogate",
    )
    restore_n_classes = int(stage1_state.get("n_classes", n_classes))
    oracle = _build_oracle_from_stage1_state(
        cfg=cfg,
        oracle_name=oracle_name,
        feature_dim=feature_dim,
        n_classes=restore_n_classes,
        state=stage1_state,
        device=device,
        restore_data=oracle_restore_data,
    )
    return Stage1Artifacts(
        surrogate=surrogate,
        oracle=oracle,
        oracle_type=stage1_state.get("oracle_type"),
        checkpoint_path=checkpoint_path,
    )


def _build_oracle_from_stage1_state(
    cfg: Mapping[str, Any],
    oracle_name: str,
    feature_dim: int,
    n_classes: int,
    state: Mapping[str, Any],
    device: torch.device,
    restore_data: OracleRestoreData | None,
) -> OracleWrapper | None:
    oracle_type = state.get("oracle_type")
    oracle_state = state.get("oracle_state")
    oracle_state_format = str(state.get("oracle_state_format", "")).strip()
    if oracle_type is None:
        return None

    if oracle_type in TORCH_ORACLE_TYPES:
        oracle_cfg = _find_oracle_config(cfg.get("oracle_models"), oracle_name)
        if oracle_cfg is None:
            print(f"[Checkpoint][Warn] oracle config missing for {oracle_name}; oracle restore skipped.")
            return None
        if oracle_state is None:
            print(f"[Checkpoint][Warn] torch oracle state missing for {oracle_name}; oracle restore skipped.")
            return None
        model = build_oracle_model(oracle_type, feature_dim, oracle_cfg, n_classes)
        if oracle_type == "mlp":
            load_mlp_state(model, oracle_state, "oracle")
        else:
            model.load_state_dict(oracle_state)
        model.to(device)
        return OracleWrapper(model, oracle_type, device)

    if oracle_state_format == "safe_linear":
        if not isinstance(oracle_state, Mapping):
            print(f"[Checkpoint][Warn] safe linear oracle payload missing for {oracle_name}; oracle restore skipped.")
            return None
        return OracleWrapper(restore_safe_oracle_model(dict(oracle_state), str(oracle_type)), str(oracle_type), device)

    if oracle_state_format == "retrain_from_config" or oracle_state is None:
        return _retrain_oracle_from_config(
            cfg=cfg,
            oracle_name=oracle_name,
            feature_dim=feature_dim,
            n_classes=n_classes,
            device=device,
            restore_data=restore_data,
        )

    return OracleWrapper(oracle_state, oracle_type, device)


def _find_oracle_config(oracle_cfgs: Any, oracle_name: str) -> dict[str, Any] | None:
    if not isinstance(oracle_cfgs, (list, tuple)):
        return None
    for cfg in oracle_cfgs:
        if not isinstance(cfg, Mapping):
            continue
        if cfg.get("name") == oracle_name:
            return dict(cfg)
    return None


def _retrain_oracle_from_config(
    *,
    cfg: Mapping[str, Any],
    oracle_name: str,
    feature_dim: int,
    n_classes: int,
    device: torch.device,
    restore_data: OracleRestoreData | None,
) -> OracleWrapper | None:
    if restore_data is None:
        print(f"[Checkpoint][Warn] oracle restore data missing for {oracle_name}; oracle restore skipped.")
        return None
    oracle_cfg = _find_oracle_config(cfg.get("oracle_models"), oracle_name)
    if oracle_cfg is None:
        print(f"[Checkpoint][Warn] oracle config missing for {oracle_name}; oracle restore skipped.")
        return None
    oracle_bundle, _ = train_oracle_from_config(
        name=oracle_name,
        cfg=dict(oracle_cfg),
        x_train=np.asarray(restore_data.x_train),
        y_train=np.asarray(restore_data.y_train),
        x_val=np.asarray(restore_data.x_val),
        y_val=np.asarray(restore_data.y_val),
        device=device,
        seed=int(restore_data.seed),
    )
    if int(oracle_bundle.n_classes) != int(n_classes):
        print(
            f"[Checkpoint][Warn] retrained oracle n_classes mismatch for {oracle_name}: "
            f"{oracle_bundle.n_classes} vs {n_classes}."
        )
    if int(np.asarray(restore_data.x_train).shape[1]) != int(feature_dim):
        print(
            f"[Checkpoint][Warn] retrained oracle feature_dim mismatch for {oracle_name}: "
            f"{np.asarray(restore_data.x_train).shape[1]} vs {feature_dim}."
        )
    return OracleWrapper(oracle_bundle.model, oracle_bundle.model_type, device)
