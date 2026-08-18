from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rdsynth.stages.stage2_diffusion import (
    train_conditional_gan,
    train_editor,
    train_latent_diffusion,
)


def _selection_eval_kwargs(stage2_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_eval_every": int(stage2_cfg.get("selection_eval_every", 0)),
        "selection_eval_samples": int(stage2_cfg.get("selection_eval_samples", 256)),
        "selection_batch_size": int(stage2_cfg.get("selection_batch_size", 256)),
        "selection_mal_anchor_alpha": float(stage2_cfg.get("selection_mal_anchor_alpha", 0.1)),
    }


def _latent_schedule_kwargs(stage2_cfg: dict[str, Any]) -> dict[str, float]:
    loss_schedule_cfg = stage2_cfg.get("loss_schedule", {}) or {}
    if not bool(loss_schedule_cfg.get("enable", False)):
        return {
            "fidelity_scale_start": 1.0,
            "fidelity_scale_end": 1.0,
            "attack_scale_start": 1.0,
            "attack_scale_end": 1.0,
        }
    return {
        "fidelity_scale_start": float(loss_schedule_cfg.get("fidelity_scale_start", 1.0)),
        "fidelity_scale_end": float(loss_schedule_cfg.get("fidelity_scale_end", 1.0)),
        "attack_scale_start": float(loss_schedule_cfg.get("attack_scale_start", 1.0)),
        "attack_scale_end": float(loss_schedule_cfg.get("attack_scale_end", 1.0)),
    }


def _latent_diffusion_kwargs(
    *,
    stage2_cfg: dict[str, Any],
    settings: Any,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    feature_names: list[str],
    surrogate: torch.nn.Module,
    device: torch.device,
    guidance_mode: str,
) -> dict[str, Any]:
    return {
        "x_ben": x_ben,
        "x_mal": x_mal,
        "feature_names": feature_names,
        "surrogate": surrogate,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "lr": settings.lr,
        "timesteps": stage2_cfg["timesteps"],
        "beta_start": stage2_cfg.get("beta_start", 1.0e-4),
        "beta_end": stage2_cfg.get("beta_end", 2.0e-2),
        "lambda_stp": stage2_cfg["lambda_stp"],
        "lambda_corr": stage2_cfg.get("lambda_corr", 0.1),
        "lambda_mmt": stage2_cfg["lambda_mmt"],
        "lambda_mmd": stage2_cfg.get("lambda_mmd", 0.05),
        "lambda_swd": stage2_cfg.get("lambda_swd", 0.05),
        "lambda_latent": stage2_cfg.get("lambda_latent", 0.1),
        "lambda_sem": stage2_cfg.get("lambda_sem", 0.2),
        "lambda_ben": stage2_cfg.get("lambda_ben", 0.02),
        "lambda_preserve": stage2_cfg.get("lambda_preserve", 0.0),
        "ben_temp": stage2_cfg.get("ben_temp", 10.0),
        "ben_loss_clip": stage2_cfg.get("ben_loss_clip", 10.0),
        "lambda_var": stage2_cfg.get("lambda_var", 0.05),
        "lambda_range": stage2_cfg.get("lambda_range", 0.1),
        "lambda_protocol": stage2_cfg.get("lambda_protocol", 0.0),
        "lambda_temporal": stage2_cfg.get("lambda_temporal", 0.0),
        "device": device,
        "latent_dim": stage2_cfg.get("latent_dim", 64),
        "ae_hidden": tuple(stage2_cfg.get("ae_hidden", [256, 128])),
        "ae_epochs": stage2_cfg.get("ae_epochs", 40),
        "ae_lr": stage2_cfg.get("ae_lr", 1.0e-3),
        "schedule_type": stage2_cfg.get("schedule_type", "cosine"),
        "cond_dropout": stage2_cfg.get("cond_dropout", 0.1),
        "denoiser_hidden": stage2_cfg.get("denoiser_hidden", 256),
        "denoiser_dropout": stage2_cfg.get("denoiser_dropout", 0.05),
        "predict_x0": stage2_cfg.get("predict_x0", True),
        "x0_head_tanh": stage2_cfg.get("x0_head_tanh", True),
        "cond_norm": stage2_cfg.get("cond_norm", True),
        "emb_norm": stage2_cfg.get("emb_norm", True),
        "eps_pred_clip": stage2_cfg.get("eps_pred_clip", 3.0),
        "swd_proj": stage2_cfg.get("swd_proj", 64),
        "mmd_max": stage2_cfg.get("mmd_max", 256),
        "var_std_floor": stage2_cfg.get("var_std_floor", 1.0e-2),
        "latent_std_floor": stage2_cfg.get("latent_std_floor", 1.0e-2),
        "latent_warmup_epochs": stage2_cfg.get("latent_warmup_epochs", 0),
        "cond_dropout_start": stage2_cfg.get("cond_dropout_start", -1.0),
        "cond_dropout_end": stage2_cfg.get("cond_dropout_end", -1.0),
        "cond_dropout_warmup_epochs": stage2_cfg.get("cond_dropout_warmup_epochs", 0),
        "grad_clip": stage2_cfg.get("grad_clip", 1.0),
        "guidance_mode": guidance_mode,
        "structure_every": stage2_cfg.get("structure_every", 1),
        **_selection_eval_kwargs(stage2_cfg),
        **_latent_schedule_kwargs(stage2_cfg),
    }


def _gan_kwargs(
    *,
    stage2_cfg: dict[str, Any],
    settings: Any,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    feature_names: list[str],
    surrogate: torch.nn.Module,
    device: torch.device,
    guidance_mode: str,
    gan_type: str,
) -> dict[str, Any]:
    return {
        "x_ben": x_ben,
        "x_mal": x_mal,
        "feature_names": feature_names,
        "surrogate": surrogate,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "lr": settings.lr,
        "device": device,
        "gan_type": gan_type,
        "noise_dim": int(stage2_cfg.get("gan_noise_dim", 64)),
        "guidance_mode": guidance_mode,
        "lambda_stp": stage2_cfg["lambda_stp"],
        "lambda_corr": stage2_cfg.get("lambda_corr", 0.1),
        "lambda_mmt": stage2_cfg["lambda_mmt"],
        "lambda_sem": stage2_cfg.get("lambda_sem", 0.2),
        "lambda_ben": stage2_cfg.get("lambda_ben", 0.02),
        "lambda_protocol": stage2_cfg.get("lambda_protocol", 0.0),
        "lambda_temporal": stage2_cfg.get("lambda_temporal", 0.0),
        "ben_temp": stage2_cfg.get("ben_temp", 10.0),
        "ben_loss_clip": stage2_cfg.get("ben_loss_clip", 10.0),
        "critic_steps": int(stage2_cfg.get("gan_critic_steps", 5)),
        "weight_clip": float(stage2_cfg.get("gan_weight_clip", 0.01)),
        **_selection_eval_kwargs(stage2_cfg),
    }


def _editor_kwargs(
    *,
    stage2_cfg: dict[str, Any],
    settings: Any,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    feature_names: list[str],
    surrogate: torch.nn.Module,
    device: torch.device,
    guidance_mode: str,
) -> dict[str, Any]:
    return {
        "x_ben": x_ben,
        "x_mal": x_mal,
        "feature_names": feature_names,
        "surrogate": surrogate,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "lr": settings.lr,
        "device": device,
        "latent_dim": stage2_cfg.get("latent_dim", 64),
        "ae_hidden": tuple(stage2_cfg.get("ae_hidden", [256, 128])),
        "editor_hidden": tuple(stage2_cfg.get("editor_hidden", [256, 128])),
        "ae_epochs": stage2_cfg.get("ae_epochs", 40),
        "ae_lr": stage2_cfg.get("ae_lr", 1.0e-3),
        "lambda_recon": stage2_cfg.get("lambda_recon", 0.5),
        "lambda_delta": stage2_cfg.get("lambda_delta", 0.05),
        "lambda_stp": stage2_cfg["lambda_stp"],
        "lambda_corr": stage2_cfg.get("lambda_corr", 0.1),
        "lambda_mmt": stage2_cfg["lambda_mmt"],
        "lambda_mmd": stage2_cfg.get("lambda_mmd", 0.05),
        "lambda_swd": stage2_cfg.get("lambda_swd", 0.05),
        "lambda_sem": stage2_cfg.get("lambda_sem", 0.2),
        "lambda_ben": stage2_cfg.get("lambda_ben", 0.02),
        "lambda_preserve": stage2_cfg.get("lambda_preserve", 0.0),
        "ben_temp": stage2_cfg.get("ben_temp", 10.0),
        "ben_loss_clip": stage2_cfg.get("ben_loss_clip", 10.0),
        "lambda_var": stage2_cfg.get("lambda_var", 0.05),
        "lambda_range": stage2_cfg.get("lambda_range", 0.1),
        "lambda_protocol": stage2_cfg.get("lambda_protocol", 0.0),
        "lambda_temporal": stage2_cfg.get("lambda_temporal", 0.0),
        "residual_scale": stage2_cfg.get("residual_scale", 0.5),
        "swd_proj": stage2_cfg.get("swd_proj", 64),
        "var_std_floor": stage2_cfg.get("var_std_floor", 1.0e-2),
        "guidance_mode": guidance_mode,
        "conditioning_enabled": bool(stage2_cfg.get("conditioning_enabled", True)),
        **_selection_eval_kwargs(stage2_cfg),
    }


def validate_stage2_training_options(
    *,
    stage2_mode: str,
    generator_backbone: str,
    guidance_mode: str,
) -> None:
    if generator_backbone not in {"ddpm", "gan", "cgan", "wgan"}:
        raise ValueError(f"Unsupported stage2.generator_backbone: {generator_backbone}")
    if guidance_mode not in {"embedding", "raw_only", "logits", "hard_label"}:
        raise ValueError(f"Unsupported stage2.surrogate_guidance_mode: {guidance_mode}")
    if stage2_mode != "latent_diffusion" and generator_backbone != "ddpm":
        raise ValueError("generator_backbone=gan/cgan/wgan currently requires stage2.mode=latent_diffusion")


def train_stage2_generator(
    *,
    cfg: dict[str, Any],
    settings: Any,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    feature_names: list[str],
    surrogate: torch.nn.Module,
    device: torch.device,
) -> Any:
    stage2_cfg = cfg["stage2"]
    stage2_mode = settings.mode
    generator_backbone = settings.generator_backbone
    guidance_mode = settings.guidance_mode
    validate_stage2_training_options(
        stage2_mode=stage2_mode,
        generator_backbone=generator_backbone,
        guidance_mode=guidance_mode,
    )

    if stage2_mode == "latent_diffusion" and generator_backbone == "ddpm":
        return train_latent_diffusion(
            **_latent_diffusion_kwargs(
                stage2_cfg=stage2_cfg,
                settings=settings,
                x_ben=x_ben,
                x_mal=x_mal,
                feature_names=feature_names,
                surrogate=surrogate,
                device=device,
                guidance_mode=guidance_mode,
            )
        )

    if stage2_mode == "latent_diffusion" and generator_backbone in {"gan", "cgan", "wgan"}:
        return train_conditional_gan(
            **_gan_kwargs(
                stage2_cfg=stage2_cfg,
                settings=settings,
                x_ben=x_ben,
                x_mal=x_mal,
                feature_names=feature_names,
                surrogate=surrogate,
                device=device,
                guidance_mode=guidance_mode,
                gan_type=generator_backbone,
            )
        )

    return train_editor(
        **_editor_kwargs(
            stage2_cfg=stage2_cfg,
            settings=settings,
            x_ben=x_ben,
            x_mal=x_mal,
            feature_names=feature_names,
            surrogate=surrogate,
            device=device,
            guidance_mode=guidance_mode,
        )
    )
