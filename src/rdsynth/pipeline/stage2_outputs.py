from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rdsynth.pipeline.stage_contracts import (
    StageManifestSpec,
    VersionedArtifactSpec,
    build_stage_output_files,
    build_versioned_artifact_payload,
    collect_manifest_arrays,
    save_stage_manifest_spec,
)
from rdsynth.utils.artifacts import save_state


def save_stage2_state(
    *,
    stage2_mode: str,
    generator_backbone: str,
    diffusion_bundle: Any,
    feature_names: list[str],
    out_path: Path,
) -> None:
    if stage2_mode == "latent_diffusion" and generator_backbone == "ddpm":
        payload = build_versioned_artifact_payload(
            VersionedArtifactSpec(
                fields={
                    "denoiser_state": diffusion_bundle.denoiser.state_dict(),
                    "encoder_state": diffusion_bundle.encoder.state_dict(),
                    "decoder_state": diffusion_bundle.decoder.state_dict(),
                    "groups": diffusion_bundle.groups,
                    "feature_names": feature_names,
                    "ben_stats": diffusion_bundle.ben_stats,
                    "latent_mean": diffusion_bundle.latent_mean.detach().cpu(),
                    "latent_std": diffusion_bundle.latent_std.detach().cpu(),
                    "predict_x0": diffusion_bundle.predict_x0,
                    "x0_head_tanh": diffusion_bundle.x0_head_tanh,
                    "best_epoch": diffusion_bundle.best_epoch,
                    "best_score": diffusion_bundle.best_score,
                    "train_log": diffusion_bundle.train_log,
                }
            )
        )
    elif stage2_mode == "latent_diffusion" and generator_backbone in {"gan", "cgan", "wgan"}:
        payload = build_versioned_artifact_payload(
            VersionedArtifactSpec(
                fields={
                    "generator_state": diffusion_bundle.generator.state_dict(),
                    "critic_state": diffusion_bundle.critic.state_dict(),
                    "groups": diffusion_bundle.groups,
                    "feature_names": feature_names,
                    "ben_stats": diffusion_bundle.ben_stats,
                    "noise_dim": diffusion_bundle.noise_dim,
                    "guidance_mode": diffusion_bundle.guidance_mode,
                    "gan_type": diffusion_bundle.gan_type,
                    "best_epoch": diffusion_bundle.best_epoch,
                    "best_score": diffusion_bundle.best_score,
                    "train_log": diffusion_bundle.train_log,
                }
            )
        )
    else:
        payload = build_versioned_artifact_payload(
            VersionedArtifactSpec(
                fields={
                    "encoder_state": diffusion_bundle.encoder.state_dict(),
                    "decoder_state": diffusion_bundle.decoder.state_dict(),
                    "editor_state": diffusion_bundle.editor.state_dict(),
                    "groups": diffusion_bundle.groups,
                    "feature_names": feature_names,
                    "ben_stats": diffusion_bundle.ben_stats,
                    "latent_dim": diffusion_bundle.latent_dim,
                    "best_epoch": diffusion_bundle.best_epoch,
                    "best_score": diffusion_bundle.best_score,
                    "train_log": diffusion_bundle.train_log,
                }
            )
        )
    save_state(payload, out_path)


def save_stage2_manifest(
    *,
    out_dir: Path,
    config_path: str,
    oracle_name: str,
    x_train: np.ndarray,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    stage2_mode: str,
    train_log: Any,
    metrics_payload: dict[str, Any],
    settings: Any,
    x_adv_pre: np.ndarray | None,
    x_adv_norm: np.ndarray | None,
    x_ben_pre: np.ndarray | None,
    x_mal_pre: np.ndarray | None,
) -> None:
    outputs = build_stage_output_files(
        primary_artifact_key="state",
        primary_artifact_name="stage2.pt",
        extra_outputs={
            "train_metrics": "stage2_train_metrics.csv" if train_log else None,
            "pareto": metrics_payload.get("pareto_path"),
            "adv_samples": "adv_samples.npz" if settings.save_samples else None,
            "intermediate_results": "intermediate_results.npz" if settings.save_intermediate_results else None,
        },
    )
    arrays = collect_manifest_arrays(
        {
            "adv_pre": x_adv_pre,
            "adv_ben_norm": x_adv_norm,
            "benign_pre": x_ben_pre,
            "mal_pre": x_mal_pre,
        }
    )
    save_stage_manifest_spec(
        StageManifestSpec(
            stage_name="stage2",
            out_dir=out_dir,
            config_path=config_path,
            inputs={
                "oracle_name": oracle_name,
                "feature_dim": int(x_train.shape[1]),
                "benign_rows": int(x_ben.shape[0]),
                "malicious_rows": int(x_mal.shape[0]),
                "mode": stage2_mode,
            },
            outputs=outputs,
            arrays=arrays,
            metrics={
                "asr_surrogate": metrics_payload.get("asr_surrogate"),
                "asr_oracle": metrics_payload.get("asr_oracle"),
                "sample_generation_time_sec": metrics_payload.get("sample_generation_time_sec"),
                "sample_generation_samples_per_sec": metrics_payload.get("sample_generation_samples_per_sec"),
                "sample_end_to_end_time_sec": metrics_payload.get("sample_end_to_end_time_sec"),
                "sample_end_to_end_samples_per_sec": metrics_payload.get("sample_end_to_end_samples_per_sec"),
                "mal_anchor_alpha": metrics_payload.get("mal_anchor_alpha"),
            },
        )
    )
