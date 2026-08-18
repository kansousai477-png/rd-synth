from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from rdsynth.models.diffusion import ConditionalDenoiser, make_cosine_schedule, make_linear_schedule
from rdsynth.pipeline.stage2_sampling import build_stage2_sampler
from rdsynth.pipeline.stage2_setup import Stage2Settings
from rdsynth.stages.stage2_bundles import EditorBundle, GanBundle, LatentDiffusionBundle
from rdsynth.stages.stage2_networks import AutoEncoder, ConditionalCritic, ConditionalGenerator, LatentEditor
from rdsynth.utils.checkpoints import load_torch_state


@dataclass(frozen=True)
class Stage2CheckpointSampler:
    sampler: Callable[[np.ndarray, float], np.ndarray]
    bundle: Any
    settings: Stage2Settings
    checkpoint_path: Path
    output_is_preprocessed: bool


def _state_dict(payload: Mapping[str, Any], key: str) -> dict[str, torch.Tensor]:
    state = payload.get(key)
    if not isinstance(state, Mapping):
        raise ValueError(f"Stage2 checkpoint missing {key}.")
    return dict(state)


def _load_sequential_state(module: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    module.load_state_dict(dict(state))
    module.eval()


def _autoencoder_from_states(
    *,
    encoder_state: Mapping[str, torch.Tensor],
    decoder_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> AutoEncoder:
    first = encoder_state["0.weight"]
    middle = encoder_state["3.weight"]
    last = encoder_state["6.weight"]
    in_dim = int(first.shape[1])
    h1 = int(first.shape[0])
    h2 = int(middle.shape[0])
    latent_dim = int(last.shape[0])
    ae = AutoEncoder(in_dim, latent_dim, (h1, h2)).to(device)
    _load_sequential_state(ae.encoder, encoder_state)
    _load_sequential_state(ae.decoder, decoder_state)
    return ae


def _schedule_from_cfg(stage2_cfg: Mapping[str, Any], device: torch.device):
    timesteps = int(stage2_cfg.get("timesteps", 100))
    beta_start = float(stage2_cfg.get("beta_start", 1.0e-4))
    beta_end = float(stage2_cfg.get("beta_end", 2.0e-2))
    schedule_type = str(stage2_cfg.get("schedule_type", "cosine")).lower()
    if schedule_type == "linear":
        return make_linear_schedule(timesteps, beta_start, beta_end, device)
    return make_cosine_schedule(timesteps, beta_start, beta_end, device)


def _restore_latent_diffusion(
    *,
    payload: Mapping[str, Any],
    stage2_cfg: Mapping[str, Any],
    device: torch.device,
) -> LatentDiffusionBundle:
    ae = _autoencoder_from_states(
        encoder_state=_state_dict(payload, "encoder_state"),
        decoder_state=_state_dict(payload, "decoder_state"),
        device=device,
    )
    denoiser_state = _state_dict(payload, "denoiser_state")
    in_dim = int(denoiser_state["in_proj.weight"].shape[1])
    hidden_dim = int(denoiser_state["in_proj.weight"].shape[0])
    cond_dim = int(denoiser_state["cond_mlp.0.weight"].shape[1])
    denoiser = ConditionalDenoiser(
        in_dim=in_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        time_dim=64,
        dropout=float(stage2_cfg.get("denoiser_dropout", 0.05)),
        predict_x0=bool(payload.get("predict_x0", stage2_cfg.get("predict_x0", True))),
    ).to(device)
    denoiser.load_state_dict(denoiser_state)
    denoiser.eval()
    latent_mean = payload.get("latent_mean")
    latent_std = payload.get("latent_std")
    if not isinstance(latent_mean, torch.Tensor) or not isinstance(latent_std, torch.Tensor):
        raise ValueError("Stage2 latent diffusion checkpoint missing latent statistics.")
    return LatentDiffusionBundle(
        denoiser=denoiser,
        encoder=ae.encoder,
        decoder=ae.decoder,
        schedule=_schedule_from_cfg(stage2_cfg, device),
        groups=dict(payload.get("groups") or {}),
        ben_stats={key: np.asarray(value) for key, value in dict(payload.get("ben_stats") or {}).items()},
        latent_mean=latent_mean.to(device),
        latent_std=latent_std.to(device),
        predict_x0=bool(payload.get("predict_x0", stage2_cfg.get("predict_x0", True))),
        x0_head_tanh=bool(payload.get("x0_head_tanh", stage2_cfg.get("x0_head_tanh", True))),
        cond_norm=bool(payload.get("cond_norm", stage2_cfg.get("cond_norm", True))),
        emb_norm=bool(payload.get("emb_norm", stage2_cfg.get("emb_norm", True))),
        eps_pred_clip=float(payload.get("eps_pred_clip", stage2_cfg.get("eps_pred_clip", 3.0))),
        train_log=payload.get("train_log"),
        best_epoch=payload.get("best_epoch"),
        best_score=payload.get("best_score"),
    )


def _restore_editor(
    *,
    payload: Mapping[str, Any],
    device: torch.device,
) -> EditorBundle:
    ae = _autoencoder_from_states(
        encoder_state=_state_dict(payload, "encoder_state"),
        decoder_state=_state_dict(payload, "decoder_state"),
        device=device,
    )
    editor_state = _state_dict(payload, "editor_state")
    cond_dim = int(editor_state["net.0.weight"].shape[1])
    h1 = int(editor_state["net.0.weight"].shape[0])
    h2 = int(editor_state["net.3.weight"].shape[0])
    latent_dim = int(editor_state["net.6.weight"].shape[0])
    editor = LatentEditor(cond_dim, latent_dim, (h1, h2)).to(device)
    editor.load_state_dict(editor_state)
    editor.eval()
    return EditorBundle(
        encoder=ae.encoder,
        decoder=ae.decoder,
        editor=editor,
        groups=dict(payload.get("groups") or {}),
        ben_stats={key: np.asarray(value) for key, value in dict(payload.get("ben_stats") or {}).items()},
        latent_dim=int(payload.get("latent_dim", latent_dim)),
        train_log=payload.get("train_log"),
        best_epoch=payload.get("best_epoch"),
        best_score=payload.get("best_score"),
    )


def _restore_gan(
    *,
    payload: Mapping[str, Any],
    device: torch.device,
) -> GanBundle:
    generator_state = _state_dict(payload, "generator_state")
    critic_state = _state_dict(payload, "critic_state")
    noise_dim = int(payload.get("noise_dim", 64))
    cond_dim = int(generator_state["net.0.weight"].shape[1]) - noise_dim
    out_dim = int(generator_state["net.6.weight"].shape[0])
    hidden = int(generator_state["net.0.weight"].shape[0])
    generator = ConditionalGenerator(noise_dim=noise_dim, cond_dim=cond_dim, out_dim=out_dim, hidden=hidden).to(device)
    generator.load_state_dict(generator_state)
    generator.eval()
    critic_hidden = int(critic_state["net.0.weight"].shape[0])
    critic = ConditionalCritic(in_dim=out_dim, cond_dim=cond_dim, hidden=critic_hidden).to(device)
    critic.load_state_dict(critic_state)
    critic.eval()
    return GanBundle(
        generator=generator,
        critic=critic,
        groups=dict(payload.get("groups") or {}),
        ben_stats={key: np.asarray(value) for key, value in dict(payload.get("ben_stats") or {}).items()},
        noise_dim=noise_dim,
        guidance_mode=str(payload.get("guidance_mode", "embedding")),
        gan_type=str(payload.get("gan_type", "cgan")),
        train_log=payload.get("train_log"),
        best_epoch=payload.get("best_epoch"),
        best_score=payload.get("best_score"),
    )


def load_stage2_checkpoint_sampler(
    *,
    cfg: Mapping[str, Any],
    project_out_dir: str | Path,
    feature_names: list[str],
    surrogate: Any,
    device: torch.device,
    benign_pool: np.ndarray,
) -> Stage2CheckpointSampler:
    stage2_cfg = cfg["stage2"]
    settings = Stage2Settings.from_cfg(stage2_cfg)
    checkpoint_path = Path(project_out_dir) / "stage2" / "stage2.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Stage2 checkpoint not found at {checkpoint_path}. Run Stage2 before pcap_conditioned Stage3."
        )
    project_cfg = cfg.get("project", {}) if isinstance(cfg.get("project"), Mapping) else {}
    payload = load_torch_state(
        checkpoint_path,
        map_location=device,
        allow_unsafe=bool(project_cfg.get("allow_unsafe_checkpoint_load", False)),
    )
    if payload is None:
        try:
            payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=device)
        except Exception as exc:
            print(
                f"[Stage2/Checkpoint][Warn] safe/typed load failed for {checkpoint_path}: {exc}; falling back to weights_only=False"
            )
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_feature_names = payload.get("feature_names")
    if saved_feature_names is not None and [str(v) for v in saved_feature_names] != [str(v) for v in feature_names]:
        raise ValueError("Stage2 checkpoint feature_names mismatch with current dataset.")

    stage2_mode = settings.mode
    generator_backbone = settings.generator_backbone
    if stage2_mode == "latent_diffusion" and generator_backbone == "ddpm":
        bundle = _restore_latent_diffusion(payload=payload, stage2_cfg=stage2_cfg, device=device)
    elif stage2_mode == "latent_diffusion" and generator_backbone in {"gan", "cgan", "wgan"}:
        bundle = _restore_gan(payload=payload, device=device)
    else:
        bundle = _restore_editor(payload=payload, device=device)

    sampler = build_stage2_sampler(
        stage2_mode=stage2_mode,
        generator_backbone=generator_backbone,
        settings=settings,
        diffusion_bundle=bundle,
        surrogate=surrogate,
        device=device,
        guidance_mode=settings.guidance_mode,
        benign_pool=benign_pool,
    )
    return Stage2CheckpointSampler(
        sampler=sampler,
        bundle=bundle,
        settings=settings,
        checkpoint_path=checkpoint_path,
        output_is_preprocessed=bool(settings.sample_denorm_output),
    )
