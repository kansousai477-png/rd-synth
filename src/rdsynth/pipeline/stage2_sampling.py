from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rdsynth.stages.stage2_diffusion import (
    sample_conditional_gan,
    sample_editor,
    sample_latent_diffusion,
)


def build_stage2_sampler(
    *,
    stage2_mode: str,
    generator_backbone: str,
    settings: Any,
    diffusion_bundle: Any,
    surrogate: Any,
    device: Any,
    guidance_mode: str,
    benign_pool: np.ndarray,
) -> Callable[[np.ndarray, float], np.ndarray]:
    def sample_with_alpha(x_mal_local: np.ndarray, alpha: float) -> np.ndarray:
        if stage2_mode == "latent_diffusion" and generator_backbone in {"gan", "cgan", "wgan"}:
            return sample_conditional_gan(
                diffusion_bundle,
                x_mal_local,
                surrogate=surrogate,
                device=device,
                batch_size=settings.sample_batch_size,
                mal_anchor_alpha=float(alpha),
                clip_minmax=settings.sample_clip_minmax,
                denorm_output=settings.sample_denorm_output,
                input_normalized=False,
            )
        if stage2_mode == "latent_diffusion":
            return sample_latent_diffusion(
                diffusion_bundle,
                x_mal_local,
                surrogate=surrogate,
                device=device,
                batch_size=settings.sample_batch_size,
                init_mode=settings.sample_init_latent,
                benign_pool=benign_pool,
                use_prior=settings.latent_use_prior,
                guidance_scale=settings.guidance_scale,
                noise_scale=settings.latent_noise_scale,
                mal_anchor_alpha=float(alpha),
                clip_minmax=settings.sample_clip_minmax,
                denorm_output=settings.sample_denorm_output,
                input_normalized=False,
                guidance_mode=guidance_mode,
            )
        return sample_editor(
            diffusion_bundle,
            x_mal_local,
            surrogate=surrogate,
            device=device,
            batch_size=settings.sample_batch_size,
            init_mode=settings.sample_init_editor,
            benign_pool=benign_pool,
            residual_scale=settings.residual_scale,
            mal_anchor_alpha=float(alpha),
            clip_minmax=settings.sample_clip_minmax,
            denorm_output=settings.sample_denorm_output,
            guidance_mode=guidance_mode,
            conditioning_enabled=getattr(diffusion_bundle, "conditioning_enabled", True),
        )

    return sample_with_alpha
