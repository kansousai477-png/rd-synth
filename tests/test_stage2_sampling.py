from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_sampling import build_stage2_sampler


class Stage2SamplingTest(unittest.TestCase):
    def _settings(self) -> SimpleNamespace:
        return SimpleNamespace(
            sample_batch_size=8,
            sample_clip_minmax=True,
            sample_denorm_output=False,
            sample_init_latent="benign_sample",
            sample_init_editor="benign_mean",
            latent_use_prior=False,
            guidance_scale=1.5,
            latent_noise_scale=1.0,
            residual_scale=0.5,
        )

    def test_build_stage2_sampler_dispatches_to_cgan_sampler(self) -> None:
        with patch(
            "rdsynth.pipeline.stage2_sampling.sample_conditional_gan", return_value=np.asarray([[1.0]])
        ) as sample_fn:
            sampler = build_stage2_sampler(
                stage2_mode="latent_diffusion",
                generator_backbone="cgan",
                settings=self._settings(),
                diffusion_bundle=object(),
                surrogate=object(),
                device="cpu",
                guidance_mode="embedding",
                benign_pool=np.zeros((1, 1), dtype=np.float32),
            )
            out = sampler(np.zeros((1, 1), dtype=np.float32), 0.3)

        self.assertEqual(out.tolist(), [[1.0]])
        sample_fn.assert_called_once()

    def test_build_stage2_sampler_dispatches_to_latent_diffusion_sampler(self) -> None:
        with patch(
            "rdsynth.pipeline.stage2_sampling.sample_latent_diffusion", return_value=np.asarray([[2.0]])
        ) as sample_fn:
            sampler = build_stage2_sampler(
                stage2_mode="latent_diffusion",
                generator_backbone="ddpm",
                settings=self._settings(),
                diffusion_bundle=object(),
                surrogate=object(),
                device="cpu",
                guidance_mode="embedding",
                benign_pool=np.zeros((1, 1), dtype=np.float32),
            )
            out = sampler(np.zeros((1, 1), dtype=np.float32), 0.4)

        self.assertEqual(out.tolist(), [[2.0]])
        sample_fn.assert_called_once()

    def test_build_stage2_sampler_dispatches_to_editor_sampler(self) -> None:
        with patch("rdsynth.pipeline.stage2_sampling.sample_editor", return_value=np.asarray([[3.0]])) as sample_fn:
            sampler = build_stage2_sampler(
                stage2_mode="editor",
                generator_backbone="ddpm",
                settings=self._settings(),
                diffusion_bundle=object(),
                surrogate=object(),
                device="cpu",
                guidance_mode="embedding",
                benign_pool=np.zeros((1, 1), dtype=np.float32),
            )
            out = sampler(np.zeros((1, 1), dtype=np.float32), 0.5)

        self.assertEqual(out.tolist(), [[3.0]])
        sample_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
