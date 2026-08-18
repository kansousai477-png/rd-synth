from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_checkpoint import load_stage2_checkpoint_sampler
from rdsynth.stages.stage2_networks import AutoEncoder, LatentEditor


class _TinySurrogate(nn.Module):
    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits = torch.stack([torch.ones(x.size(0), device=x.device), torch.zeros(x.size(0), device=x.device)], dim=1)
        if return_features:
            return logits, x
        return logits


class Stage2CheckpointSamplerTest(unittest.TestCase):
    def test_loads_editor_checkpoint_and_samples_conditioned_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage2_dir = root / "stage2"
            stage2_dir.mkdir(parents=True)

            ae = AutoEncoder(in_dim=2, latent_dim=2, hidden=(4, 3))
            editor = LatentEditor(in_dim=4, latent_dim=2, hidden=(5, 4))
            payload = {
                "artifact_version": 1,
                "encoder_state": ae.encoder.state_dict(),
                "decoder_state": ae.decoder.state_dict(),
                "editor_state": editor.state_dict(),
                "groups": {},
                "feature_names": ["f0", "f1"],
                "ben_stats": {
                    "mean": np.zeros(2, dtype=np.float32),
                    "std": np.ones(2, dtype=np.float32),
                    "min": np.full(2, -3.0, dtype=np.float32),
                    "max": np.full(2, 3.0, dtype=np.float32),
                    "denorm_mean": np.zeros(2, dtype=np.float32),
                    "denorm_std": np.ones(2, dtype=np.float32),
                },
                "latent_dim": 2,
            }
            torch.save(payload, stage2_dir / "stage2.pt")

            cfg = {
                "project": {"out_dir": str(root), "allow_unsafe_checkpoint_load": True},
                "stage2": {
                    "oracle_name": "mlp_small",
                    "require_stage1": False,
                    "mode": "editor",
                    "generator_backbone": "ddpm",
                    "surrogate_guidance_mode": "embedding",
                    "epochs": 1,
                    "batch_size": 2,
                    "lr": 1.0e-3,
                    "sample_batch_size": 2,
                    "sample_denorm_output": True,
                },
            }
            loaded = load_stage2_checkpoint_sampler(
                cfg=cfg,
                project_out_dir=root,
                feature_names=["f0", "f1"],
                surrogate=_TinySurrogate(),
                device=torch.device("cpu"),
                benign_pool=np.zeros((2, 2), dtype=np.float32),
            )
            adv = loaded.sampler(np.asarray([[0.2, 0.3]], dtype=np.float32), 0.0)

        self.assertEqual(adv.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(adv)))
        self.assertTrue(loaded.output_is_preprocessed)


if __name__ == "__main__":
    unittest.main()
