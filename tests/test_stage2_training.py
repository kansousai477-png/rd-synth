from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_training import train_stage2_generator, validate_stage2_training_options


class _Settings:
    def __init__(self, mode: str, generator_backbone: str, guidance_mode: str) -> None:
        self.mode = mode
        self.generator_backbone = generator_backbone
        self.guidance_mode = guidance_mode
        self.epochs = 1
        self.batch_size = 2
        self.lr = 1.0e-3


class Stage2TrainingTest(unittest.TestCase):
    def test_validate_stage2_training_options_rejects_invalid_combo(self) -> None:
        with self.assertRaises(ValueError):
            validate_stage2_training_options(
                stage2_mode="editor",
                generator_backbone="cgan",
                guidance_mode="embedding",
            )

    @patch("rdsynth.pipeline.stage2_training.train_latent_diffusion")
    def test_train_stage2_generator_dispatches_ddpm(self, mock_train) -> None:
        mock_train.return_value = "bundle"
        cfg = {"stage2": {"timesteps": 10, "lambda_stp": 1.0, "lambda_mmt": 1.0}}
        result = train_stage2_generator(
            cfg=cfg,
            settings=_Settings("latent_diffusion", "ddpm", "embedding"),
            x_ben=np.zeros((2, 2), dtype=np.float32),
            x_mal=np.zeros((2, 2), dtype=np.float32),
            feature_names=["f0", "f1"],
            surrogate=torch.nn.Linear(2, 2),
            device=torch.device("cpu"),
        )
        self.assertEqual(result, "bundle")
        self.assertTrue(mock_train.called)


if __name__ == "__main__":
    unittest.main()
