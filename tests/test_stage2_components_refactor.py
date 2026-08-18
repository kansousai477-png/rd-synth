from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.baselines.paper_attacks import generate_paper_attack_baseline
from rdsynth.stages import paper_attack_baselines, stage2_components
from rdsynth.stages.stage2_conditioning import (
    compose_condition_input,
    surrogate_embedding_dim,
    surrogate_guidance_dim,
    surrogate_output_dim,
)
from rdsynth.stages.stage2_networks import (
    AutoEncoder,
    ConditionalCritic,
    ConditionalGenerator,
    ConditionEncoder,
    LatentEditor,
)
from rdsynth.stages.stage2_training_utils import freeze_module, train_autoencoder


class _FeatureSurrogate(torch.nn.Module):
    feature_dim = 4
    num_classes = 3

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.stack([x[:, 0], x[:, 1], x[:, 0] - x[:, 1]], dim=1)
        features = torch.cat([x, x[:, :2]], dim=1)
        return logits, features[:, : self.feature_dim]


class _SequentialSurrogate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage2ComponentRefactorTest(unittest.TestCase):
    def test_conditioning_helpers_cover_modes_and_fallbacks(self) -> None:
        feature_surrogate = _FeatureSurrogate()
        seq_surrogate = _SequentialSurrogate()
        x = torch.tensor([[1.0, 2.0], [0.5, -0.5]], dtype=torch.float32)
        logits, features = feature_surrogate(x)

        self.assertEqual(surrogate_embedding_dim(feature_surrogate, 9), 4)
        self.assertEqual(surrogate_embedding_dim(seq_surrogate, 9), 2)
        self.assertEqual(surrogate_output_dim(feature_surrogate), 3)
        self.assertEqual(surrogate_output_dim(seq_surrogate), 2)
        self.assertEqual(surrogate_guidance_dim(feature_surrogate, 9, "raw_only"), 0)

        raw = compose_condition_input(x, (logits, features), guidance_mode="raw_only", cond_norm=True)
        logits_cond = compose_condition_input(x, (logits, features), guidance_mode="logits", guidance_norm=True)
        embed_cond = compose_condition_input(x, (logits, features), guidance_mode="embedding", guidance_norm=True)

        self.assertEqual(tuple(raw.shape), (2, 2))
        self.assertEqual(tuple(logits_cond.shape), (2, 5))
        self.assertEqual(tuple(embed_cond.shape), (2, 6))
        self.assertTrue(torch.allclose(torch.norm(embed_cond[:, -4:], dim=1), torch.ones(2), atol=1.0e-5))

    def test_network_modules_produce_expected_shapes(self) -> None:
        x = torch.randn(3, 6)
        cond = torch.randn(3, 4)
        noise = torch.randn(3, 5)

        encoder = ConditionEncoder(in_dim=4, emb_dim=7, hidden=8)
        autoencoder = AutoEncoder(in_dim=6, latent_dim=3, hidden=(8, 5))
        editor = LatentEditor(in_dim=6, latent_dim=3, hidden=(7, 4))
        generator = ConditionalGenerator(noise_dim=5, cond_dim=4, out_dim=6, hidden=9)
        critic = ConditionalCritic(in_dim=6, cond_dim=4, hidden=9)

        self.assertEqual(tuple(encoder(cond).shape), (3, 7))
        self.assertEqual(tuple(autoencoder(x).shape), (3, 6))
        self.assertEqual(tuple(editor(x).shape), (3, 3))
        self.assertEqual(tuple(generator(noise, cond).shape), (3, 6))
        self.assertEqual(tuple(critic(x, cond).shape), (3,))

    def test_training_utils_freeze_and_train_autoencoder(self) -> None:
        torch.manual_seed(0)
        ae = AutoEncoder(in_dim=2, latent_dim=1, hidden=(4, 3))
        dataset = TensorDataset(torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=3, shuffle=False)

        before = ae(torch.tensor([[0.25, 0.25]], dtype=torch.float32)).detach()
        train_autoencoder(ae, loader, epochs=5, lr=5.0e-2)
        after = ae(torch.tensor([[0.25, 0.25]], dtype=torch.float32)).detach()

        self.assertFalse(torch.equal(before, after))
        freeze_module(ae)
        self.assertFalse(ae.training)
        self.assertTrue(all(not param.requires_grad for param in ae.parameters()))

    def test_compatibility_facades_reexport_public_api(self) -> None:
        self.assertIs(stage2_components.compose_condition_input, compose_condition_input)
        self.assertIs(stage2_components.AutoEncoder, AutoEncoder)
        self.assertIs(stage2_components.train_autoencoder, train_autoencoder)
        self.assertIs(paper_attack_baselines.generate_paper_attack_baseline, generate_paper_attack_baseline)


if __name__ == "__main__":
    unittest.main()
