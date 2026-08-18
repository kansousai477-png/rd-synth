from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.baselines.paper_attacks import (
    PAPER_BASELINE_SPECS,
    generate_paper_attack_baseline,
    traffic_space_baseline_names,
)


class PaperAttackBaselinesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_names = ["Dst Port", "ACK Flag Count", "Flow IAT Mean", "Packet Length Mean"]
        self.x_ben = np.array(
            [
                [80.0, 0.0, 5.0, 80.0],
                [443.0, 0.0, 6.0, 82.0],
                [53.0, 1.0, 4.5, 78.0],
            ],
            dtype=np.float64,
        )
        self.x_mal = np.array(
            [
                [8080.0, 1.0, 50.0, 600.0],
                [4444.0, 1.0, 45.0, 550.0],
            ],
            dtype=np.float64,
        )

    def _score_fn(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x[:, 2] + 0.01 * x[:, 3]

    def test_all_registered_baselines_generate_same_shape(self) -> None:
        for name in PAPER_BASELINE_SPECS:
            adv = generate_paper_attack_baseline(
                name=name,
                x_mal_pre=self.x_mal,
                x_ben_pre=self.x_ben,
                feature_names=self.feature_names,
                score_fn=self._score_fn,
                seed=7,
            )
            self.assertEqual(adv.shape, self.x_mal.shape)
            self.assertTrue(np.isfinite(adv).all())

    def test_traffic_space_registry_contains_expected_methods(self) -> None:
        names = traffic_space_baseline_names()
        self.assertIn("gpmt_lite", names)
        self.assertIn("progen_lite", names)
        self.assertIn("amoeba_lite", names)
        self.assertIn("netdiffusion_lite", names)
        self.assertNotIn("idsgan_lite", names)

    def test_function_preserving_baseline_keeps_binary_flag_unchanged(self) -> None:
        adv = generate_paper_attack_baseline(
            name="digfupas_lite",
            x_mal_pre=self.x_mal,
            x_ben_pre=self.x_ben,
            feature_names=self.feature_names,
            score_fn=self._score_fn,
            seed=3,
        )
        np.testing.assert_allclose(adv[:, 1], self.x_mal[:, 1], atol=1.0e-8)

    def test_amoeba_lite_improves_black_box_score_on_toy_case(self) -> None:
        adv = generate_paper_attack_baseline(
            name="amoeba_lite",
            x_mal_pre=self.x_mal,
            x_ben_pre=self.x_ben,
            feature_names=self.feature_names,
            score_fn=self._score_fn,
            device=torch.device("cpu"),
            seed=5,
        )
        self.assertLessEqual(float(np.mean(self._score_fn(adv))), float(np.mean(self._score_fn(self.x_mal))))

    def test_progen_lite_improves_black_box_score_on_toy_case(self) -> None:
        adv = generate_paper_attack_baseline(
            name="progen_lite",
            x_mal_pre=self.x_mal,
            x_ben_pre=self.x_ben,
            feature_names=self.feature_names,
            score_fn=self._score_fn,
            device=torch.device("cpu"),
            seed=17,
        )
        self.assertLessEqual(float(np.mean(self._score_fn(adv))), float(np.mean(self._score_fn(self.x_mal))))

    def test_gpmt_lite_keeps_port_feature_fixed(self) -> None:
        adv = generate_paper_attack_baseline(
            name="gpmt_lite",
            x_mal_pre=self.x_mal,
            x_ben_pre=self.x_ben,
            feature_names=self.feature_names,
            score_fn=self._score_fn,
            device=torch.device("cpu"),
            seed=13,
        )
        np.testing.assert_allclose(adv[:, 0], self.x_mal[:, 0], atol=1.0e-8)

    def test_netdiffusion_lite_keeps_port_feature_fixed(self) -> None:
        adv = generate_paper_attack_baseline(
            name="netdiffusion_lite",
            x_mal_pre=self.x_mal,
            x_ben_pre=self.x_ben,
            feature_names=self.feature_names,
            score_fn=self._score_fn,
            device=torch.device("cpu"),
            seed=19,
        )
        np.testing.assert_allclose(adv[:, 0], self.x_mal[:, 0], atol=1.0e-8)

    def test_vulnergan_lite_supports_poison_training_inputs(self) -> None:
        x_train = np.vstack([self.x_ben, self.x_mal])
        y_train = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        adv = generate_paper_attack_baseline(
            name="vulnergan_lite",
            x_mal_pre=self.x_mal,
            x_ben_pre=self.x_ben,
            feature_names=self.feature_names,
            score_fn=self._score_fn,
            x_train_pre=x_train,
            y_train=y_train,
            device=torch.device("cpu"),
            seed=11,
        )
        self.assertEqual(adv.shape, self.x_mal.shape)
        self.assertTrue(np.isfinite(adv).all())


if __name__ == "__main__":
    unittest.main()
