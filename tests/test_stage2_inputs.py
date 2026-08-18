from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_inputs import (
    coerce_port_allowlist,
    prepare_stage2_constraint_inputs,
    prepare_stage2_eval_inputs,
)


class _IdentityPreprocessor:
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)


class _DummySettings:
    eval_samples = 2
    sample_denorm_output = False


class Stage2InputsTest(unittest.TestCase):
    def test_coerce_port_allowlist_accepts_strings_and_sequences(self) -> None:
        self.assertEqual(coerce_port_allowlist("80, 443, bad"), [80, 443])
        self.assertEqual(coerce_port_allowlist([53, "8080", "bad"]), [53, 8080])

    def test_prepare_stage2_eval_inputs_builds_eval_views(self) -> None:
        cfg = {"stage2": {"eval_denorm_metrics": True, "sample_pullback_alpha": 0.1, "sample_pullback_k": 2}}
        diffusion_bundle = SimpleNamespace(
            ben_stats={
                "denorm_mean": np.asarray([1.0, 2.0], dtype=np.float32),
                "denorm_std": np.asarray([2.0, 4.0], dtype=np.float32),
            }
        )
        x_ben = np.asarray([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]], dtype=np.float32)
        x_mal = np.asarray([[3.0, 4.0], [7.0, 8.0], [11.0, 12.0]], dtype=np.float32)

        inputs = prepare_stage2_eval_inputs(
            cfg=cfg,
            settings=_DummySettings(),
            seed=7,
            x_ben=x_ben,
            x_mal=x_mal,
            diffusion_bundle=diffusion_bundle,
            preprocessor=_IdentityPreprocessor(),
        )

        self.assertEqual(inputs.x_ben_eval.shape, (2, 2))
        self.assertEqual(inputs.x_mal_eval.shape, (2, 2))
        self.assertTrue(inputs.eval_denorm)
        self.assertIsNotNone(inputs.x_ben_denorm)
        self.assertIsNotNone(inputs.pre_min)
        self.assertEqual(inputs.pull_alpha, 0.1)
        self.assertEqual(inputs.pull_k, 2)

    def test_prepare_stage2_constraint_inputs_sets_metrics_and_targets(self) -> None:
        cfg = {
            "stage2": {
                "constraints": {"enable": True},
                "deployable_constraints": {
                    "enable": True,
                    "port_policy": "set",
                    "flag_policy": "clip",
                    "temporal_policy": "clip_benign",
                    "port_allowlist": "80,443",
                },
            }
        }
        diffusion_bundle = SimpleNamespace(ben_stats={"min": np.zeros(2), "max": np.ones(2)})
        metrics_payload: dict[str, object] = {}
        inputs = prepare_stage2_constraint_inputs(
            cfg=cfg,
            diffusion_bundle=diffusion_bundle,
            preprocessor=_IdentityPreprocessor(),
            feature_names=["dst_port", "flow_duration"],
            x_ben=np.asarray([[80.0, 1.0], [443.0, 2.0]], dtype=np.float32),
            x_ben_norm=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            metrics_payload=metrics_payload,
        )

        self.assertTrue(inputs.constraints_enabled)
        self.assertTrue(inputs.deploy_enabled)
        self.assertEqual(inputs.port_allowlist, [80, 443])
        self.assertIsNotNone(inputs.constraints_spec)
        self.assertIsNotNone(inputs.x_ben_mod_targets)
        self.assertTrue(metrics_payload["constraints_enabled"])
        self.assertTrue(metrics_payload["deployable_constraints_enabled"])


if __name__ == "__main__":
    unittest.main()
