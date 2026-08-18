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

from rdsynth.pipeline.stage2_eval import Stage2EvalHelper, sanitize_feature_array


class _IdentityPreprocessor:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


class Stage2EvalHelperTest(unittest.TestCase):
    def _make_helper(self, **overrides: object) -> Stage2EvalHelper:
        kwargs = dict(
            cfg={"stage2": {"sample_denorm_output": False, "post_clip_norm_range": True, "sample_batch_size": 8}},
            seed=1,
            preprocessor=_IdentityPreprocessor(),
            bundle_feature_names=["f1", "f2"],
            denorm_mean=np.zeros(2, dtype=np.float64),
            denorm_std=np.ones(2, dtype=np.float64),
            x_ben_norm=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            x_ben_raw_full=np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float64),
            constraints_enabled=False,
            constraints_spec=None,
            constraints_cfg={},
            deploy_enabled=False,
            traffic_schema={"name": "toy"},
            port_policy="keep",
            flag_policy="clip",
            temporal_policy="clip_benign",
            port_allowlist=[],
            diffusion_bundle=SimpleNamespace(ben_stats={"min": np.array([-1.0, -1.0]), "max": np.array([1.0, 1.0])}),
            norm_bounds_min=np.array([-1.0, -1.0], dtype=np.float64),
            norm_bounds_max=np.array([1.0, 1.0], dtype=np.float64),
            norm_nonneg=np.zeros(2, dtype=bool),
            x_ben_mod_targets=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            remap_target_mean=np.array([0.2, 0.3], dtype=np.float64),
            remap_target_scale=np.array([1.0, 1.0], dtype=np.float64),
            pull_alpha=0.0,
            pull_k=0,
            moment_alpha=0.0,
            moment_std_floor=1.0e-3,
            surrogate_predict_probs=lambda x, batch: (
                np.zeros(len(x), dtype=int),
                np.column_stack([np.full(len(x), 0.8), np.full(len(x), 0.2)]),
            ),
            oracle_predict_probs=lambda x, batch: (
                np.zeros(len(x), dtype=int),
                np.column_stack([np.full(len(x), 0.9), np.full(len(x), 0.1)]),
            ),
            oracle=object(),
        )
        kwargs.update(overrides)
        return Stage2EvalHelper(**kwargs)

    def test_sanitize_feature_array_replaces_non_finite_values(self) -> None:
        values = sanitize_feature_array(np.array([[np.nan, np.inf, -np.inf, 7.0e5]], dtype=np.float64), clip_value=10.0)
        np.testing.assert_allclose(values, np.array([[0.0, 0.0, 0.0, 10.0]], dtype=np.float64))

    def test_postprocess_adv_clips_to_norm_bounds(self) -> None:
        helper = self._make_helper(
            cfg={"stage2": {"sample_denorm_output": False, "post_clip_norm_range": True}},
            x_ben_raw_full=None,
            x_ben_mod_targets=None,
            remap_target_mean=None,
            remap_target_scale=None,
            oracle=None,
            oracle_predict_probs=lambda x, batch: (np.array([], dtype=int), None),
        )
        adv_pre, adv_norm = helper.postprocess_adv(
            np.array([[5.0, -5.0]], dtype=np.float64),
            np.array([[0.0, 0.0]], dtype=np.float64),
        )
        np.testing.assert_allclose(adv_norm, np.array([[1.0, -1.0]], dtype=np.float64))
        np.testing.assert_allclose(adv_pre, np.array([[1.0, -1.0]], dtype=np.float64))

    def test_postprocess_adv_caps_pullback_k_to_available_benign_rows(self) -> None:
        helper = self._make_helper(
            cfg={"stage2": {"sample_denorm_output": False, "post_clip_norm_range": False}},
            x_ben_norm=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
            x_ben_raw_full=None,
            norm_bounds_min=None,
            norm_bounds_max=None,
            pull_alpha=0.5,
            pull_k=5,
        )

        adv_pre, adv_norm = helper.postprocess_adv(
            np.array([[4.0, 4.0]], dtype=np.float64),
            np.array([[0.0, 0.0]], dtype=np.float64),
        )

        np.testing.assert_allclose(adv_norm, np.array([[2.25, 2.25]], dtype=np.float64))
        np.testing.assert_allclose(adv_pre, np.array([[2.25, 2.25]], dtype=np.float64))

    def test_postprocess_adv_applies_constraints_and_schema_projection(self) -> None:
        helper = self._make_helper(
            constraints_enabled=True,
            constraints_spec=None,
            constraints_cfg={"clip": True, "round_integer": False},
            deploy_enabled=True,
        )
        with (
            patch("rdsynth.pipeline.stage2_eval.infer_constraints", return_value={"spec": True}) as infer_constraints,
            patch(
                "rdsynth.pipeline.stage2_eval.apply_constraints",
                side_effect=lambda x, spec, clip, round_integer: x + 1.0,
            ) as apply_constraints,
            patch(
                "rdsynth.pipeline.stage2_eval.apply_schema_projection",
                side_effect=lambda **kwargs: kwargs["x_adv"] + 2.0,
            ) as apply_schema_projection,
        ):
            adv_pre, adv_norm = helper.postprocess_adv(
                np.array([[0.5, -0.5]], dtype=np.float64),
                np.array([[0.1, 0.2]], dtype=np.float64),
            )

        infer_constraints.assert_called_once()
        apply_constraints.assert_called_once()
        apply_schema_projection.assert_called_once()
        np.testing.assert_allclose(adv_pre, np.array([[1.0, 1.0]], dtype=np.float64))
        np.testing.assert_allclose(adv_norm, np.array([[1.0, 1.0]], dtype=np.float64))

    def test_evaluate_candidate_combines_metrics_and_scores(self) -> None:
        helper = self._make_helper()
        with (
            patch(
                "rdsynth.pipeline.stage2_eval.compute_stage2_metrics",
                return_value=SimpleNamespace(
                    as_dict=lambda: {
                        "FFD": 0.2,
                        "SWD": 0.3,
                        "Energy": 0.4,
                        "C2ST-AUC": 0.5,
                        "C2ST-Acc": 0.6,
                    }
                ),
            ),
            patch("rdsynth.pipeline.stage2_eval.nearest_reference_distance", return_value=1.25),
            patch("rdsynth.pipeline.stage2_eval.paired_sample_l2", return_value=2.5),
            patch(
                "rdsynth.pipeline.stage2_eval.build_remap_targets",
                return_value=np.array([[0.2, 0.4]], dtype=np.float64),
            ),
            patch(
                "rdsynth.pipeline.stage2_eval.build_rule_based_modifications",
                return_value=np.array([[0.3, 0.5]], dtype=np.float64),
            ),
            patch(
                "rdsynth.pipeline.stage2_eval.clip_modifications",
                return_value=np.array([[0.25, 0.45]], dtype=np.float64),
            ),
        ):
            adv_pre, adv_norm, row = helper.evaluate_candidate(
                np.array([[0.5, 0.2]], dtype=np.float64),
                np.array([[0.4, 0.1]], dtype=np.float64),
            )

        np.testing.assert_allclose(adv_pre, np.array([[0.5, 0.2]], dtype=np.float64))
        np.testing.assert_allclose(adv_norm, np.array([[0.5, 0.2]], dtype=np.float64))
        self.assertEqual(row["ffd"], 0.2)
        self.assertEqual(row["adv_to_ben_l2"], 1.25)
        self.assertEqual(row["adv_to_mal_l2"], 2.5)
        self.assertAlmostEqual(row["asr_surrogate"], 1.0)
        self.assertAlmostEqual(row["asr_oracle"], 1.0)
        self.assertIn("remapability_score", row)
        self.assertIn("support_score", row)
        # selection_score = asr_oracle - 0.01 * ffd = 1.0 - 0.002 = 0.998
        self.assertAlmostEqual(row["selection_score"], 1.0 - 0.01 * 0.2, places=5)
        # stage3_closed_loop_score = asr_oracle = 1.0
        self.assertAlmostEqual(row["stage3_closed_loop_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
