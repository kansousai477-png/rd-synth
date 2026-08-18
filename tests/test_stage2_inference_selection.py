from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_inference import make_stage2_predictors
from rdsynth.pipeline.stage2_selection import auto_select_stage2_candidates, select_per_sample_candidates


class _SimpleSurrogate(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((-x[:, 0], x[:, 0]), dim=1)


class _DummyEvalHelper:
    def evaluate_candidate(self, x_adv: np.ndarray, x_mal: np.ndarray):
        row = {
            "selection_score": float(np.mean(x_adv)),
            "decision_score": float(np.mean(x_adv)),
            "asr_oracle": float(np.mean(x_adv)),
            "asr_surrogate": float(np.mean(x_adv)),
            "adv_pmal_oracle": 0.1,
            "adv_pmal_surrogate": 0.1,
            "c2st_auc": 0.0,
            "adv_to_ben_l2": 0.0,
            "ffd": 0.0,
        }
        return x_adv, x_adv, row


class _ConstrainedEvalHelper:
    def evaluate_candidate(self, x_adv: np.ndarray, x_mal: np.ndarray):
        alpha = float(x_adv[0, 0])
        row = {
            "selection_score": 0.5 + alpha,
            "decision_score": 0.5 + alpha,
            "asr_oracle": 1.0,
            "asr_surrogate": 1.0,
            "adv_pmal_oracle": alpha,
            "adv_pmal_surrogate": alpha,
            "c2st_auc": 0.0,
            "adv_to_ben_l2": 0.0,
            "adv_to_mal_l2": 0.0,
            "ffd": 0.0,
        }
        return x_adv, x_adv, row


class Stage2InferenceSelectionTest(unittest.TestCase):
    def test_make_stage2_predictors_returns_attack_scores(self) -> None:
        predictors = make_stage2_predictors(
            surrogate=_SimpleSurrogate(),
            oracle=None,
            device=torch.device("cpu"),
            batch_size=8,
        )
        _, probs = predictors.surrogate_predict_probs(np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32))
        self.assertEqual(probs.shape, (2, 2))
        scores = predictors.attack_score_fn(np.asarray([[1.0, 0.0]], dtype=np.float32))
        self.assertEqual(scores.shape, (1,))

    def test_auto_select_stage2_candidates_global_mode(self) -> None:
        metrics = {}
        result = auto_select_stage2_candidates(
            pareto_cfg={"enable": True, "auto_select": True, "anchor_grid": [0.0, 1.0], "selection": {}},
            eval_helper=_DummyEvalHelper(),
            sample_with_alpha=lambda x, alpha: np.full_like(x, alpha),
            x_mal_eval=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            x_ben_norm=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            attack_score_fn=lambda x: np.mean(x, axis=1),
            metrics_payload=metrics,
        )
        self.assertIsNotNone(result.selected_adv_pre)
        self.assertIn("selected_candidate_mode", metrics)

    def test_auto_select_stage2_candidates_respects_hard_constraints(self) -> None:
        metrics = {}
        result = auto_select_stage2_candidates(
            pareto_cfg={
                "enable": True,
                "auto_select": True,
                "anchor_grid": [0.1, 0.9],
                "selection": {"max_adv_pmal_oracle": 0.5},
            },
            eval_helper=_ConstrainedEvalHelper(),
            sample_with_alpha=lambda x, alpha: np.full_like(x, alpha),
            x_mal_eval=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            x_ben_norm=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            attack_score_fn=lambda x: np.mean(x, axis=1),
            metrics_payload=metrics,
        )
        self.assertIsNotNone(result.selected_row)
        self.assertLessEqual(float(result.selected_row["adv_pmal_oracle"]), 0.5)
        self.assertEqual(metrics["candidate_count_total"], 2)
        self.assertEqual(metrics["candidate_count_feasible"], 1)
        self.assertFalse(metrics["selected_candidate_constraints_fallback"])

    def test_auto_select_stage2_candidates_refines_alpha_grid_iteratively(self) -> None:
        metrics = {}

        class RefineEvalHelper:
            def evaluate_candidate(self, x_adv: np.ndarray, x_mal: np.ndarray):
                alpha = float(x_adv[0, 0])
                score = 1.0 - abs(alpha - 0.25)
                row = {
                    "selection_score": score,
                    "decision_score": score,
                    "asr_oracle": score,
                    "asr_surrogate": score,
                    "adv_pmal_oracle": 1.0 - score,
                    "adv_pmal_surrogate": 1.0 - score,
                    "c2st_auc": 0.0,
                    "adv_to_ben_l2": 0.0,
                    "ffd": 0.0,
                }
                return x_adv, x_adv, row

        result = auto_select_stage2_candidates(
            pareto_cfg={
                "enable": True,
                "auto_select": True,
                "anchor_grid": [0.0, 0.5],
                "selection": {
                    "iterative_rounds": 2,
                    "iterative_points": 3,
                    "iterative_radius": 0.25,
                },
            },
            eval_helper=RefineEvalHelper(),
            sample_with_alpha=lambda x, alpha: np.full_like(x, alpha),
            x_mal_eval=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            x_ben_norm=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            attack_score_fn=lambda x: np.mean(x, axis=1),
            metrics_payload=metrics,
        )

        self.assertIsNotNone(result.selected_row)
        self.assertEqual(float(result.selected_row["mal_anchor_alpha"]), 0.25)
        self.assertEqual(metrics["candidate_count_total"], 3)
        self.assertEqual(metrics["candidate_selection_iterative_rounds_used"], 2)

    def test_per_sample_selection_handles_different_benign_reference_count(self) -> None:
        candidate_cache = {
            0.0: (
                np.asarray([[0.0, 0.0], [10.0, 10.0], [2.0, 2.0]], dtype=np.float32),
                np.asarray([[0.0, 0.0], [10.0, 10.0], [2.0, 2.0]], dtype=np.float32),
                {},
            ),
            1.0: (
                np.asarray([[5.0, 5.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.float32),
                np.asarray([[5.0, 5.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.float32),
                {},
            ),
        }
        benign_reference = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        adv_pre, adv_norm, summary = select_per_sample_candidates(
            candidate_cache,
            [0.0, 1.0],
            score_fn=lambda x: np.zeros(x.shape[0], dtype=np.float64),
            distance_weight=1.0,
            benign_reference=benign_reference,
        )

        self.assertEqual(adv_pre.shape, (3, 2))
        np.testing.assert_allclose(adv_norm[0], [0.0, 0.0])
        np.testing.assert_allclose(adv_norm[1], [1.0, 1.0])
        self.assertAlmostEqual(summary["selected_alpha_frac_0p0"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["selected_alpha_frac_1p0"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
