from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_execution import (
    Stage2ExecutionSelection,
    execute_stage2_sample,
    run_stage2_candidate_selection,
)
from rdsynth.utils.query_oracle import QueryOracle


class Stage2ExecutionTest(unittest.TestCase):
    def test_run_stage2_candidate_selection_uses_selected_result(self) -> None:
        metrics_payload: dict[str, object] = {}
        selection_result = SimpleNamespace(
            selected_alpha=0.3,
            selected_adv_pre=np.asarray([[1.0]], dtype=np.float32),
            selected_adv_norm=np.asarray([[2.0]], dtype=np.float32),
            selected_row={"selection_score": 1.5},
        )
        with patch("rdsynth.pipeline.stage2_execution.auto_select_stage2_candidates", return_value=selection_result):
            selection = run_stage2_candidate_selection(
                pareto_cfg={},
                eval_helper=object(),
                sample_with_alpha=lambda x, alpha: x,
                x_mal_eval=np.asarray([[0.0]], dtype=np.float32),
                x_ben_norm=np.asarray([[0.0]], dtype=np.float32),
                attack_query_oracle=QueryOracle(lambda x: np.zeros((len(x),), dtype=np.float64)),
                metrics_payload=metrics_payload,
                default_alpha=0.1,
            )

        self.assertEqual(selection.selected_alpha, 0.3)
        self.assertIn("candidate_selection_time_sec", metrics_payload)

    def test_execute_stage2_sample_uses_cached_selected_samples(self) -> None:
        metrics_payload: dict[str, object] = {}
        selection = Stage2ExecutionSelection(
            selected_alpha=0.2,
            selected_adv_pre=np.asarray([[1.0, 2.0]], dtype=np.float32),
            selected_adv_norm=np.asarray([[3.0, 4.0]], dtype=np.float32),
            selected_row={"selection_score": 2.0},
            selection_runtime_sec=0.1,
        )
        query_oracle = QueryOracle(lambda x: np.zeros((len(x),), dtype=np.float64))
        eval_helper = Mock()

        result = execute_stage2_sample(
            sample_with_alpha=lambda x, alpha: np.asarray([[9.0, 9.0]], dtype=np.float32),
            eval_helper=eval_helper,
            attack_query_oracle=query_oracle,
            metrics_payload=metrics_payload,
            x_mal_eval=np.asarray([[0.0, 0.0]], dtype=np.float32),
            selection=selection,
            warn_fn=lambda _: None,
        )

        np.testing.assert_allclose(result.x_adv_pre, selection.selected_adv_pre)
        np.testing.assert_allclose(result.x_adv_norm, selection.selected_adv_norm)
        self.assertEqual(metrics_payload["selected_candidate_selection_score"], 2.0)
        self.assertEqual(metrics_payload["attack_score_query_count"], 0)
        eval_helper.postprocess_adv.assert_not_called()

    def test_execute_stage2_sample_postprocesses_generated_samples(self) -> None:
        metrics_payload: dict[str, object] = {}
        selection = Stage2ExecutionSelection(
            selected_alpha=0.4,
            selected_adv_pre=None,
            selected_adv_norm=None,
            selected_row=None,
            selection_runtime_sec=0.1,
        )
        query_oracle = QueryOracle(lambda x: np.ones((len(x),), dtype=np.float64))
        eval_helper = Mock()
        eval_helper.postprocess_adv.return_value = (
            np.asarray([[5.0, 6.0]], dtype=np.float32),
            np.asarray([[7.0, 8.0]], dtype=np.float32),
        )

        result = execute_stage2_sample(
            sample_with_alpha=lambda x, alpha: np.asarray([[1.0, 2.0]], dtype=np.float32),
            eval_helper=eval_helper,
            attack_query_oracle=query_oracle,
            metrics_payload=metrics_payload,
            x_mal_eval=np.asarray([[0.0, 0.0]], dtype=np.float32),
            selection=selection,
            warn_fn=lambda _: None,
        )

        np.testing.assert_allclose(result.x_adv_pre, np.asarray([[5.0, 6.0]], dtype=np.float32))
        np.testing.assert_allclose(result.x_adv_norm, np.asarray([[7.0, 8.0]], dtype=np.float32))
        self.assertIn("sample_nan_rate", metrics_payload)


if __name__ == "__main__":
    unittest.main()
