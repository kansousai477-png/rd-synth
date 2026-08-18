from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_runtime import (
    _invoke_predictor,
    persist_stage2_metrics,
    run_stage2_pareto_eval,
    update_attack_metrics,
    update_sample_distribution_summary,
)


class Stage2RuntimeMoreTest(unittest.TestCase):
    def test_invoke_predictor_supports_both_signatures(self) -> None:
        x = np.asarray([[1.0], [2.0]], dtype=np.float32)

        def with_batch(arr: np.ndarray, batch_size: int):
            return np.zeros(len(arr), dtype=int), np.ones((len(arr), 2), dtype=np.float32) * batch_size

        def without_batch(arr: np.ndarray):
            return np.ones(len(arr), dtype=int), None

        preds, probs = _invoke_predictor(with_batch, x, 3)
        self.assertEqual(preds.tolist(), [0, 0])
        self.assertEqual(probs.shape, (2, 2))

        preds2, probs2 = _invoke_predictor(without_batch, x, 3)
        self.assertEqual(preds2.tolist(), [1, 1])
        self.assertIsNone(probs2)

    def test_update_attack_metrics_handles_surrogate_and_oracle_paths(self) -> None:
        metrics_payload: dict[str, object] = {}
        x_adv_pre = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        x_mal_pre = np.asarray([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)

        def surrogate_predict_probs(arr: np.ndarray, _batch: int):
            preds = np.asarray([0, 1], dtype=int)
            probs = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
            return preds[: len(arr)], probs[: len(arr)]

        def oracle_predict_probs(arr: np.ndarray, _batch: int):
            preds = np.asarray([0, 0], dtype=int)
            probs = np.asarray([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32)
            return preds[: len(arr)], probs[: len(arr)]

        update_attack_metrics(
            metrics_payload=metrics_payload,
            surrogate=object(),
            oracle=object(),
            surrogate_predict_probs=surrogate_predict_probs,
            oracle_predict_probs=oracle_predict_probs,
            x_adv_pre=x_adv_pre,
            x_mal_pre=x_mal_pre,
            batch_size=16,
            sample_runtime_sec=0.25,
        )

        self.assertEqual(metrics_payload["asr_surrogate"], 0.5)
        self.assertEqual(metrics_payload["asr_oracle"], 1.0)
        self.assertIn("surrogate_paper_attack_success_rate", metrics_payload)
        self.assertIn("oracle_paper_attack_success_rate", metrics_payload)

    def test_update_sample_distribution_summary_records_range_and_iat_metrics(self) -> None:
        metrics_payload: dict[str, object] = {}
        x_adv_norm = np.asarray([[2.0, 0.2], [0.0, 0.4]], dtype=np.float32)
        x_ben_norm = np.asarray([[0.5, 0.0], [0.3, 0.1]], dtype=np.float32)
        x_mal_norm = np.asarray([[1.0, 0.5], [0.8, 0.4]], dtype=np.float32)

        with patch("builtins.print") as print_mock:
            update_sample_distribution_summary(
                metrics_payload=metrics_payload,
                feature_names=["Flow IAT Mean", "Other"],
                x_adv_norm=x_adv_norm,
                x_ben_norm=x_ben_norm,
                x_mal_norm=x_mal_norm,
                lo=np.zeros(2, dtype=np.float32),
                hi=np.ones(2, dtype=np.float32),
            )

        self.assertGreater(metrics_payload["sample_range_violation_rate"], 0.0)
        self.assertIn("iat_adv_ben_mean_abs", metrics_payload)
        self.assertTrue(print_mock.called)

    def test_run_stage2_pareto_eval_generates_curve_metrics_when_enabled(self) -> None:
        cfg = {"stage2": {"pareto_eval": {"enable": True, "anchor_grid": [0.0, 0.5], "max_samples": 2}}}
        metrics_payload: dict[str, object] = {}
        eval_rows: list[dict[str, float]] = []

        def sample_with_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
            return x + alpha

        def save_pareto_front(path: Path, rows: list[dict[str, float]]) -> Path:
            eval_rows.extend(rows)
            return path

        eval_helper = SimpleNamespace(
            evaluate_candidate=lambda adv, mal, x_ben_norm_ref: (
                adv,
                mal,
                {
                    "asr_surrogate": 0.5,
                    "adv_to_mal_l2": float(np.mean(adv)),
                    "query_count": 2.0,
                },
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch("rdsynth.pipeline.stage2_runtime.save_records_csv") as save_records:
                run_stage2_pareto_eval(
                    cfg=cfg,
                    seed=7,
                    out_dir=out_dir,
                    x_mal_eval=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    x_ben_eval=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                    denorm_mean=np.zeros(2, dtype=np.float32),
                    denorm_std=np.ones(2, dtype=np.float32),
                    eval_helper=eval_helper,
                    sample_with_alpha=sample_with_alpha,
                    metrics_payload=metrics_payload,
                    save_pareto_front=save_pareto_front,
                )

        self.assertEqual(len(eval_rows), 2)
        self.assertIn("pareto_auc_asr_vs_distortion", metrics_payload)
        self.assertIn("pareto_tradeoff_curve_path", metrics_payload)
        save_records.assert_called_once()

    def test_persist_stage2_metrics_delegates_to_artifact_writers(self) -> None:
        metrics_payload = {"a": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with (
                patch("rdsynth.pipeline.stage2_runtime.save_metrics") as save_metrics,
                patch("rdsynth.pipeline.stage2_runtime.save_metrics_csv") as save_metrics_csv,
            ):
                persist_stage2_metrics(metrics_payload, out_dir)

        save_metrics.assert_called_once_with(metrics_payload, out_dir)
        save_metrics_csv.assert_called_once_with(metrics_payload, out_dir)


if __name__ == "__main__":
    unittest.main()
