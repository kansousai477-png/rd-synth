from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_baselines import _baseline_artifact_meta, run_stage2_baselines


class Stage2BaselinesMoreTest(unittest.TestCase):
    def test_baseline_artifact_meta_handles_builtin_and_paper_baselines(self) -> None:
        family, traffic_space = _baseline_artifact_meta("identity")
        self.assertEqual(family, "control_identity")
        self.assertFalse(traffic_space)

        spec = type("Spec", (), {"family": "paper_attack", "traffic_space": True})()
        with patch("rdsynth.pipeline.stage2_baselines.get_paper_baseline_spec", return_value=spec):
            family2, traffic_space2 = _baseline_artifact_meta("gpmt_lite")
        self.assertEqual(family2, "paper_attack")
        self.assertTrue(traffic_space2)

    def test_run_stage2_baselines_returns_early_when_disabled(self) -> None:
        cfg = {"stage2": {"baselines": {"enable": False}}, "stage3": {}}
        metrics_payload: dict[str, object] = {}
        with patch("builtins.print") as print_mock:
            run_stage2_baselines(
                cfg=cfg,
                bundle_feature_names=["f0"],
                out_dir=ROOT / "outputs",
                device="cpu",
                seed=7,
                surrogate=object(),
                oracle=None,
                y_train=np.asarray([0, 1]),
                x_train=np.asarray([[0.0], [1.0]], dtype=np.float32),
                x_ben_pre=np.asarray([[0.0]], dtype=np.float32),
                x_mal_pre=np.asarray([[1.0]], dtype=np.float32),
                x_ben_norm=np.asarray([[0.0]], dtype=np.float32),
                x_mal_norm=np.asarray([[1.0]], dtype=np.float32),
                denorm_mean=np.asarray([0.0], dtype=np.float32),
                denorm_std=np.asarray([1.0], dtype=np.float32),
                norm_bounds_min=np.asarray([0.0], dtype=np.float32),
                norm_bounds_max=np.asarray([1.0], dtype=np.float32),
                norm_nonneg=np.asarray([False]),
                metrics_payload=metrics_payload,
                sample_denorm=True,
                mal_benign_rate=None,
                mal_benign_rate_oracle=None,
                attack_score_fn=lambda x: np.zeros((len(x),), dtype=np.float64),
                surrogate_predict_probs=lambda x, batch: (
                    np.zeros(len(x), dtype=int),
                    np.zeros((len(x), 2), dtype=np.float32),
                ),
                oracle_predict_probs=lambda x, batch: (np.zeros(len(x), dtype=int), None),
                postprocess_adv=lambda x, x_mal, a, b: (x, x),
            )
        self.assertEqual(metrics_payload, {})
        print_mock.assert_not_called()

    def test_run_stage2_baselines_identity_and_paper_paths_record_metrics(self) -> None:
        cfg = {
            "stage2": {
                "sample_batch_size": 4,
                "metrics_max_real": 8,
                "metrics_max_gen": 8,
                "baselines": {
                    "enable": True,
                    "methods": ["identity", "gpmt_lite"],
                    "eval_metrics": True,
                    "paper_budget_scale": 0.5,
                    "query_budget": 3,
                    "paper_query_budget": 2,
                    "hard_label_queries": True,
                },
            },
            "stage3": {"pcap_compare_baselines": True},
        }
        metrics_payload: dict[str, object] = {}
        spec = type("Spec", (), {"family": "paper_attack", "traffic_space": True})()

        def surrogate_predict_probs(x: np.ndarray, _batch: int):
            return np.zeros(len(x), dtype=int), np.tile(np.asarray([[0.6, 0.4]], dtype=np.float32), (len(x), 1))

        def oracle_predict_probs(x: np.ndarray, _batch: int):
            return np.ones(len(x), dtype=int), np.tile(np.asarray([[0.4, 0.6]], dtype=np.float32), (len(x), 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with (
                patch(
                    "rdsynth.pipeline.stage2_baselines.get_paper_baseline_spec",
                    side_effect=lambda name: spec if name == "gpmt_lite" else None,
                ),
                patch(
                    "rdsynth.pipeline.stage2_baselines.generate_paper_attack_baseline",
                    return_value=np.asarray([[0.2, 0.8]], dtype=np.float32),
                ) as gen_paper,
                patch(
                    "rdsynth.pipeline.stage2_baselines.compute_stage2_metrics",
                    return_value=type(
                        "M",
                        (),
                        {
                            "as_dict": lambda self: {
                                "FFD": 1.0,
                                "SWD": 2.0,
                                "Energy": 3.0,
                                "C2ST-AUC": 0.8,
                                "C2ST-Acc": 0.7,
                            }
                        },
                    )(),
                ),
                patch("rdsynth.pipeline.stage2_baselines.nearest_reference_distance", return_value=1.5),
                patch("rdsynth.pipeline.stage2_baselines.paired_sample_l2", return_value=0.5),
                patch("rdsynth.pipeline.stage2_baselines.np.savez_compressed") as savez,
            ):
                run_stage2_baselines(
                    cfg=cfg,
                    bundle_feature_names=["f0", "f1"],
                    out_dir=out_dir,
                    device="cpu",
                    seed=7,
                    surrogate=object(),
                    oracle=object(),
                    y_train=np.asarray([0, 1], dtype=np.int64),
                    x_train=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                    x_ben_pre=np.asarray([[0.0, 0.0]], dtype=np.float32),
                    x_mal_pre=np.asarray([[1.0, 1.0]], dtype=np.float32),
                    x_ben_norm=np.asarray([[0.0, 0.0]], dtype=np.float32),
                    x_mal_norm=np.asarray([[1.0, 1.0]], dtype=np.float32),
                    denorm_mean=np.asarray([0.0, 0.0], dtype=np.float32),
                    denorm_std=np.asarray([1.0, 1.0], dtype=np.float32),
                    norm_bounds_min=np.asarray([0.0, 0.0], dtype=np.float32),
                    norm_bounds_max=np.asarray([1.0, 1.0], dtype=np.float32),
                    norm_nonneg=np.asarray([False, False]),
                    metrics_payload=metrics_payload,
                    sample_denorm=True,
                    mal_benign_rate=0.1,
                    mal_benign_rate_oracle=0.2,
                    attack_score_fn=lambda x: np.zeros((len(x),), dtype=np.float64),
                    surrogate_predict_probs=surrogate_predict_probs,
                    oracle_predict_probs=oracle_predict_probs,
                    postprocess_adv=lambda x, x_mal, a, b: (x, x),
                )

        self.assertIn("baseline_identity_asr_surrogate", metrics_payload)
        self.assertIn("baseline_gpmt_lite_query_budget", metrics_payload)
        self.assertTrue(metrics_payload["baseline_identity_hard_label_queries"])
        self.assertEqual(gen_paper.call_count, 1)
        self.assertEqual(savez.call_count, 2)


if __name__ == "__main__":
    unittest.main()
