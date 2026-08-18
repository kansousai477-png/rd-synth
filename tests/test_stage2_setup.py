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

from rdsynth.pipeline.stage2_inputs import Stage2ConstraintInputs, Stage2EvalInputs
from rdsynth.pipeline.stage2_setup import build_stage2_eval_helper, build_stage2_predictor_setup


class Stage2SetupTest(unittest.TestCase):
    def test_build_stage2_predictor_setup_updates_metrics_and_budget(self) -> None:
        metrics_payload: dict[str, object] = {}
        predictors = Mock()
        predictors.attack_score_fn = Mock()
        with patch("rdsynth.pipeline.stage2_setup.make_stage2_predictors", return_value=predictors):
            setup = build_stage2_predictor_setup(
                stage2_cfg={"blackbox_eval": {"query_budget": 12, "hard_label_queries": True}},
                surrogate=object(),
                oracle=object(),
                device="cpu",
                batch_size=32,
                metrics_payload=metrics_payload,
            )

        self.assertIs(setup.predictors, predictors)
        self.assertEqual(setup.attack_query_oracle.max_queries, 12)
        self.assertTrue(metrics_payload["attack_score_hard_label_queries"])
        self.assertEqual(metrics_payload["attack_score_query_budget"], 12)

    def test_build_stage2_eval_helper_passes_structured_inputs(self) -> None:
        eval_inputs = Stage2EvalInputs(
            x_mal_eval=np.zeros((1, 2), dtype=np.float32),
            x_ben_eval=np.zeros((1, 2), dtype=np.float32),
            denorm_mean=np.zeros(2, dtype=np.float32),
            denorm_std=np.ones(2, dtype=np.float32),
            x_ben_norm=np.zeros((1, 2), dtype=np.float32),
            x_ben_pre=np.zeros((1, 2), dtype=np.float32),
            x_mal_norm=np.zeros((1, 2), dtype=np.float32),
            x_mal_pre=np.zeros((1, 2), dtype=np.float32),
            x_adv_denorm=None,
            x_ben_denorm=None,
            x_mal_denorm=None,
            eval_denorm=False,
            sample_denorm=False,
            pre_min=None,
            pre_max=None,
            pull_alpha=0.1,
            pull_k=2,
            moment_alpha=0.2,
            moment_std_floor=1.0e-3,
        )
        constraint_inputs = Stage2ConstraintInputs(
            constraints_enabled=True,
            constraints_spec={"ok": True},
            constraints_cfg={"enable": True},
            deploy_enabled=True,
            port_policy="set",
            flag_policy="clip",
            temporal_policy="clip_benign",
            port_allowlist=[80],
            norm_bounds_min=np.zeros(2, dtype=np.float32),
            norm_bounds_max=np.ones(2, dtype=np.float32),
            norm_nonneg=np.zeros(2, dtype=bool),
            x_ben_raw_full=np.zeros((1, 2), dtype=np.float32),
            x_ben_mod_targets=np.zeros((1, 7), dtype=np.float32),
            remap_target_mean=np.zeros(7, dtype=np.float32),
            remap_target_scale=np.ones(7, dtype=np.float32),
        )
        predictors = SimpleNamespace(
            surrogate_predict_probs=lambda x: (np.zeros(len(x), dtype=int), np.zeros((len(x), 2), dtype=np.float32)),
            oracle_predict_probs=lambda x: (np.zeros(len(x), dtype=int), np.zeros((len(x), 2), dtype=np.float32)),
        )

        helper = build_stage2_eval_helper(
            cfg={"stage2": {}},
            seed=7,
            preprocessor=object(),
            bundle_feature_names=["a", "b"],
            diffusion_bundle=SimpleNamespace(ben_stats={}),
            traffic_schema="schema",
            oracle="oracle",
            eval_inputs=eval_inputs,
            constraint_inputs=constraint_inputs,
            predictors=predictors,
        )

        self.assertEqual(helper.pull_alpha, 0.1)
        self.assertEqual(helper.port_allowlist, [80])
        self.assertTrue(helper.constraints_enabled)
        self.assertEqual(helper.traffic_schema, "schema")


if __name__ == "__main__":
    unittest.main()
