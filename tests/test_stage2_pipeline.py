from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.stage2 import (
    _build_training_metrics_payload,
    _finalize_stage2_outputs,
    _persist_training_log,
    _run_stage2_attack_slice_evals,
    _run_stage2_eval_metrics,
    _schema_counts,
    main,
)
from rdsynth.pipeline.stage2_runtime import Stage2DistributionMetricsResult


class _MetricsStub:
    def __init__(self, payload: dict[str, float]) -> None:
        self._payload = payload

    def as_dict(self) -> dict[str, float]:
        return dict(self._payload)


class Stage2PipelineHelpersTest(unittest.TestCase):
    def test_schema_counts_and_training_payload_helpers(self) -> None:
        traffic_schema = SimpleNamespace(
            port_idx=np.zeros(2, dtype=int),
            flag_idx=np.zeros(3, dtype=int),
            temporal_idx=np.zeros(4, dtype=int),
            ratio_idx=np.zeros(1, dtype=int),
            count_idx=np.zeros(5, dtype=int),
        )
        self.assertEqual(
            _schema_counts(traffic_schema),
            {
                "schema_port_features": 2,
                "schema_flag_features": 3,
                "schema_temporal_features": 4,
                "schema_ratio_features": 1,
                "schema_count_features": 5,
            },
        )

        diffusion_bundle = SimpleNamespace(
            groups={"temporal": [0]}, best_epoch=3, best_score=0.75, train_log=[{"epoch": 1}]
        )
        metrics_payload, train_log = _build_training_metrics_payload(
            diffusion_bundle=diffusion_bundle,
            generator_backbone="ddpm",
            guidance_mode="embedding",
            train_runtime_sec=2.5,
            schema_counts={"schema_port_features": 2},
        )
        self.assertEqual(train_log, [{"epoch": 1}])
        self.assertEqual(metrics_payload["generator_backbone"], "ddpm")
        self.assertEqual(metrics_payload["train_selection_best_epoch"], 3)
        self.assertEqual(metrics_payload["train_selection_best_score"], 0.75)

    def test_persist_training_log_and_finalize_outputs(self) -> None:
        out_dir = ROOT / "outputs" / "stage2_pipeline_test"
        metrics_payload: dict[str, object] = {}

        with patch("rdsynth.pipeline.stage2.save_training_log_csv") as save_log:
            _persist_training_log(out_dir=out_dir, train_log=[{"epoch": 1}], metrics_payload=metrics_payload)
            save_log.assert_called_once()
            self.assertIn("train_selection_log_path", metrics_payload)

        with (
            patch("rdsynth.pipeline.stage2.save_stage2_state") as save_state,
            patch("rdsynth.pipeline.stage2.save_stage2_manifest") as save_manifest,
            patch("builtins.print") as print_mock,
        ):
            _finalize_stage2_outputs(
                config_path="configs/demo.yaml",
                out_dir=out_dir,
                oracle_name="default",
                stage2_mode="editor",
                generator_backbone="ddpm",
                diffusion_bundle=SimpleNamespace(),
                feature_names=["f0", "f1"],
                x_train=np.zeros((4, 2), dtype=np.float32),
                x_ben=np.zeros((2, 2), dtype=np.float32),
                x_mal=np.zeros((2, 2), dtype=np.float32),
                train_log=[{"epoch": 1}],
                metrics_payload={"ok": True},
                settings=SimpleNamespace(),
                x_adv_pre=None,
                x_adv_norm=None,
                x_ben_pre=None,
                x_mal_pre=None,
            )
            save_state.assert_called_once()
            save_manifest.assert_called_once()
            print_mock.assert_called()


class Stage2PipelineEvalTest(unittest.TestCase):
    def test_attack_slice_eval_can_be_disabled_or_limited(self) -> None:
        bundle = SimpleNamespace(
            raw_y_test=np.asarray(["A", "B", "Benign", "C"], dtype=object),
            y_test=np.asarray([1, 1, 0, 1], dtype=np.int64),
        )
        kwargs = dict(
            settings=SimpleNamespace(save_samples=True),
            seed=7,
            device="cpu",
            out_dir=ROOT / "outputs" / "stage2_attack_slice_limit",
            preprocessor=SimpleNamespace(),
            traffic_schema=SimpleNamespace(),
            bundle=bundle,
            x_train=np.zeros((4, 2), dtype=np.float32),
            y_train=np.asarray([0, 1, 0, 1], dtype=np.int64),
            x_ben=np.zeros((2, 2), dtype=np.float32),
            x_mal=np.ones((2, 2), dtype=np.float32),
            diffusion_bundle=SimpleNamespace(),
            surrogate=SimpleNamespace(),
            oracle=SimpleNamespace(),
            train_runtime_sec=0.1,
            stage2_mode="latent_diffusion",
            generator_backbone="ddpm",
            guidance_mode="embedding",
            feature_names=["f0", "f1"],
        )

        disabled = _run_stage2_attack_slice_evals(cfg={"stage2": {"attack_slice_eval_enabled": False}}, **kwargs)
        self.assertEqual(disabled, [])

        with patch(
            "rdsynth.pipeline.stage2._run_stage2_eval_metrics",
            return_value={"x_adv_pre": np.zeros((1, 2), dtype=np.float32)},
        ) as run_eval:
            rows = _run_stage2_attack_slice_evals(
                cfg={"stage2": {"attack_slice_eval_max_labels": 1}},
                **kwargs,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attack_type"], "A")
        run_eval.assert_called_once()

    def test_run_stage2_eval_metrics_executes_pipeline_and_persists_samples(self) -> None:
        cfg = {
            "stage2": {"pareto_eval": {}, "metrics_max_real": 8, "metrics_max_gen": 8},
        }
        settings = SimpleNamespace(
            sample_batch_size=2,
            mal_anchor_alpha=0.3,
            post_clip_norm_range=True,
            save_samples=True,
            save_intermediate_results=True,
        )
        out_dir = ROOT / "outputs" / "stage2_eval_test"
        feature_names = ["f0", "f1"]
        x_train = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        y_train = np.asarray([0, 1], dtype=np.int64)
        x_ben = np.asarray([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        x_mal = np.asarray([[1.0, 1.0], [0.8, 0.7]], dtype=np.float32)
        metrics_payload: dict[str, object] = {}

        predictors = SimpleNamespace(
            attack_score_fn=lambda x: np.zeros((len(x),), dtype=np.float64),
            surrogate_predict_probs=lambda x: (np.zeros(len(x), dtype=int), np.zeros((len(x), 2), dtype=np.float32)),
            oracle_predict_probs=lambda x: (np.zeros(len(x), dtype=int), np.zeros((len(x), 2), dtype=np.float32)),
        )
        predictor_setup = SimpleNamespace(predictors=predictors, attack_query_oracle=Mock())
        eval_inputs = SimpleNamespace(
            x_mal_eval=np.asarray([[0.4, 0.6]], dtype=np.float32),
            x_ben_eval=np.asarray([[0.1, 0.2]], dtype=np.float32),
            denorm_mean=np.zeros(2, dtype=np.float32),
            denorm_std=np.ones(2, dtype=np.float32),
            sample_denorm=True,
            x_ben_norm=np.asarray([[0.1, 0.2]], dtype=np.float32),
            x_ben_pre=np.asarray([[0.1, 0.2]], dtype=np.float32),
            x_mal_norm=np.asarray([[0.4, 0.6]], dtype=np.float32),
            x_mal_pre=np.asarray([[0.4, 0.6]], dtype=np.float32),
            x_adv_denorm=np.asarray([[9.0, 9.0]], dtype=np.float32),
            x_ben_denorm=np.asarray([[1.0, 1.0]], dtype=np.float32),
            x_mal_denorm=np.asarray([[2.0, 2.0]], dtype=np.float32),
            eval_denorm=True,
            pre_min=np.zeros(2, dtype=np.float32),
            pre_max=np.ones(2, dtype=np.float32),
            pull_alpha=0.1,
            pull_k=2,
            moment_alpha=0.2,
            moment_std_floor=1.0e-3,
        )
        constraint_inputs = SimpleNamespace(
            norm_bounds_min=np.zeros(2, dtype=np.float32),
            norm_bounds_max=np.ones(2, dtype=np.float32),
            norm_nonneg=np.zeros(2, dtype=bool),
        )
        selection = SimpleNamespace(selection_runtime_sec=0.25)
        sample_execution = SimpleNamespace(
            x_adv_pre=np.asarray([[1.5, -1.0]], dtype=np.float32),
            x_adv_norm=np.asarray([[0.6, 0.7]], dtype=np.float32),
            sample_runtime_sec=0.5,
            query_stats=SimpleNamespace(query_count=2),
        )
        eval_helper = SimpleNamespace(postprocess_adv=Mock())
        diffusion_bundle = SimpleNamespace(ben_stats={"min": np.zeros(2), "max": np.ones(2)})
        metrics_result = Stage2DistributionMetricsResult(
            metrics_norm=_MetricsStub({"FFD": 1.0}),
            adv_ben_l2=1.5,
            adv_mal_l2=0.5,
            metrics_denorm=_MetricsStub({"FFD": 2.0}),
            adv_ben_l2_denorm=3.0,
            adv_mal_l2_denorm=4.0,
        )

        def _update_attack_metrics(**kwargs) -> None:
            payload = kwargs["metrics_payload"]
            payload["asr_oracle"] = 1.0
            payload["asr_surrogate"] = 0.5
            payload["mal_benign_rate"] = 0.0
            payload["mal_benign_rate_oracle"] = 0.0

        with (
            patch("rdsynth.pipeline.stage2.build_stage2_predictor_setup", return_value=predictor_setup),
            patch("rdsynth.pipeline.stage2.prepare_stage2_eval_inputs", return_value=eval_inputs),
            patch("rdsynth.pipeline.stage2.prepare_stage2_constraint_inputs", return_value=constraint_inputs),
            patch("rdsynth.pipeline.stage2.build_stage2_sampler", return_value="sampler"),
            patch("rdsynth.pipeline.stage2.build_stage2_eval_helper", return_value=eval_helper),
            patch("rdsynth.pipeline.stage2.run_stage2_candidate_selection", return_value=selection),
            patch("rdsynth.pipeline.stage2.execute_stage2_sample", return_value=sample_execution),
            patch("rdsynth.pipeline.stage2.record_sample_runtime") as record_runtime,
            patch("rdsynth.pipeline.stage2.update_attack_metrics", side_effect=_update_attack_metrics),
            patch("rdsynth.pipeline.stage2.update_sample_distribution_summary") as update_summary,
            patch("rdsynth.pipeline.stage2.compute_stage2_distribution_metrics", return_value=metrics_result),
            patch("rdsynth.pipeline.stage2.print_stage2_metric_tables") as print_tables,
            patch("rdsynth.pipeline.stage2.run_stage2_baselines") as run_baselines,
            patch("rdsynth.pipeline.stage2.run_stage2_pareto_eval") as run_pareto,
            patch("rdsynth.pipeline.stage2.build_stage2_artifact_payload", return_value={"adv": np.asarray([1])}),
            patch("rdsynth.pipeline.stage2.persist_stage2_metrics") as persist_metrics,
            patch("rdsynth.pipeline.stage2.np.savez_compressed") as savez,
        ):
            preprocessor = SimpleNamespace(
                inverse_transform=Mock(return_value=np.asarray([[0.9, 1.0]], dtype=np.float32))
            )
            result = _run_stage2_eval_metrics(
                cfg=cfg,
                stage2_cfg=cfg["stage2"],
                settings=settings,
                seed=7,
                device="cpu",
                out_dir=out_dir,
                preprocessor=preprocessor,
                traffic_schema="schema",
                x_train=x_train,
                y_train=y_train,
                x_ben=x_ben,
                x_mal=x_mal,
                diffusion_bundle=diffusion_bundle,
                surrogate="surrogate",
                oracle="oracle",
                metrics_payload=metrics_payload,
                train_runtime_sec=1.25,
                stage2_mode="editor",
                generator_backbone="ddpm",
                guidance_mode="embedding",
                feature_names=feature_names,
            )

        self.assertEqual(result["x_adv_norm"].shape, (1, 2))
        self.assertEqual(metrics_payload["sample_end_to_end_time_sec"], 2.0)
        self.assertEqual(metrics_payload["attack_score_queries_per_success_oracle"], 2.0)
        self.assertEqual(metrics_payload["attack_score_queries_per_success_surrogate"], 4.0)
        record_runtime.assert_called_once()
        update_summary.assert_called_once()
        print_tables.assert_called_once()
        run_baselines.assert_called_once()
        run_pareto.assert_called_once()
        persist_metrics.assert_called_once()
        self.assertEqual(savez.call_count, 2)


class Stage2MainTest(unittest.TestCase):
    def test_main_skips_eval_and_warns_when_stage1_optional_checkpoint_missing(self) -> None:
        runtime = SimpleNamespace(
            cfg={"stage2": {"raw": True}},
            seed=7,
            device="cpu",
            out_dir=ROOT / "outputs" / "stage2_main_skip",
            stage_cfg={"dummy": True},
            config_path=ROOT / "configs" / "demo.yaml",
        )
        bundle = SimpleNamespace(
            x_train=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            y_train=np.asarray([0, 1], dtype=np.int64),
            x_val=np.asarray([[0.2, 0.2]], dtype=np.float32),
            y_val=np.asarray([0], dtype=np.int64),
            feature_names=["f0", "f1"],
        )
        settings = SimpleNamespace(
            oracle_name="default",
            require_stage1=False,
            mode="editor",
            generator_backbone="ddpm",
            guidance_mode="embedding",
            eval_metrics=False,
        )
        artifacts = SimpleNamespace(
            surrogate="surrogate", oracle="oracle", checkpoint_path=SimpleNamespace(exists=lambda: False)
        )
        diffusion_bundle = SimpleNamespace(groups={}, train_log=None)

        with (
            patch("rdsynth.pipeline.stage2.load_stage_runtime", return_value=runtime),
            patch("rdsynth.pipeline.stage2.Stage2Settings.from_cfg", return_value=settings),
            patch("rdsynth.pipeline.stage2.load_data_context", return_value=SimpleNamespace(bundle=bundle)),
            patch(
                "rdsynth.pipeline.stage2.DatasetPreprocessor.from_bundle",
                return_value=SimpleNamespace(inverse_transform=Mock(return_value=bundle.x_train)),
            ),
            patch(
                "rdsynth.pipeline.stage2.infer_traffic_feature_schema",
                return_value=SimpleNamespace(
                    port_idx=np.array([]),
                    flag_idx=np.array([]),
                    temporal_idx=np.array([]),
                    ratio_idx=np.array([]),
                    count_idx=np.array([]),
                ),
            ),
            patch("rdsynth.pipeline.stage2.load_stage2_artifacts", return_value=artifacts),
            patch("rdsynth.pipeline.stage2.train_stage2_generator", return_value=diffusion_bundle),
            patch("rdsynth.pipeline.stage2.save_config") as save_config,
            patch("rdsynth.pipeline.stage2._build_training_metrics_payload", return_value=({"m": 1}, None)),
            patch("rdsynth.pipeline.stage2._persist_training_log") as persist_log,
            patch("rdsynth.pipeline.stage2._run_stage2_eval_metrics") as run_eval,
            patch("rdsynth.pipeline.stage2._finalize_stage2_outputs") as finalize,
            patch("rdsynth.pipeline.stage2.time.perf_counter", side_effect=[10.0, 12.0]),
            patch("builtins.print") as print_mock,
        ):
            main("configs/demo.yaml")

        save_config.assert_called_once()
        persist_log.assert_called_once()
        run_eval.assert_not_called()
        finalize.assert_called_once()
        print_mock.assert_any_call("[Stage2][Warn] Stage1 checkpoint missing; using an untrained surrogate.")

    def test_main_runs_eval_path_and_forwards_eval_arrays(self) -> None:
        runtime = SimpleNamespace(
            cfg={"stage2": {"raw": True}},
            seed=7,
            device="cpu",
            out_dir=ROOT / "outputs" / "stage2_main_eval",
            stage_cfg={"dummy": True},
            config_path=ROOT / "configs" / "demo.yaml",
        )
        bundle = SimpleNamespace(
            x_train=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            y_train=np.asarray([0, 1], dtype=np.int64),
            x_val=np.asarray([[0.2, 0.2]], dtype=np.float32),
            y_val=np.asarray([0], dtype=np.int64),
            feature_names=["f0", "f1"],
        )
        settings = SimpleNamespace(
            oracle_name="default",
            require_stage1=True,
            mode="latent_diffusion",
            generator_backbone="ddpm",
            guidance_mode="embedding",
            eval_metrics=True,
        )
        artifacts = SimpleNamespace(
            surrogate="surrogate", oracle="oracle", checkpoint_path=SimpleNamespace(exists=lambda: True)
        )
        diffusion_bundle = SimpleNamespace(groups={}, train_log=[{"epoch": 1}])
        eval_result = {
            "x_adv_pre": np.asarray([[1.0]], dtype=np.float32),
            "x_adv_norm": np.asarray([[2.0]], dtype=np.float32),
            "x_ben_pre": np.asarray([[3.0]], dtype=np.float32),
            "x_mal_pre": np.asarray([[4.0]], dtype=np.float32),
        }

        with (
            patch("rdsynth.pipeline.stage2.load_stage_runtime", return_value=runtime),
            patch("rdsynth.pipeline.stage2.Stage2Settings.from_cfg", return_value=settings),
            patch("rdsynth.pipeline.stage2.load_data_context", return_value=SimpleNamespace(bundle=bundle)),
            patch(
                "rdsynth.pipeline.stage2.DatasetPreprocessor.from_bundle",
                return_value=SimpleNamespace(inverse_transform=Mock(return_value=bundle.x_train)),
            ),
            patch(
                "rdsynth.pipeline.stage2.infer_traffic_feature_schema",
                return_value=SimpleNamespace(
                    port_idx=np.array([]),
                    flag_idx=np.array([]),
                    temporal_idx=np.array([]),
                    ratio_idx=np.array([]),
                    count_idx=np.array([]),
                ),
            ),
            patch("rdsynth.pipeline.stage2.load_stage2_artifacts", return_value=artifacts),
            patch("rdsynth.pipeline.stage2.train_stage2_generator", return_value=diffusion_bundle),
            patch("rdsynth.pipeline.stage2.save_config"),
            patch("rdsynth.pipeline.stage2._build_training_metrics_payload", return_value=({"m": 1}, [{"epoch": 1}])),
            patch("rdsynth.pipeline.stage2._persist_training_log"),
            patch("rdsynth.pipeline.stage2._run_stage2_eval_metrics", return_value=eval_result) as run_eval,
            patch("rdsynth.pipeline.stage2._finalize_stage2_outputs") as finalize,
            patch("rdsynth.pipeline.stage2.time.perf_counter", side_effect=[20.0, 21.0]),
        ):
            main("configs/demo.yaml")

        run_eval.assert_called_once()
        finalize.assert_called_once()
        finalize_kwargs = finalize.call_args.kwargs
        self.assertEqual(finalize_kwargs["x_adv_pre"].tolist(), [[1.0]])
        self.assertEqual(finalize_kwargs["x_mal_pre"].tolist(), [[4.0]])


if __name__ == "__main__":
    unittest.main()
