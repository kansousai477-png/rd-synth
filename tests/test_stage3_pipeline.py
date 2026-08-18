from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.stage3 import _pcap_feature_quality_block_reason, _source_slug, main


class Stage3PipelineTest(unittest.TestCase):
    def test_source_slug_is_short_and_stable_for_long_pcap_names(self) -> None:
        long_path = Path("C:/data") / ("very_long_attack_capture_name_" * 8 + ".pcap")
        slug = _source_slug(long_path, 3)

        self.assertLessEqual(len(slug), 48)
        self.assertRegex(slug, r"^source_03_[0-9a-f]{10}_")
        self.assertEqual(slug, _source_slug(long_path, 3))

    def test_pcap_feature_quality_block_reason_prefers_fallback_then_fill_then_status(self) -> None:
        self.assertEqual(_pcap_feature_quality_block_reason({"pcap_feature_fallback_count": 1}), "")
        self.assertEqual(_pcap_feature_quality_block_reason({"pcap_feature_fill_count": 2}), "fill_value_features_used")
        self.assertEqual(
            _pcap_feature_quality_block_reason({"pcap_feature_statuses": ["ok", "partial"]}),
            "feature_status_partial",
        )
        self.assertEqual(_pcap_feature_quality_block_reason({}), "")

    def test_stage3_main_handles_no_adv_samples_and_saves_remap_only_summary(self) -> None:
        runtime = SimpleNamespace(
            cfg={
                "project": {"out_dir": str(ROOT / "outputs" / "stage3_project")},
                "stage1": {"sur_hidden": [8]},
                "stage2": {"oracle_name": "mlp_small"},
            },
            seed=7,
            device="cpu",
            out_dir=ROOT / "outputs" / "stage3_main_test",
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
            remap_train_source="all",
            remap_train_objective="identity",
            loss="huber",
            huber_beta=1.0,
            grad_clip=1.0,
            target_clip_sigma=5.0,
            weight_decay=1.0e-4,
            remap_mode="direct",
            remap_min_r2=0.0,
            remap_blend_alpha=0.7,
            remap_collapse_ratio_threshold=0.25,
            epochs=2,
            batch_size=4,
            lr=1.0e-3,
            protocol_auto_fix=True,
            feature_aliases_path="",
            feature_backend="auto",
            cicflowmeter_cmd="java -jar CICFlowMeter.jar",
            cicflowmeter_timeout=300,
            oracle_name="mlp_small",
            pcap_eval_use_oracle=False,
            pcap_align_min_coverage=0.85,
            pcap_eval_batch_size=128,
            pcap_cache_enable=False,
            pcap_cache_dir="",
            pcap_feature_fail_closed=False,
            pcap_feature_fail_on_partial_alignment=False,
            pcap_search_alphas=[1.0],
            pcap_path="",
            pcap_scan_dir="",
            pcap_scan_limit=0,
            pcap_scan_min_prob=0.5,
            pcap_scan_glob="*.pcap",
            pcap_scan_compare_existing=True,
            pcap_scan_pmal_weight=0.7,
            pcap_scan_target_fit_weight=0.2,
            pcap_scan_target_mod_fit_weight=0.1,
            pcap_source_selection_mode="best",
            pcap_source_sample_n=1,
            pcap_source_sample_seed=0,
            adv_samples_path="",
            copy_adv_samples=False,
            pcap_apply_fields=[],
            pcap_search_bidirectional=True,
            pcap_search_field_subsets=False,
            pcap_search_probe_topk=2,
            pcap_search_rounds=1,
            pcap_tcp_fixup=True,
            pcap_dst_port_policy="keep",
            pcap_dst_port_allowlist=[],
            pcap_out_dir="",
            pcap_apply_n=1,
            pcap_eval=False,
            pcap_compare_baselines=False,
            pcap_baseline_jobs=1,
            save_intermediate_results=False,
        )
        pcap_features = SimpleNamespace(metrics_snapshot=lambda: {"pcap_feature_statuses": ["ok"]})
        pcap_selection = SimpleNamespace(
            selected_path=None,
            selected_source="",
            evasion_valid=None,
            scan_min_prob=0.5,
            source_selection_mode="best",
            candidate_paths=[],
        )

        with (
            patch("rdsynth.pipeline.stage3.load_stage_runtime", return_value=runtime),
            patch("rdsynth.pipeline.stage3.Stage3Settings.from_cfg", return_value=settings),
            patch(
                "rdsynth.pipeline.stage3.detect_stage3_environment",
                return_value=SimpleNamespace(
                    scapy_available=False, nfstream_available=False, cicflowmeter_available=False
                ),
            ),
            patch("rdsynth.pipeline.stage3.load_data_context", return_value=SimpleNamespace(bundle=bundle)),
            patch(
                "rdsynth.pipeline.stage3.DatasetPreprocessor.from_bundle",
                return_value=SimpleNamespace(
                    inverse_transform=Mock(side_effect=lambda x: x),
                    feature_mean=Mock(return_value=np.zeros(2, dtype=np.float32)),
                ),
            ),
            patch("rdsynth.pipeline.stage3.validate_remap_mode"),
            patch("rdsynth.pipeline.stage3.load_feature_aliases", return_value={}),
            patch(
                "rdsynth.pipeline.stage3.load_stage3_artifacts",
                return_value=SimpleNamespace(
                    surrogate=None, oracle=None, checkpoint_path=SimpleNamespace(exists=lambda: False)
                ),
            ),
            patch(
                "rdsynth.pipeline.stage3.resolve_pcap_eval_model",
                return_value=SimpleNamespace(pcap_eval_model=None, pcap_eval_model_name="none"),
            ),
            patch("rdsynth.pipeline.stage3.PcapFeatureExtractor", return_value=pcap_features),
            patch("rdsynth.pipeline.stage3.build_pcap_selection", return_value=pcap_selection),
            patch(
                "rdsynth.pipeline.stage3.resolve_adv_samples_path", return_value=ROOT / "outputs" / "missing_adv.npz"
            ),
            patch("rdsynth.pipeline.stage3.resolve_selected_pcap", return_value=pcap_selection),
            patch("rdsynth.pipeline.stage3.save_metrics") as save_metrics,
            patch("rdsynth.pipeline.stage3.save_metrics_csv") as save_metrics_csv,
            patch("rdsynth.pipeline.stage3.build_stage_output_files", return_value={"state": "none"}) as build_outputs,
            patch("rdsynth.pipeline.stage3.save_stage_manifest_spec") as save_manifest,
            patch("rdsynth.pipeline.stage3.save_config") as save_config,
            patch("builtins.print") as print_mock,
        ):
            main("configs/demo.yaml")

        save_config.assert_called_once()
        save_metrics.assert_called_once()
        save_metrics_csv.assert_called_once()
        build_outputs.assert_called_once()
        save_manifest.assert_called_once()
        manifest_spec = save_manifest.call_args.args[0]
        self.assertEqual(manifest_spec.metrics["adv_samples_count"], 0)
        self.assertTrue(any("[Stage3] summary" in str(call.args[0]) for call in print_mock.call_args_list))

    def test_stage3_main_processes_adv_and_search_writes_pcaps(self) -> None:
        runtime = SimpleNamespace(
            cfg={
                "project": {"out_dir": str(ROOT / "outputs" / "stage3_project2")},
                "stage1": {"sur_hidden": [8]},
                "stage2": {"oracle_name": "mlp_small"},
            },
            seed=7,
            device="cpu",
            out_dir=ROOT / "outputs" / "stage3_main_test2",
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
            remap_train_source="all",
            remap_train_objective="identity",
            loss="huber",
            huber_beta=1.0,
            grad_clip=1.0,
            target_clip_sigma=5.0,
            weight_decay=1.0e-4,
            remap_mode="direct",
            remap_min_r2=0.0,
            remap_blend_alpha=0.7,
            remap_collapse_ratio_threshold=0.25,
            epochs=2,
            batch_size=4,
            lr=1.0e-3,
            protocol_auto_fix=True,
            feature_aliases_path="",
            feature_backend="auto",
            cicflowmeter_cmd="java -jar CICFlowMeter.jar",
            cicflowmeter_timeout=300,
            oracle_name="mlp_small",
            pcap_eval_use_oracle=False,
            pcap_align_min_coverage=0.85,
            pcap_eval_batch_size=128,
            pcap_cache_enable=False,
            pcap_cache_dir="",
            pcap_feature_fail_closed=False,
            pcap_feature_fail_on_partial_alignment=False,
            pcap_search_alphas=[1.0],
            pcap_path="orig.pcap",
            pcap_scan_dir="",
            pcap_scan_limit=0,
            pcap_scan_min_prob=0.5,
            pcap_scan_glob="*.pcap",
            pcap_scan_compare_existing=True,
            pcap_scan_pmal_weight=0.7,
            pcap_scan_target_fit_weight=0.2,
            pcap_scan_target_mod_fit_weight=0.1,
            pcap_source_selection_mode="best",
            pcap_source_sample_n=1,
            pcap_source_sample_seed=0,
            adv_samples_path="adv_samples.npz",
            copy_adv_samples=False,
            pcap_apply_fields=[],
            pcap_search_bidirectional=True,
            pcap_search_field_subsets=False,
            pcap_search_probe_topk=2,
            pcap_search_rounds=1,
            pcap_tcp_fixup=True,
            pcap_dst_port_policy="keep",
            pcap_dst_port_allowlist=[],
            pcap_out_dir="",
            pcap_apply_n=1,
            pcap_eval=False,
            pcap_compare_baselines=False,
            pcap_baseline_jobs=1,
            save_intermediate_results=False,
        )
        pcap_features = SimpleNamespace(metrics_snapshot=lambda: {"pcap_feature_statuses": ["ok"]})
        selected_path = ROOT / "outputs" / "selected.pcap"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_bytes(b"pcap")
        selection = SimpleNamespace(
            selected_path=selected_path,
            selected_source="given",
            evasion_valid=True,
            scan_min_prob=0.5,
            source_selection_mode="best",
            candidate_paths=[selected_path],
        )
        fake_search_result = SimpleNamespace(
            mods=np.asarray([[0.2, 0.3]], dtype=np.float32),
            pcap_written_count=1,
            pcap_kept_original_count=0,
            pcap_apply_time_sec=0.5,
            pcap_packet_count=2,
            pcap_pcaps_per_sec=2.0,
            pcap_packet_throughput_pps=10.0,
            pcap_selected_alpha_mean=0.25,
            pcap_selected_alphas=[0.25],
            pcap_selected_field_sets=["payload_scale"],
            pcap_selected_deployability_score_mean=0.7,
            pcap_selected_response_l2_mean=0.2,
            pcap_modified=True,
            pcap_out_dir="pcaps",
        )

        with ExitStack() as stack:
            stack.enter_context(patch("rdsynth.pipeline.stage3.load_stage_runtime", return_value=runtime))
            stack.enter_context(patch("rdsynth.pipeline.stage3.Stage3Settings.from_cfg", return_value=settings))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.detect_stage3_environment",
                    return_value=SimpleNamespace(
                        scapy_available=True, nfstream_available=False, cicflowmeter_available=False
                    ),
                )
            )
            stack.enter_context(
                patch("rdsynth.pipeline.stage3.load_data_context", return_value=SimpleNamespace(bundle=bundle))
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.DatasetPreprocessor.from_bundle",
                    return_value=SimpleNamespace(
                        inverse_transform=Mock(side_effect=lambda x: x),
                        feature_mean=Mock(return_value=np.zeros(2, dtype=np.float32)),
                    ),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.validate_remap_mode"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.load_feature_aliases", return_value={}))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.load_stage3_artifacts",
                    return_value=SimpleNamespace(
                        surrogate=None,
                        oracle=None,
                        checkpoint_path=SimpleNamespace(exists=lambda: False),
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.resolve_pcap_eval_model",
                    return_value=SimpleNamespace(pcap_eval_model=object(), pcap_eval_model_name="surrogate"),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.PcapFeatureExtractor", return_value=pcap_features))
            stack.enter_context(patch("rdsynth.pipeline.stage3.build_pcap_selection", return_value=selection))
            stack.enter_context(patch("rdsynth.pipeline.stage3.resolve_adv_samples_path", return_value=Path(__file__)))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.load_adv_samples",
                    return_value=SimpleNamespace(
                        adv=np.asarray([[0.1, 0.2]], dtype=np.float32),
                        adv_norm=None,
                        adv_mean=None,
                        adv_std=None,
                        loaded=True,
                        count=1,
                        adv_space="pre",
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.build_rule_based_modifications",
                    return_value=np.asarray([[0.3, 0.4]], dtype=np.float32),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.clip_modifications", side_effect=lambda x: x))
            stack.enter_context(patch("rdsynth.pipeline.stage3.resolve_selected_pcap", return_value=selection))
            stack.enter_context(patch("rdsynth.pipeline.stage3.record_pcap_apply_settings"))
            stack.enter_context(
                patch("rdsynth.pipeline.stage3._pcap_output_dir", return_value=ROOT / "outputs" / "pcaps_out")
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.prepare_pcap_search_context",
                    return_value=SimpleNamespace(
                        pcap_target_mod=np.asarray([0.3, 0.4], dtype=np.float32),
                        orig_pmal_for_selection=0.8,
                        orig_feat_pre_mean=np.asarray([0.0, 0.0], dtype=np.float32),
                        target_metric_fn=lambda path, target: (0.1, 0.2, 0.1, 0.2, {}),
                    ),
                )
            )
            search_and_write = stack.enter_context(
                patch("rdsynth.pipeline.stage3.search_and_write_pcaps", return_value=fake_search_result)
            )
            save_metrics = stack.enter_context(patch("rdsynth.pipeline.stage3.save_metrics"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.save_metrics_csv"))
            stack.enter_context(
                patch("rdsynth.pipeline.stage3.build_stage_output_files", return_value={"state": "none"})
            )
            save_manifest = stack.enter_context(patch("rdsynth.pipeline.stage3.save_stage_manifest_spec"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.save_config"))
            stack.enter_context(patch("builtins.print"))
            stack.enter_context(
                patch.dict(
                    "sys.modules",
                    {
                        "scapy": SimpleNamespace(),
                        "scapy.all": SimpleNamespace(rdpcap=lambda path: [1, 2], wrpcap=lambda path, pkts: None),
                    },
                )
            )
            main("configs/demo.yaml")

        self.assertGreaterEqual(search_and_write.call_count, 1)
        save_metrics.assert_called_once()
        manifest_spec = save_manifest.call_args.args[0]
        self.assertEqual(manifest_spec.metrics["adv_samples_count"], 1)
        self.assertTrue(manifest_spec.metrics["pcap_modified"])

    def test_stage3_main_random_source_mode_processes_multiple_pcaps(self) -> None:
        runtime = SimpleNamespace(
            cfg={
                "project": {"out_dir": str(ROOT / "outputs" / "stage3_project3")},
                "stage1": {"sur_hidden": [8]},
                "stage2": {"oracle_name": "mlp_small"},
            },
            seed=7,
            device="cpu",
            out_dir=ROOT / "outputs" / "stage3_main_test3",
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
            remap_train_source="all",
            remap_train_objective="identity",
            loss="huber",
            huber_beta=1.0,
            grad_clip=1.0,
            target_clip_sigma=5.0,
            weight_decay=1.0e-4,
            remap_mode="direct",
            remap_min_r2=0.0,
            remap_blend_alpha=0.7,
            remap_collapse_ratio_threshold=0.25,
            epochs=2,
            batch_size=4,
            lr=1.0e-3,
            protocol_auto_fix=True,
            feature_aliases_path="",
            feature_backend="auto",
            cicflowmeter_cmd="java -jar CICFlowMeter.jar",
            cicflowmeter_timeout=300,
            oracle_name="mlp_small",
            pcap_eval_use_oracle=False,
            pcap_align_min_coverage=0.85,
            pcap_eval_batch_size=128,
            pcap_cache_enable=False,
            pcap_cache_dir="",
            pcap_feature_fail_closed=False,
            pcap_feature_fail_on_partial_alignment=False,
            pcap_search_alphas=[1.0],
            pcap_path="",
            pcap_scan_dir="",
            pcap_scan_limit=10,
            pcap_scan_min_prob=0.5,
            pcap_scan_glob="*.pcap",
            pcap_scan_compare_existing=True,
            pcap_scan_pmal_weight=0.7,
            pcap_scan_target_fit_weight=0.2,
            pcap_scan_target_mod_fit_weight=0.1,
            pcap_source_selection_mode="random",
            pcap_source_sample_n=2,
            pcap_source_sample_seed=7,
            adv_samples_path="adv_samples.npz",
            copy_adv_samples=False,
            pcap_apply_fields=[],
            pcap_search_bidirectional=True,
            pcap_search_field_subsets=False,
            pcap_search_probe_topk=2,
            pcap_search_rounds=1,
            pcap_tcp_fixup=True,
            pcap_dst_port_policy="keep",
            pcap_dst_port_allowlist=[],
            pcap_out_dir="",
            pcap_apply_n=1,
            pcap_eval=False,
            pcap_compare_baselines=False,
            pcap_baseline_jobs=1,
            save_intermediate_results=False,
        )
        pcap_features = SimpleNamespace(metrics_snapshot=lambda: {"pcap_feature_statuses": ["ok"]})
        source_a = ROOT / "outputs" / "source_a.pcap"
        source_b = ROOT / "outputs" / "source_b.pcap"
        source_a.parent.mkdir(parents=True, exist_ok=True)
        source_a.write_bytes(b"a")
        source_b.write_bytes(b"b")
        selection = SimpleNamespace(
            selected_path=source_a,
            selected_source="scan_random",
            evasion_valid=True,
            scan_min_prob=0.5,
            source_selection_mode="random",
            candidate_paths=[source_a, source_b],
        )
        fake_search_result = SimpleNamespace(
            mods=np.asarray([[0.2, 0.3]], dtype=np.float32),
            pcap_written_count=1,
            pcap_kept_original_count=0,
            pcap_apply_time_sec=0.5,
            pcap_packet_count=2,
            pcap_pcaps_per_sec=2.0,
            pcap_packet_throughput_pps=10.0,
            pcap_selected_alpha_mean=0.25,
            pcap_selected_alphas=[0.25],
            pcap_selected_field_sets=["payload_scale"],
            pcap_selected_deployability_score_mean=0.7,
            pcap_selected_response_l2_mean=0.2,
            pcap_modified=True,
            pcap_out_dir="pcaps",
        )

        with ExitStack() as stack:
            stack.enter_context(patch("rdsynth.pipeline.stage3.load_stage_runtime", return_value=runtime))
            stack.enter_context(patch("rdsynth.pipeline.stage3.Stage3Settings.from_cfg", return_value=settings))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.detect_stage3_environment",
                    return_value=SimpleNamespace(
                        scapy_available=True, nfstream_available=False, cicflowmeter_available=False
                    ),
                )
            )
            stack.enter_context(
                patch("rdsynth.pipeline.stage3.load_data_context", return_value=SimpleNamespace(bundle=bundle))
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.DatasetPreprocessor.from_bundle",
                    return_value=SimpleNamespace(
                        inverse_transform=Mock(side_effect=lambda x: x),
                        feature_mean=Mock(return_value=np.zeros(2, dtype=np.float32)),
                    ),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.validate_remap_mode"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.load_feature_aliases", return_value={}))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.load_stage3_artifacts",
                    return_value=SimpleNamespace(
                        surrogate=None,
                        oracle=None,
                        checkpoint_path=SimpleNamespace(exists=lambda: False),
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.resolve_pcap_eval_model",
                    return_value=SimpleNamespace(pcap_eval_model=object(), pcap_eval_model_name="surrogate"),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.PcapFeatureExtractor", return_value=pcap_features))
            stack.enter_context(patch("rdsynth.pipeline.stage3.build_pcap_selection", return_value=selection))
            stack.enter_context(patch("rdsynth.pipeline.stage3.resolve_adv_samples_path", return_value=Path(__file__)))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.load_adv_samples",
                    return_value=SimpleNamespace(
                        adv=np.asarray([[0.1, 0.2]], dtype=np.float32),
                        adv_norm=None,
                        adv_mean=None,
                        adv_std=None,
                        loaded=True,
                        count=1,
                        adv_space="pre",
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.build_rule_based_modifications",
                    return_value=np.asarray([[0.3, 0.4]], dtype=np.float32),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.clip_modifications", side_effect=lambda x: x))
            stack.enter_context(patch("rdsynth.pipeline.stage3.resolve_selected_pcap", return_value=selection))
            stack.enter_context(patch("rdsynth.pipeline.stage3.record_pcap_apply_settings"))
            stack.enter_context(
                patch("rdsynth.pipeline.stage3._pcap_output_dir", return_value=ROOT / "outputs" / "pcaps_out_multi")
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.prepare_pcap_search_context",
                    return_value=SimpleNamespace(
                        pcap_target_mod=np.asarray([0.3, 0.4], dtype=np.float32),
                        orig_pmal_for_selection=0.8,
                        orig_feat_pre_mean=np.asarray([0.0, 0.0], dtype=np.float32),
                        target_metric_fn=lambda path, target: (0.1, 0.2, 0.1, 0.2, {}),
                    ),
                )
            )
            search_and_write = stack.enter_context(
                patch("rdsynth.pipeline.stage3.search_and_write_pcaps", return_value=fake_search_result)
            )
            save_metrics = stack.enter_context(patch("rdsynth.pipeline.stage3.save_metrics"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.save_metrics_csv"))
            stack.enter_context(
                patch("rdsynth.pipeline.stage3.build_stage_output_files", return_value={"state": "none"})
            )
            save_manifest = stack.enter_context(patch("rdsynth.pipeline.stage3.save_stage_manifest_spec"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.save_config"))
            stack.enter_context(patch("builtins.print"))
            stack.enter_context(
                patch.dict(
                    "sys.modules",
                    {
                        "scapy": SimpleNamespace(),
                        "scapy.all": SimpleNamespace(rdpcap=lambda path: [1, 2], wrpcap=lambda path, pkts: None),
                    },
                )
            )
            main("configs/demo.yaml")

        self.assertGreaterEqual(search_and_write.call_count, 2)
        save_metrics.assert_called_once()
        manifest_spec = save_manifest.call_args.args[0]
        self.assertTrue(manifest_spec.metrics["pcap_modified"])

    def test_stage3_main_random_hard_mode_processes_multiple_pcaps(self) -> None:
        runtime = SimpleNamespace(
            cfg={
                "project": {"out_dir": str(ROOT / "outputs" / "stage3_project4")},
                "stage1": {"sur_hidden": [8]},
                "stage2": {"oracle_name": "mlp_small"},
            },
            seed=7,
            device="cpu",
            out_dir=ROOT / "outputs" / "stage3_main_test4",
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
            remap_train_source="all",
            remap_train_objective="identity",
            loss="huber",
            huber_beta=1.0,
            grad_clip=1.0,
            target_clip_sigma=5.0,
            weight_decay=1.0e-4,
            remap_mode="direct",
            remap_min_r2=0.0,
            remap_blend_alpha=0.7,
            remap_collapse_ratio_threshold=0.25,
            epochs=2,
            batch_size=4,
            lr=1.0e-3,
            protocol_auto_fix=True,
            feature_aliases_path="",
            feature_backend="auto",
            cicflowmeter_cmd="java -jar CICFlowMeter.jar",
            cicflowmeter_timeout=300,
            oracle_name="mlp_small",
            pcap_eval_use_oracle=False,
            pcap_align_min_coverage=0.85,
            pcap_eval_batch_size=128,
            pcap_cache_enable=False,
            pcap_cache_dir="",
            pcap_feature_fail_closed=False,
            pcap_feature_fail_on_partial_alignment=False,
            pcap_search_alphas=[1.0],
            pcap_path="",
            pcap_scan_dir="",
            pcap_scan_limit=10,
            pcap_scan_min_prob=0.5,
            pcap_scan_glob="*.pcap",
            pcap_scan_compare_existing=True,
            pcap_scan_pmal_weight=0.7,
            pcap_scan_target_fit_weight=0.2,
            pcap_scan_target_mod_fit_weight=0.1,
            pcap_source_selection_mode="random_hard",
            pcap_source_sample_n=2,
            pcap_source_sample_seed=7,
            adv_samples_path="adv_samples.npz",
            copy_adv_samples=False,
            pcap_apply_fields=[],
            pcap_search_bidirectional=True,
            pcap_search_field_subsets=False,
            pcap_search_probe_topk=2,
            pcap_search_rounds=1,
            pcap_tcp_fixup=True,
            pcap_dst_port_policy="keep",
            pcap_dst_port_allowlist=[],
            pcap_out_dir="",
            pcap_apply_n=1,
            pcap_eval=False,
            pcap_compare_baselines=False,
            pcap_baseline_jobs=1,
            save_intermediate_results=False,
        )
        pcap_features = SimpleNamespace(metrics_snapshot=lambda: {"pcap_feature_statuses": ["ok"]})
        source_a = ROOT / "outputs" / "source_hard_a.pcap"
        source_b = ROOT / "outputs" / "source_hard_b.pcap"
        source_a.parent.mkdir(parents=True, exist_ok=True)
        source_a.write_bytes(b"a")
        source_b.write_bytes(b"b")
        selection = SimpleNamespace(
            selected_path=source_a,
            selected_source="scan_random_hard",
            evasion_valid=True,
            scan_min_prob=0.5,
            source_selection_mode="random_hard",
            candidate_paths=[source_a, source_b],
        )
        fake_search_result = SimpleNamespace(
            mods=np.asarray([[0.2, 0.3]], dtype=np.float32),
            pcap_written_count=1,
            pcap_kept_original_count=0,
            pcap_apply_time_sec=0.5,
            pcap_packet_count=2,
            pcap_pcaps_per_sec=2.0,
            pcap_packet_throughput_pps=10.0,
            pcap_selected_alpha_mean=0.25,
            pcap_selected_alphas=[0.25],
            pcap_selected_field_sets=["payload_scale"],
            pcap_selected_deployability_score_mean=0.7,
            pcap_selected_response_l2_mean=0.2,
            pcap_modified=True,
            pcap_out_dir="pcaps",
        )

        with ExitStack() as stack:
            stack.enter_context(patch("rdsynth.pipeline.stage3.load_stage_runtime", return_value=runtime))
            stack.enter_context(patch("rdsynth.pipeline.stage3.Stage3Settings.from_cfg", return_value=settings))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.detect_stage3_environment",
                    return_value=SimpleNamespace(
                        scapy_available=True, nfstream_available=False, cicflowmeter_available=False
                    ),
                )
            )
            stack.enter_context(
                patch("rdsynth.pipeline.stage3.load_data_context", return_value=SimpleNamespace(bundle=bundle))
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.DatasetPreprocessor.from_bundle",
                    return_value=SimpleNamespace(
                        inverse_transform=Mock(side_effect=lambda x: x),
                        feature_mean=Mock(return_value=np.zeros(2, dtype=np.float32)),
                    ),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.validate_remap_mode"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.load_feature_aliases", return_value={}))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.load_stage3_artifacts",
                    return_value=SimpleNamespace(
                        surrogate=None,
                        oracle=None,
                        checkpoint_path=SimpleNamespace(exists=lambda: False),
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.resolve_pcap_eval_model",
                    return_value=SimpleNamespace(pcap_eval_model=object(), pcap_eval_model_name="surrogate"),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.PcapFeatureExtractor", return_value=pcap_features))
            stack.enter_context(patch("rdsynth.pipeline.stage3.build_pcap_selection", return_value=selection))
            stack.enter_context(patch("rdsynth.pipeline.stage3.resolve_adv_samples_path", return_value=Path(__file__)))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.load_adv_samples",
                    return_value=SimpleNamespace(
                        adv=np.asarray([[0.1, 0.2]], dtype=np.float32),
                        adv_norm=None,
                        adv_mean=None,
                        adv_std=None,
                        loaded=True,
                        count=1,
                        adv_space="pre",
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.build_rule_based_modifications",
                    return_value=np.asarray([[0.3, 0.4]], dtype=np.float32),
                )
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.clip_modifications", side_effect=lambda x: x))
            stack.enter_context(patch("rdsynth.pipeline.stage3.resolve_selected_pcap", return_value=selection))
            stack.enter_context(patch("rdsynth.pipeline.stage3.record_pcap_apply_settings"))
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3._pcap_output_dir", return_value=ROOT / "outputs" / "pcaps_out_multi_hard"
                )
            )
            stack.enter_context(
                patch(
                    "rdsynth.pipeline.stage3.prepare_pcap_search_context",
                    return_value=SimpleNamespace(
                        pcap_target_mod=np.asarray([0.3, 0.4], dtype=np.float32),
                        orig_pmal_for_selection=0.8,
                        orig_feat_pre_mean=np.asarray([0.0, 0.0], dtype=np.float32),
                        target_metric_fn=lambda path, target: (0.1, 0.2, 0.1, 0.2, {}),
                    ),
                )
            )
            search_and_write = stack.enter_context(
                patch("rdsynth.pipeline.stage3.search_and_write_pcaps", return_value=fake_search_result)
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.save_metrics"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.save_metrics_csv"))
            stack.enter_context(
                patch("rdsynth.pipeline.stage3.build_stage_output_files", return_value={"state": "none"})
            )
            stack.enter_context(patch("rdsynth.pipeline.stage3.save_stage_manifest_spec"))
            stack.enter_context(patch("rdsynth.pipeline.stage3.save_config"))
            stack.enter_context(patch("builtins.print"))
            stack.enter_context(
                patch.dict(
                    "sys.modules",
                    {
                        "scapy": SimpleNamespace(),
                        "scapy.all": SimpleNamespace(rdpcap=lambda path: [1, 2], wrpcap=lambda path, pkts: None),
                    },
                )
            )
            main("configs/demo.yaml")

        self.assertGreaterEqual(search_and_write.call_count, 2)


if __name__ == "__main__":
    unittest.main()
