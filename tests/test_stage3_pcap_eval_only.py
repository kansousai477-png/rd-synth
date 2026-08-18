from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_pcap_eval_only import run_stage3_pcap_eval_only


class Stage3PcapEvalOnlyTest(unittest.TestCase):
    def test_run_stage3_pcap_eval_only_rewrites_eval_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage3_out = root / "outputs" / "stage3"
            stage3_out.mkdir(parents=True, exist_ok=True)
            orig_pcap = root / "orig.pcap"
            orig_pcap.write_bytes(b"pcap")
            adv_dir = stage3_out / "pcap"
            adv_dir.mkdir(parents=True, exist_ok=True)
            (adv_dir / "adv_0000.pcap").write_bytes(b"pcap")
            (stage3_out / "manifest.json").write_text(
                json.dumps({"inputs": {"pcap_path": str(orig_pcap)}}),
                encoding="utf-8",
            )
            (stage3_out / "metrics.json").write_text(
                json.dumps({"pcap_out_dir": str(adv_dir), "pcap_evasion_valid": True}),
                encoding="utf-8",
            )

            runtime = SimpleNamespace(
                cfg={
                    "project": {"out_dir": str(root / "outputs")},
                    "stage1": {"sur_hidden": [8]},
                    "stage2": {"oracle_name": "mlp_small"},
                },
                seed=7,
                device="cpu",
                out_dir=stage3_out,
                stage_cfg={},
            )
            settings = SimpleNamespace(
                oracle_name="mlp_small",
                pcap_eval_use_oracle=False,
                feature_backend="auto",
                feature_aliases_path="",
                cicflowmeter_cmd="java -jar CICFlowMeter.jar",
                cicflowmeter_timeout=300,
                pcap_align_min_coverage=0.85,
                pcap_feature_fail_closed=False,
                pcap_feature_fail_on_partial_alignment=False,
                pcap_eval_batch_size=32,
                pcap_cache_enable=False,
                pcap_cache_dir="",
                pcap_path="",
                pcap_out_dir="",
                adv_samples_path="",
                pcap_scan_min_prob=0.5,
            )
            bundle = SimpleNamespace(
                x_train=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                y_train=np.asarray([0, 1], dtype=np.int64),
                x_val=np.asarray([[0.2, 0.2]], dtype=np.float32),
                y_val=np.asarray([0], dtype=np.int64),
                feature_names=["f0", "f1"],
            )
            pcap_features = SimpleNamespace(metrics_snapshot=lambda: {"pcap_feature_statuses": ["ok"]})

            with (
                patch("rdsynth.pipeline.stage3_pcap_eval_only.load_stage_runtime", return_value=runtime),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.Stage3Settings.from_cfg", return_value=settings),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.detect_stage3_environment",
                    return_value=SimpleNamespace(
                        scapy_available=False, nfstream_available=False, cicflowmeter_available=False
                    ),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.load_data_context",
                    return_value=SimpleNamespace(bundle=bundle),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.DatasetPreprocessor.from_bundle",
                    return_value=SimpleNamespace(feature_mean=Mock(return_value=np.zeros(2, dtype=np.float32))),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.load_stage3_artifacts",
                    return_value=SimpleNamespace(
                        surrogate=None, oracle=object(), checkpoint_path=SimpleNamespace(exists=lambda: False)
                    ),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.resolve_pcap_eval_model",
                    return_value=SimpleNamespace(pcap_eval_model=object(), pcap_eval_model_name="oracle"),
                ),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.load_feature_aliases", return_value={}),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.PcapFeatureExtractor", return_value=pcap_features),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.resolve_adv_samples_path",
                    return_value=root / "adv_samples.npz",
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.load_adv_samples",
                    return_value=SimpleNamespace(adv=np.asarray([[0.1, 0.2]], dtype=np.float32), loaded=True, count=1),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.evaluate_original_pcap",
                    return_value=SimpleNamespace(
                        row={"pcap": str(orig_pcap), "prob_malicious": 0.9}, sanity={}, pcap_evasion_valid=True
                    ),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.evaluate_adversarial_pcaps",
                    return_value=SimpleNamespace(
                        rows=[{"pcap": str(adv_dir / "adv_0000.pcap"), "prob_malicious": 0.1, "pred_label": 0}],
                        target_l2_vals=[1.0],
                        target_mae_vals=[0.5],
                        fatal_validity_flags=[0.0],
                        sanity_values={
                            "nonmonotonic_rate": [],
                            "transport_missing_rate": [],
                            "tcp_seq_backwards_rate": [],
                            "tcp_flag_invalid_rate": [],
                            "tcp_syn_fin_rate": [],
                            "tcp_syn_rst_rate": [],
                            "tcp_fin_rst_rate": [],
                        },
                    ),
                ),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.finalize_pcap_eval") as finalize_eval,
                patch("rdsynth.pipeline.stage3_pcap_eval_only.aggregate_pcap_sanity"),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.save_metrics") as save_metrics,
                patch("rdsynth.pipeline.stage3_pcap_eval_only.save_metrics_csv") as save_metrics_csv,
                patch("builtins.print"),
            ):
                run_stage3_pcap_eval_only("configs/demo.yaml")

            finalize_eval.assert_called_once()
            save_metrics.assert_called_once()
            save_metrics_csv.assert_called_once()

    def test_run_stage3_pcap_eval_only_reuses_recorded_pcap_ids_training_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stage3_out = root / "outputs" / "stage3"
            stage3_out.mkdir(parents=True, exist_ok=True)
            orig_pcap = root / "orig.pcap"
            pool_pcap = root / "pool.pcap"
            orig_pcap.write_bytes(b"pcap")
            pool_pcap.write_bytes(b"pcap")
            adv_dir = stage3_out / "pcap"
            adv_dir.mkdir(parents=True, exist_ok=True)
            (stage3_out / "manifest.json").write_text(
                json.dumps({"inputs": {"pcap_path": str(orig_pcap)}}),
                encoding="utf-8",
            )
            (stage3_out / "metrics.json").write_text(
                json.dumps(
                    {
                        "pcap_out_dir": str(adv_dir),
                        "pcap_evasion_valid": True,
                        "pcap_ids_malicious_pcaps_configured": [str(pool_pcap)],
                    }
                ),
                encoding="utf-8",
            )
            runtime = SimpleNamespace(
                cfg={
                    "project": {"out_dir": str(root / "outputs")},
                    "stage1": {"sur_hidden": [8]},
                    "stage2": {"oracle_name": "mlp_small"},
                },
                seed=7,
                device="cpu",
                out_dir=stage3_out,
                stage_cfg={},
            )
            settings = SimpleNamespace(
                oracle_name="mlp_small",
                pcap_eval_use_oracle=False,
                pcap_eval_use_ids=True,
                feature_backend="auto",
                feature_aliases_path="",
                cicflowmeter_cmd="java -jar CICFlowMeter.jar",
                cicflowmeter_timeout=300,
                pcap_align_min_coverage=0.85,
                pcap_feature_fail_closed=False,
                pcap_feature_fail_on_partial_alignment=False,
                pcap_eval_batch_size=32,
                pcap_cache_enable=False,
                pcap_cache_dir="",
                pcap_path="",
                pcap_out_dir="",
                adv_samples_path="",
                pcap_scan_min_prob=0.5,
            )
            bundle = SimpleNamespace(
                x_train=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                y_train=np.asarray([0, 1], dtype=np.int64),
                x_val=np.asarray([[0.2, 0.2]], dtype=np.float32),
                y_val=np.asarray([0], dtype=np.int64),
                feature_names=["f0", "f1"],
            )
            preprocessor = SimpleNamespace(feature_mean=Mock(return_value=np.zeros(2, dtype=np.float32)))
            pcap_features = SimpleNamespace(metrics_snapshot=lambda: {"pcap_feature_statuses": ["ok"]})
            with (
                patch("rdsynth.pipeline.stage3_pcap_eval_only.load_stage_runtime", return_value=runtime),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.Stage3Settings.from_cfg", return_value=settings),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.detect_stage3_environment",
                    return_value=SimpleNamespace(
                        scapy_available=False, nfstream_available=False, cicflowmeter_available=False
                    ),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.load_data_context",
                    return_value=SimpleNamespace(bundle=bundle),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.DatasetPreprocessor.from_bundle", return_value=preprocessor
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.load_stage3_artifacts",
                    return_value=SimpleNamespace(
                        surrogate=None, oracle=object(), checkpoint_path=SimpleNamespace(exists=lambda: False)
                    ),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.resolve_pcap_eval_model",
                    return_value=SimpleNamespace(pcap_eval_model=object(), pcap_eval_model_name="ids"),
                ),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.train_stage3_ids") as train_ids,
                patch("rdsynth.pipeline.stage3_pcap_eval_only.load_feature_aliases", return_value={}),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.PcapFeatureExtractor", return_value=pcap_features),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.resolve_adv_samples_path",
                    return_value=root / "adv_samples.npz",
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.load_adv_samples",
                    return_value=SimpleNamespace(adv=np.asarray([[0.1, 0.2]], dtype=np.float32), loaded=True, count=1),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.evaluate_original_pcap",
                    return_value=SimpleNamespace(row={}, sanity={}, pcap_evasion_valid=True),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_eval_only.evaluate_adversarial_pcaps",
                    return_value=SimpleNamespace(
                        rows=[],
                        target_l2_vals=[],
                        target_mae_vals=[],
                        fatal_validity_flags=[],
                        sanity_values={
                            "nonmonotonic_rate": [],
                            "transport_missing_rate": [],
                            "tcp_seq_backwards_rate": [],
                            "tcp_flag_invalid_rate": [],
                            "tcp_syn_fin_rate": [],
                            "tcp_syn_rst_rate": [],
                            "tcp_fin_rst_rate": [],
                        },
                    ),
                ),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.finalize_pcap_eval"),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.aggregate_pcap_sanity"),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.save_metrics"),
                patch("rdsynth.pipeline.stage3_pcap_eval_only.save_metrics_csv"),
                patch("builtins.print"),
            ):
                train_ids.return_value = SimpleNamespace(ids_bundle=object(), metrics={})
                run_stage3_pcap_eval_only("configs/demo.yaml")

            self.assertEqual(train_ids.call_args.kwargs["malicious_pcaps"], [pool_pcap])


if __name__ == "__main__":
    unittest.main()
