from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_pcap_eval import (
    _fatal_validity_flag,
    aggregate_pcap_sanity,
    evaluate_adversarial_pcaps,
    evaluate_original_pcap,
    extend_sanity_values,
    finalize_pcap_eval,
)


class _DummyPcapFeatures:
    def metrics_snapshot(self) -> dict[str, object]:
        return {"dummy_metric": 1}


class _EvalPcapFeatures:
    def extract(self, pcap_file: str):
        path = Path(pcap_file).name
        if path == "orig.pcap":
            return (
                np.asarray([[1.0, 2.0]], dtype=np.float32),
                "nfstream",
                {"status": "ok", "alignment": {"coverage": 1.0, "missing": 0}},
            )
        return (
            np.asarray([[3.0, 4.0]], dtype=np.float32),
            "nfstream",
            {"status": "ok", "alignment": {"coverage": 0.8, "missing": 1}},
        )

    def classify_features(self, feat: np.ndarray):
        if float(feat[0, 0]) == 1.0:
            return np.asarray([0.1, 0.9], dtype=np.float32), feat
        return np.asarray([0.7, 0.3], dtype=np.float32), feat


class Stage3PcapEvalTest(unittest.TestCase):
    def test_aggregate_pcap_sanity_sets_expected_metrics(self) -> None:
        metrics = {}
        aggregate_pcap_sanity(
            {
                "nonmonotonic_rate": [0.1, 0.3],
                "tcp_flag_invalid_rate": [0.0, 0.2],
            },
            metrics,
        )
        self.assertEqual(metrics["pcap_sanity_nonmonotonic_rate"], 0.2)
        self.assertEqual(metrics["pcap_sanity_tcp_flag_invalid_rate"], 0.1)

    def test_extend_sanity_values_collects_matching_keys(self) -> None:
        sanity_vals = {
            "nonmonotonic_rate": [],
            "transport_missing_rate": [],
            "tcp_seq_backwards_rate": [],
            "tcp_flag_invalid_rate": [],
            "tcp_syn_fin_rate": [],
            "tcp_syn_rst_rate": [],
            "tcp_fin_rst_rate": [],
        }
        extend_sanity_values(
            sanity_vals,
            {
                "sanity_nonmonotonic_rate": 0.2,
                "sanity_tcp_flag_invalid_rate": 0.4,
            },
        )
        self.assertEqual(sanity_vals["nonmonotonic_rate"], [0.2])
        self.assertEqual(sanity_vals["tcp_flag_invalid_rate"], [0.4])

    def test_evaluate_original_pcap_sets_metrics_and_row(self) -> None:
        metrics = {}
        result = evaluate_original_pcap(
            pcap_path=Path("orig.pcap"),
            source_name="orig.pcap",
            pcap_features=_EvalPcapFeatures(),
            scapy_available=False,
            metrics_payload=metrics,
            pcap_evasion_valid=None,
            scan_min_prob=0.5,
        )
        self.assertTrue(result.pcap_evasion_valid)
        self.assertEqual(result.row["pred_label"], 1)
        self.assertEqual(metrics["pcap_orig_pred_malicious"], 1.0)
        self.assertAlmostEqual(metrics["pcap_orig_prob_malicious"], 0.9, places=6)

    def test_evaluate_adversarial_pcaps_collects_rows_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adv_pcap = Path(tmp_dir) / "adv_0000.pcap"
            adv_pcap.write_bytes(b"x")
            result = evaluate_adversarial_pcaps(
                adv_pcaps=[adv_pcap],
                source_name="orig.pcap",
                pcap_features=_EvalPcapFeatures(),
                scapy_available=False,
                adv=np.asarray([[1.0, 1.0]], dtype=np.float32),
                feature_names=["a", "b"],
                original_sanity={},
            )

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["pred_label"], 0)
        self.assertEqual(len(result.target_l2_vals), 1)
        self.assertEqual(len(result.target_mae_vals), 1)
        self.assertEqual(result.fatal_validity_flags, [0.0])

    def test_fatal_validity_flag_ignores_source_pcap_baseline_noise(self) -> None:
        self.assertEqual(
            _fatal_validity_flag(
                {"sanity_tcp_seq_backwards_rate": 0.0011},
                original_sanity={"sanity_tcp_seq_backwards_rate": 0.0011},
            ),
            0.0,
        )
        self.assertEqual(
            _fatal_validity_flag(
                {"sanity_tcp_seq_backwards_rate": 0.0050},
                original_sanity={"sanity_tcp_seq_backwards_rate": 0.0011},
            ),
            1.0,
        )

    def test_finalize_pcap_eval_writes_csv_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = {
                "pcap_orig_pred_malicious": 1.0,
                "pcap_apply_time_sec": 1.0,
                "pcap_pcaps_per_sec": 2.0,
                "pcap_packet_throughput_pps": 10.0,
            }
            rows = [
                {
                    "pcap": "orig.pcap",
                    "source_name": "orig.pcap",
                    "is_original": 1,
                    "flow_count": 1,
                    "feature_backend": "nfstream",
                    "feature_status": "ok",
                    "feature_reason": "",
                    "alignment_coverage": 1.0,
                    "alignment_missing": 0,
                    "prob_benign": 0.1,
                    "prob_malicious": 0.9,
                    "pred_label": 1,
                    "target_idx": "",
                    "target_l2": "",
                    "target_mae": "",
                    "sanity_nonmonotonic_rate": 0.0,
                    "sanity_transport_missing_rate": 0.0,
                    "sanity_tcp_seq_backwards_rate": 0.0,
                    "sanity_tcp_flag_invalid_rate": 0.0,
                    "sanity_tcp_syn_fin_rate": 0.0,
                    "sanity_tcp_syn_rst_rate": 0.0,
                    "sanity_tcp_fin_rst_rate": 0.0,
                },
                {
                    "pcap": "adv_0000.pcap",
                    "source_name": "orig.pcap",
                    "is_original": 0,
                    "flow_count": 1,
                    "feature_backend": "nfstream",
                    "feature_status": "ok",
                    "feature_reason": "",
                    "alignment_coverage": 0.8,
                    "alignment_missing": 1,
                    "prob_benign": 0.7,
                    "prob_malicious": 0.3,
                    "pred_label": 0,
                    "target_idx": 0,
                    "target_l2": 1.2,
                    "target_mae": 0.5,
                    "sanity_nonmonotonic_rate": 0.0,
                    "sanity_transport_missing_rate": 0.0,
                    "sanity_tcp_seq_backwards_rate": 0.0,
                    "sanity_tcp_flag_invalid_rate": 0.0,
                    "sanity_tcp_syn_fin_rate": 0.0,
                    "sanity_tcp_syn_rst_rate": 0.0,
                    "sanity_tcp_fin_rst_rate": 0.0,
                },
            ]
            out_rows = finalize_pcap_eval(
                out_dir=Path(tmp_dir),
                eval_rows=rows,
                metrics_payload=metrics,
                pcap_features=_DummyPcapFeatures(),
                target_l2_vals=[1.2],
                target_mae_vals=[0.5],
                fatal_validity_flags=[0.0],
            )
            self.assertEqual(len(out_rows), 2)
            self.assertTrue((Path(tmp_dir) / "pcap_eval.csv").exists())
            self.assertTrue(metrics["pcap_eval"])
            self.assertEqual(metrics["pcap_feature_statuses"], ["ok"])
            self.assertEqual(metrics["dummy_metric"], 1)
            self.assertEqual(metrics["pcap_source_attack_success_rate"], 0.0)
            self.assertEqual(metrics["pcap_adv_attack_success_rate"], 1.0)
            self.assertEqual(metrics["pcap_source_detected_count"], 1)
            self.assertEqual(metrics["pcap_source_already_evasive_count"], 0)
            self.assertEqual(metrics["pcap_replay_eligible_source_count"], 1)
            self.assertEqual(metrics["pcap_conditional_attack_success_rate"], 1.0)
            self.assertEqual(metrics["paper_pcap_attack_success_rate"], 1.0)
            self.assertEqual(metrics["pcap_source_flow_attack_success_rate"], 0.0)
            self.assertEqual(metrics["pcap_adv_flow_attack_success_rate"], 1.0)

    def test_finalize_pcap_eval_does_not_count_already_evasive_source_as_paper_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics = {
                "pcap_apply_time_sec": 1.0,
                "pcap_pcaps_per_sec": 2.0,
                "pcap_packet_throughput_pps": 10.0,
            }
            rows = [
                {
                    "pcap": "orig.pcap",
                    "source_name": "orig.pcap",
                    "is_original": 1,
                    "flow_count": 1,
                    "feature_backend": "nfstream",
                    "feature_status": "ok",
                    "feature_reason": "",
                    "alignment_coverage": 1.0,
                    "alignment_missing": 0,
                    "prob_benign": 0.8,
                    "prob_malicious": 0.2,
                    "pred_label": 0,
                    "target_idx": "",
                    "target_l2": "",
                    "target_mae": "",
                    "sanity_nonmonotonic_rate": 0.0,
                    "sanity_transport_missing_rate": 0.0,
                    "sanity_tcp_seq_backwards_rate": 0.0,
                    "sanity_tcp_flag_invalid_rate": 0.0,
                    "sanity_tcp_syn_fin_rate": 0.0,
                    "sanity_tcp_syn_rst_rate": 0.0,
                    "sanity_tcp_fin_rst_rate": 0.0,
                }
            ]
            finalize_pcap_eval(
                out_dir=Path(tmp_dir),
                eval_rows=rows,
                metrics_payload=metrics,
                pcap_features=_DummyPcapFeatures(),
                target_l2_vals=[],
                target_mae_vals=[],
                fatal_validity_flags=[],
            )

        self.assertEqual(metrics["pcap_source_detected_count"], 0)
        self.assertEqual(metrics["pcap_source_already_evasive_count"], 1)
        self.assertEqual(metrics["pcap_replay_eligible_source_count"], 0)
        self.assertTrue(np.isnan(metrics["paper_pcap_attack_success_rate"]))


if __name__ == "__main__":
    unittest.main()
