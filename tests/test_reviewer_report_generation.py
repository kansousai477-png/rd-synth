from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

import generate_reviewer_report_cn as report


class ReviewerReportGenerationTest(unittest.TestCase):
    def test_ablation_coverage_marks_missing_variants(self) -> None:
        rows = [
            {"variant": "full", "attack_type": "Bot", "seed": "42"},
            {"variant": "backbone_wgan", "attack_type": "Bot", "seed": "42"},
        ]
        out = report.build_ablation_coverage(rows, expected_variants=["full", "backbone_wgan", "backbone_cgan"])
        by_variant = {row["variant"]: row for row in out}
        self.assertEqual(by_variant["full"]["status"], "completed")
        self.assertEqual(by_variant["backbone_cgan"]["status"], "missing")
        self.assertEqual(by_variant["backbone_cgan"]["n_runs"], "0")

    def test_limitation_lines_scope_to_flow_level_detectors(self) -> None:
        lines = report.limitation_lines()
        text = "\n".join(lines)
        self.assertIn("Raw-Packet NIDS", text)
        self.assertIn("流级统计特征", text)

    def test_aggregate_efficiency_summarizes_main_run_timings(self) -> None:
        rows = [
            {
                "attack_type": "Bot",
                "stage1_total_train_time_sec": "10.0",
                "stage1_surrogate_query_qps": "100.0",
                "stage2_end_to_end_time_sec": "20.0",
                "stage2_end_to_end_samples_per_sec": "50.0",
                "stage2_queries_per_success_oracle": "4.0",
                "stage3_total_time_sec": "30.0",
                "stage3_pcap_apply_time_sec": "5.0",
                "stage3_pcap_eval_time_sec": "6.0",
                "stage3_pcap_packet_throughput_pps": "1000.0",
            },
            {
                "attack_type": "Bot",
                "stage1_total_train_time_sec": "12.0",
                "stage1_surrogate_query_qps": "120.0",
                "stage2_end_to_end_time_sec": "24.0",
                "stage2_end_to_end_samples_per_sec": "60.0",
                "stage2_queries_per_success_oracle": "6.0",
                "stage3_total_time_sec": "36.0",
                "stage3_pcap_apply_time_sec": "7.0",
                "stage3_pcap_eval_time_sec": "8.0",
                "stage3_pcap_packet_throughput_pps": "1200.0",
            },
        ]
        out = report.aggregate_efficiency(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["attack_type"], "Bot")
        self.assertEqual(out[0]["stage2_end_to_end_time_sec_mean"], "22.000000")
        self.assertEqual(out[0]["stage3_packets_per_sec_mean"], "1100.000000")

    def test_build_failure_case_studies_orders_by_conflict_score(self) -> None:
        rows = [
            {
                "attack_type": "A",
                "seed": "42",
                "out_dir": "o1",
                "pcap_selected_name": "a.pcap",
                "stage3_score_scope": "full",
                "stage3_deployability_score": "0.9",
                "stage3_remap_quality_score": "0.95",
                "stage3_pcap_attack_success_rate": "0.8",
                "stage3_pcap_target_l2_mean": "5.0",
                "stage3_pcap_alignment_coverage": "0.99",
                "stage3_pcap_alignment_missing": "0",
                "stage3_pcap_valid_fatal_rate": "0.0",
                "stage3_remap_mod_source": "learned",
                "stage3_remap_collapse_ratio": "1.0",
            },
            {
                "attack_type": "B",
                "seed": "43",
                "out_dir": "o2",
                "pcap_selected_name": "b.pcap",
                "stage3_score_scope": "remap_only",
                "stage3_deployability_score": "0.3",
                "stage3_remap_quality_score": "0.9",
                "stage3_pcap_attack_success_rate": "0.0",
                "stage3_pcap_target_l2_mean": "30.0",
                "stage3_pcap_alignment_coverage": "0.6",
                "stage3_pcap_alignment_missing": "4",
                "stage3_pcap_valid_fatal_rate": "0.2",
                "stage3_remap_mod_source": "blended",
                "stage3_remap_collapse_ratio": "0.4",
            },
        ]
        out = report.build_failure_case_studies(rows, top_k=2)
        self.assertEqual(out[0]["attack_type"], "B")
        self.assertEqual(out[0]["rank"], "1")


if __name__ == "__main__":
    unittest.main()
