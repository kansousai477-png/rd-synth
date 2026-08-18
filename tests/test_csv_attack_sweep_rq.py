from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from _run_csv_attack_sweep import _rq_coverage_rows, _rq_summary_rows, _write_markdown_report


class CsvAttackSweepRqReportTest(unittest.TestCase):
    def test_rq_helpers_and_report_include_rq_sections(self) -> None:
        rows = [
            {
                "variant": "full",
                "attack_type": "Bot",
                "attack_variant": "Bot::full",
                "stage1_agreement": "0.92",
                "stage1_baseline_agreement": "0.70",
                "stage1_decision_score": "0.81",
                "stage1_oracle_acc": "0.97",
                "stage2_decision_score": "0.78",
                "stage2_attack_score": "0.84",
                "stage2_fidelity_score": "0.73",
                "stage2_constraint_score": "0.76",
                "stage2_norm_ffd": "12.0",
                "stage2_norm_swd": "0.2",
                "stage2_asr_oracle": "0.95",
                "stage2_adv_pmal_oracle": "0.11",
                "stage3_decision_score": "0.69",
                "stage3_deployability_score": "0.72",
                "stage3_remap_quality_score": "0.75",
                "stage3_remap_r2": "0.80",
                "stage3_adv_benign_rate": "0.70",
                "stage3_pcap_alignment_coverage": "0.94",
                "stage3_pcap_target_l2_mean": "7.2",
                "stage3_pcap_target_mae_mean": "0.8",
                "stage3_pcap_valid_fatal_rate": "0.0",
                "stage3_remap_mod_source": "learned",
            },
            {
                "variant": "backbone_wgan",
                "attack_type": "Bot",
                "attack_variant": "Bot::backbone_wgan",
                "stage1_agreement": "0.92",
                "stage1_baseline_agreement": "0.70",
                "stage1_decision_score": "0.81",
                "stage1_oracle_acc": "0.97",
                "stage2_decision_score": "0.70",
                "stage2_attack_score": "0.78",
                "stage2_fidelity_score": "0.60",
                "stage2_constraint_score": "0.71",
                "stage2_norm_ffd": "18.0",
                "stage2_norm_swd": "0.4",
                "stage2_asr_oracle": "0.91",
                "stage2_adv_pmal_oracle": "0.18",
                "stage3_decision_score": "0.62",
                "stage3_deployability_score": "0.65",
                "stage3_remap_quality_score": "0.66",
                "stage3_remap_r2": "0.72",
                "stage3_adv_benign_rate": "0.61",
                "stage3_pcap_alignment_coverage": "0.88",
                "stage3_pcap_target_l2_mean": "12.2",
                "stage3_pcap_target_mae_mean": "1.1",
                "stage3_pcap_valid_fatal_rate": "0.1",
                "stage3_remap_mod_source": "blended",
            },
        ]

        rq_summary = _rq_summary_rows(rows)
        rq_coverage = _rq_coverage_rows(rows)
        self.assertEqual(rq_summary[0]["rq_id"], "RQ1")
        self.assertEqual(rq_coverage[1]["rq_id"], "RQ2")
        self.assertEqual(rq_coverage[1]["coverage"], "covered")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            family_path = root / "family_summary.csv"
            with family_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["family", "attacks", "stage2_decision_score"])
                writer.writeheader()
                writer.writerow({"family": "full:Bot", "attacks": "1", "stage2_decision_score": "0.78"})
            report_path = root / "EXPERIMENT_REPORT.md"
            _write_markdown_report(report_path, rows, family_path, "ToyDataset", root)
            text = report_path.read_text(encoding="utf-8")

        self.assertIn("## RQ Coverage", text)
        self.assertIn("## RQ-Oriented Discussion", text)
        self.assertIn("### RQ1. Surrogate Extraction Effectiveness", text)
        self.assertIn("### RQ5. Packet Realism After Remapping", text)
        self.assertIn("### RQ6. Online Validation", text)


if __name__ == "__main__":
    unittest.main()
