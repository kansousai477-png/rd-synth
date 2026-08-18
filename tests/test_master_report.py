from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

import generate_reviewer_suite_master_report_cn as report


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else ["placeholder"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


class MasterReportTest(unittest.TestCase):
    def test_build_report_contains_overview_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_root = root / "reports" / "dataset_audit"

            _write_csv(
                root / "nb15" / "main_runs.csv",
                [
                    {
                        "attack_type": "GLOBAL",
                        "stage1_decision_score": "0.8",
                        "stage2_decision_score": "0.7",
                        "stage2_asr_oracle": "1.0",
                        "stage2_norm_ffd": "10.0",
                        "stage2_norm_swd": "0.2",
                        "stage3_score_scope": "full",
                        "stage3_decision_score": "0.6",
                        "stage3_deployability_score": "0.95",
                        "stage3_pcap_attack_success_rate": "1.0",
                        "stage3_source_attack_success_rate": "0.0",
                        "stage3_adv_attack_success_rate": "1.0",
                        "stage3_source_flow_attack_success_rate": "0.0",
                        "stage3_adv_flow_attack_success_rate": "0.8",
                        "stage3_pcap_valid_fatal_rate": "0.0",
                        "stage3_total_time_sec": "10.0",
                    }
                ],
            )
            _write_csv(
                root / "nb15" / "ablation_coverage.csv",
                [
                    {
                        "variant": "full",
                        "status": "completed",
                        "n_runs": "1",
                        "attack_count": "1",
                        "seed_count": "1",
                    },
                    {
                        "variant": "random_remap",
                        "status": "missing",
                        "n_runs": "0",
                        "attack_count": "0",
                        "seed_count": "0",
                    },
                ],
            )
            _write_csv(
                root / "2017" / "main_runs.csv",
                [
                    {
                        "attack_type": "GLOBAL",
                        "stage1_decision_score": "0.7",
                        "stage2_decision_score": "0.6",
                        "stage2_asr_oracle": "0.9",
                        "stage2_norm_ffd": "12.0",
                        "stage2_norm_swd": "0.3",
                        "stage3_score_scope": "full",
                        "stage3_decision_score": "0.5",
                        "stage3_deployability_score": "0.90",
                        "stage3_pcap_attack_success_rate": "0.8",
                        "stage3_source_attack_success_rate": "0.1",
                        "stage3_adv_attack_success_rate": "0.8",
                        "stage3_source_flow_attack_success_rate": "0.2",
                        "stage3_adv_flow_attack_success_rate": "0.7",
                        "stage3_pcap_valid_fatal_rate": "0.1",
                        "stage3_total_time_sec": "12.0",
                    }
                ],
            )
            _write_csv(
                root / "2017" / "ablation_coverage.csv",
                [
                    {
                        "variant": "full",
                        "status": "completed",
                        "n_runs": "1",
                        "attack_count": "1",
                        "seed_count": "1",
                    }
                ],
            )
            _write_csv(
                audit_root / "dataset_audit_summary.csv",
                [
                    {
                        "dataset": "cic_unsw",
                        "sampled_rows": "1000",
                        "feature_count": "70",
                        "positive_rate": "0.8",
                        "duplicate_rate_sample": "0.01",
                        "split_overlap_rate": "0.00",
                        "top_auc_feature": "Flow Duration",
                        "top_auc_value": "0.77",
                    },
                    {
                        "dataset": "cic_ids2017",
                        "sampled_rows": "1200",
                        "feature_count": "80",
                        "positive_rate": "0.6",
                        "duplicate_rate_sample": "0.02",
                        "split_overlap_rate": "0.01",
                        "top_auc_feature": "Flow Bytes/s",
                        "top_auc_value": "0.70",
                    },
                ],
            )

            report_path, _, _ = report.build_report(root, ["nb15", "2017"], audit_root)
            text = report_path.read_text(encoding="utf-8-sig")

            self.assertIn("Cross-Dataset Overview", text)
            self.assertIn("Dataset Audit Overview", text)
            self.assertIn("Stage1 指标速读", text)
            self.assertIn("Stage2 指标速读", text)
            self.assertIn("Stage3 指标速读", text)
            self.assertIn("Ablation Coverage", text)
            self.assertIn("missing", text)
            self.assertIn("Source PCAP Evasion", text)
            self.assertIn("Adv PCAP Evasion", text)
            self.assertIn("Source Flow Evasion", text)
            self.assertIn("Adv Flow Evasion", text)
            self.assertIn("CIC NB15", text)
            self.assertIn("CIC-IDS2017", text)


if __name__ == "__main__":
    unittest.main()
