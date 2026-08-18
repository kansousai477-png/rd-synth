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

import generate_dataset_full_report_cn as report


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else ["placeholder"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


class DatasetFullReportTest(unittest.TestCase):
    def test_build_report_includes_audit_and_global_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "nb15"
            audit_root = root / "reports" / "dataset_audit"

            _write_csv(
                dataset_root / "main_runs.csv",
                [
                    {
                        "attack_type": "GLOBAL",
                        "seed": "42",
                        "stage1_decision_score": "0.8",
                        "stage1_agreement": "0.9",
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
                        "stage3_adv_flow_attack_success_rate": "0.9",
                        "stage3_pcap_valid_fatal_rate": "0.0",
                        "pcap_selected_name": "carrier.pcap",
                    }
                ],
            )
            _write_csv(
                dataset_root / "rq1_matrix_summary.csv",
                [
                    {
                        "attack_type": "GLOBAL",
                        "seed": "42",
                        "ids_count": "6",
                        "diag_mean": "0.9",
                        "within_group_mean": "0.8",
                        "cross_group_mean": "0.3",
                    }
                ],
            )
            _write_csv(dataset_root / "attack_level_summary.csv", [{"attack_type": "GLOBAL", "n_seeds": "1"}])
            _write_csv(
                dataset_root / "stage2_outcome_summary.csv", [{"attack_type": "GLOBAL", "asr_main_ids_mean": "1.0"}]
            )
            _write_csv(dataset_root / "stage2_baseline_summary.csv", [{"method": "pgd", "attack_type": "GLOBAL"}])
            _write_csv(dataset_root / "stage3_baseline_summary.csv", [{"method": "pgd", "attack_type": "GLOBAL"}])
            _write_csv(
                dataset_root / "main_transfer_ids_summary.csv", [{"ids_name": "logistic_small", "adv_asr_mean": "0.9"}]
            )
            _write_csv(
                dataset_root / "failure_boundary_summary.csv", [{"attack_type": "GLOBAL", "failure_signature": "none"}]
            )
            _write_csv(dataset_root / "ablation_variant_summary.csv", [{"variant": "full", "n_runs": "1"}])
            _write_csv(dataset_root / "ablation_coverage.csv", [{"variant": "full", "status": "ok"}])
            _write_csv(
                dataset_root / "efficiency_summary.csv",
                [{"attack_type": "GLOBAL", "stage3_total_time_sec_mean": "10.0"}],
            )
            _write_csv(
                audit_root / "dataset_audit_summary.csv",
                [
                    {
                        "dataset": "cic_unsw",
                        "sampled_rows": "1000",
                        "feature_count": "70",
                        "positive_rate": "0.8",
                        "constant_cols": "1",
                        "near_constant_cols": "2",
                        "duplicate_rate_sample": "0.01",
                        "split_overlap_rate": "0.00",
                        "top_auc_feature": "Flow Duration",
                        "top_auc_value": "0.77",
                    }
                ],
            )

            report_path, _ = report.build_report(root, "nb15", audit_root)
            text = report_path.read_text(encoding="utf-8-sig")

            self.assertIn("Dataset Audit", text)
            self.assertIn("GLOBAL 主结果", text)
            self.assertIn("Stage1 指标速读", text)
            self.assertIn("Stage2 指标速读", text)
            self.assertIn("Stage3 指标读表词典", text)
            self.assertIn("Source PCAP Evasion", text)
            self.assertIn("Adv PCAP Evasion", text)
            self.assertIn("Source Flow Evasion", text)
            self.assertIn("Adv Flow Evasion", text)
            self.assertIn("Flow Duration", text)


if __name__ == "__main__":
    unittest.main()
