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

import generate_nb15_all_in_one_reports as report


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else ["Attack"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


class Nb15AllInOneReportsTest(unittest.TestCase):
    def test_build_all_in_one_writes_reports_with_support_and_transfer_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "nb15"

            _write_csv(
                dataset_root / "nb15_global_main_table.csv",
                [{"Attack": "GLOBAL", "Stage2 ASR": "1.0000", "Stage3 replay ASR": "1.0000", "Fatal rate": "0.0000"}],
            )
            _write_csv(
                dataset_root / "nb15_stage2_global_method_compare.csv",
                [
                    {
                        "Attack": "GLOBAL",
                        "Method": "RD-Synth",
                        "ASR_oracle": "1.0000",
                        "ASR_surrogate": "1.0000",
                        "Score": "0.9000",
                        "FFD": "10.0",
                        "SWD": "0.2",
                    },
                    {
                        "Attack": "GLOBAL",
                        "Method": "global_random",
                        "ASR_oracle": "0.9000",
                        "ASR_surrogate": "0.9000",
                        "Score": "0.7000",
                        "FFD": "5.0",
                        "SWD": "0.1",
                    },
                ],
            )
            _write_csv(
                dataset_root / "nb15_stage2_realism_table.csv",
                [
                    {
                        "Attack": "GLOBAL",
                        "Method": "RD-Synth",
                        "Family": "ours",
                        "FFD": "10.0",
                        "SWD": "0.2",
                        "Energy": "0.3",
                        "C2ST_AUC": "0.6",
                        "Coverage@5": "0.8",
                        "kNN_R": "0.7",
                        "kNN_P": "0.6",
                        "Corr_Delta": "0.1",
                        "CovSpec_L2": "1.0",
                        "CovTrace": "0.2",
                        "PairDist_KS": "0.3",
                        "PairMean": "0.4",
                        "Queries_per_success": "5.0",
                        "AdvToMal_L2": "2.0",
                        "Time_sec": "3.0",
                    },
                    {
                        "Attack": "GLOBAL",
                        "Method": "global_random",
                        "Family": "control",
                        "FFD": "5.0",
                        "SWD": "0.1",
                        "Energy": "0.2",
                        "C2ST_AUC": "0.5",
                        "Coverage@5": "0.9",
                        "kNN_R": "0.8",
                        "kNN_P": "0.7",
                        "Corr_Delta": "0.2",
                        "CovSpec_L2": "0.9",
                        "CovTrace": "0.1",
                        "PairDist_KS": "0.2",
                        "PairMean": "0.3",
                        "Queries_per_success": "4.0",
                        "AdvToMal_L2": "1.0",
                        "Time_sec": "2.0",
                    },
                ],
            )
            _write_csv(
                dataset_root / "nb15_stage2_cgd_table.csv",
                [
                    {"Method": "RD-Synth", "CGD_AVG": "0.2", "CGD_ST": "0.1", "CGD_SP": "0.2", "CGD_TP": "0.3"},
                    {"Method": "global_random", "CGD_AVG": "0.1", "CGD_ST": "0.1", "CGD_SP": "0.1", "CGD_TP": "0.1"},
                ],
            )
            _write_csv(
                dataset_root / "unsw_attack_slice_table.csv",
                [
                    {
                        "Attack": "DoS",
                        "Eval rows": "100",
                        "ASR_oracle": "1.0000",
                        "ASR_surrogate": "1.0000",
                        "FFD": "8.0",
                        "SWD": "0.2",
                        "AdvToMal_L2": "1.5",
                        "Score": "0.9",
                        "Time_sec": "5.0",
                        "Stage1 agreement": "0.99",
                        "Stage1 baseline agreement": "0.95",
                    }
                ],
            )
            _write_csv(
                dataset_root / "unsw_ablation_detail_table.csv",
                [{"Variant": "full", "Score_aux": "0.9", "S2_FFD": "10.0", "S3_Fatal": "0.0", "Remap_R2": "0.99"}],
            )
            _write_csv(
                dataset_root / "unsw_stage3_summary_table.csv",
                [
                    {
                        "Replay ASR": "1.0000",
                        "Deployability": "0.9800",
                        "Fatal rate": "0.0000",
                        "Stage2 oracle": "mlp_small",
                        "Stage3 ids": "mlp_small",
                        "Stage3 main_ids": "mlp_small",
                    }
                ],
            )
            _write_csv(
                dataset_root / "unsw_stage3_carrier_eval_table.csv",
                [
                    {
                        "Attack": "GLOBAL",
                        "Carrier": "carrier_a",
                        "Eval_status": "adv_replayed",
                        "Source_pred": "0",
                        "Adv_pred": "0",
                        "Carrier_ASR": "1",
                        "Source_pmal": "0.10",
                        "Adv_pmal": "0.01",
                        "Alignment_coverage": "1.0",
                        "Alignment_missing": "0",
                        "Target_L2": "2.0",
                        "Target_MAE": "1.0",
                        "Feature_backend": "nfstream",
                        "Feature_status": "ok",
                        "Flow_count": "10",
                        "Sanity_nonmonotonic": "0.0",
                        "Sanity_transport_missing": "0.0",
                        "Sanity_tcp_flag_invalid": "0.0",
                        "Sanity_tcp_seq_backwards": "0.0",
                    }
                ],
            )
            _write_csv(
                dataset_root / "nb15_stage2_support_table.csv",
                [
                    {
                        "Attack": "GLOBAL",
                        "Candidate mode": "per_sample_attack_score",
                        "Pullback α": "0.1",
                        "Pullback k": "5",
                        "Moment α": "0.1",
                        "Selected α": "-",
                        "Stage2 score": "0.8",
                        "FFD": "10.0",
                        "SWD": "0.2",
                    }
                ],
            )
            _write_csv(
                dataset_root / "unsw_transfer_ids_table.csv",
                [
                    {
                        "Attack": "GLOBAL",
                        "Transfer IDS": "logistic_small",
                        "Test Acc": "0.99",
                        "Test F1": "0.99",
                        "Adv ASR": "1.0",
                        "ΔASR vs main": "0.0",
                        "Seeds": "1",
                    }
                ],
            )
            _write_csv(dataset_root / "unsw_stage3_hard_carrier_table.csv", [])
            _write_csv(
                dataset_root / "unsw_stage3_baseline_policy_table.csv",
                [
                    {
                        "Attack": "GLOBAL",
                        "Method": "fgsm",
                        "Baseline group": "feature_only_control",
                        "Eval mode": "feature_only_random_remap_control",
                        "PCAP status": "evaluated",
                        "Skip reason": "",
                        "Deployability": "0.98",
                        "Replay ASR": "1.0",
                        "Target L2": "8.0",
                    }
                ],
            )
            _write_csv(
                dataset_root / "unsw_stage3_remap_distortion_table.csv",
                [{"Field": "dst_port_new", "MAE": "2.0", "RMSE": "3.0"}],
            )
            _write_csv(
                dataset_root / "unsw_stage3_protocol_legality_table.csv",
                [{"ValidFatal@0": "1.0", "TCP_Seq_Backwards_Rate": "0.0"}],
            )

            review_path, feishu_path = report.build_all_in_one(root, "nb15")

            self.assertTrue(review_path.exists())
            self.assertTrue(feishu_path.exists())
            full_report = (dataset_root / "NB15_FULL_REPORT_CN.md").read_text(encoding="utf-8-sig")
            short_report = review_path.read_text(encoding="utf-8-sig")
            self.assertIn("Stage2 Support-Aware Selection", full_report)
            self.assertIn("Transfer IDS", full_report)
            self.assertIn("Stage3 Baseline Realization Policy", full_report)
            self.assertIn("Stage2 Support-Aware Selection", short_report)


if __name__ == "__main__":
    unittest.main()
