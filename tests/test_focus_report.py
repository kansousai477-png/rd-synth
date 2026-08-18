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

import generate_reviewer_suite_focus_report_cn as report


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else ["placeholder"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


class FocusReportTest(unittest.TestCase):
    def test_pcap_fatal_uses_relative_sanity_regression(self) -> None:
        original = {
            "sanity_transport_missing_rate": "0",
            "sanity_tcp_seq_backwards_rate": "0.00042435875706214687",
            "sanity_tcp_flag_invalid_rate": "0",
        }
        unchanged = {
            "sanity_transport_missing_rate": "0",
            "sanity_tcp_seq_backwards_rate": "0.00042435875706214687",
            "sanity_tcp_flag_invalid_rate": "0",
        }
        regressed = {
            "sanity_transport_missing_rate": "0",
            "sanity_tcp_seq_backwards_rate": "0.001",
            "sanity_tcp_flag_invalid_rate": "0",
        }

        self.assertFalse(report._is_fatal_pcap(unchanged, original))
        self.assertTrue(report._is_fatal_pcap(regressed, original))

    def test_build_report_uses_paper_metrics_not_internal_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "nb15" / "main" / "seed_42" / "global"
            _write_csv(
                root / "nb15" / "main_runs.csv",
                [
                    {
                        "dataset": "nb15",
                        "attack_type": "GLOBAL",
                        "out_dir": str(out_dir),
                        "stage1_agreement": "0.98",
                        "stage1_baseline_agreement": "0.90",
                        "stage1_surrogate_query_count": "100",
                        "stage2_asr_oracle": "0.95",
                        "stage2_asr_surrogate": "0.94",
                        "stage2_norm_ffd": "12.0",
                        "stage2_norm_swd": "0.3",
                        "stage2_norm_advtomal_l2": "4.5",
                        "stage2_norm_corr_delta": "0.1",
                        "stage2_queries_per_success_oracle": "0.0",
                        "stage3_source_attack_success_rate": "0.2",
                        "stage3_adv_attack_success_rate": "0.8",
                        "stage3_source_flow_attack_success_rate": "0.1",
                        "stage3_adv_flow_attack_success_rate": "0.7",
                        "stage3_pcap_adv_prob_malicious_mean": "0.8",
                        "stage3_pcap_valid_fatal_rate": "0.0",
                        "stage3_pcap_target_l2_mean": "3.0",
                    }
                ],
            )
            _write_csv(
                root / "nb15" / "rq1_matrix_summary.csv",
                [{"attack_type": "GLOBAL", "ids_count": "6", "oracle_count": "6"}],
            )
            _write_csv(
                root / "nb15" / "stage2_attack_runs.csv",
                [
                    {
                        "attack_type": "DoS",
                        "stage2_eval_attack_rows": "10",
                        "asr_oracle": "0.91",
                        "asr_surrogate": "0.90",
                        "norm_FFD": "11.0",
                        "norm_SWD": "0.2",
                        "norm_AdvToMal_L2": "4.0",
                        "attack_score_queries_per_success_oracle": "0.0",
                    }
                ],
            )
            _write_csv(
                root / "nb15" / "main_stage2_baselines.csv",
                [
                    {
                        "attack_type": "GLOBAL",
                        "method": "fgsm",
                        "family": "baseline",
                        "baseline_level": "standard",
                        "asr_oracle": "0.70",
                        "asr_surrogate": "0.65",
                        "norm_ffd": "20.0",
                        "norm_swd": "0.5",
                        "norm_advtomal_l2": "3.0",
                    }
                ],
            )
            _write_csv(
                root / "nb15" / "main_stage3_baselines.csv",
                [
                    {
                        "attack_type": "GLOBAL",
                        "method": "amoeba_lite",
                        "baseline_group": "native_packet_comparable",
                        "evaluation_mode": "shared_backend_proxy",
                        "pcap_attack_success_rate": "0.6",
                        "deployability_score": "0.9",
                        "pcap_adv_prob_malicious_mean": "0.2",
                        "pcap_target_l2_mean": "4.0",
                        "pcap_status": "evaluated",
                    }
                ],
            )
            _write_csv(
                root / "nb15" / "ablation_runs.csv",
                [
                    {
                        "variant": "full",
                        "stage2_asr_oracle": "0.95",
                        "stage2_norm_ffd": "12.0",
                        "stage3_pcap_attack_success_rate": "0.8",
                        "stage3_adv_attack_success_rate": "0.8",
                        "stage3_adv_flow_attack_success_rate": "0.7",
                        "stage3_pcap_valid_fatal_rate": "0.0",
                    },
                    {
                        "variant": "backbone_gan",
                        "stage2_asr_oracle": "0.90",
                        "stage2_norm_ffd": "13.0",
                        "stage3_pcap_attack_success_rate": "0.7",
                        "stage3_adv_attack_success_rate": "0.7",
                        "stage3_adv_flow_attack_success_rate": "0.6",
                        "stage3_pcap_valid_fatal_rate": "0.1",
                    },
                ],
            )
            _write_csv(
                out_dir / "stage3" / "pcap_eval.csv",
                [
                    {
                        "source_name": "bad.pcap",
                        "is_original": "1",
                        "pred_label": "1",
                        "prob_malicious": "0.9",
                        "flow_count": "4",
                        "target_l2": "2.0",
                        "sanity_transport_missing_rate": "0",
                        "sanity_tcp_seq_backwards_rate": "0",
                        "sanity_tcp_flag_invalid_rate": "0",
                    },
                    {
                        "source_name": "bad.pcap",
                        "is_original": "0",
                        "pred_label": "1",
                        "prob_malicious": "0.8",
                        "flow_count": "4",
                        "target_l2": "2.0",
                        "sanity_transport_missing_rate": "0",
                        "sanity_tcp_seq_backwards_rate": "0",
                        "sanity_tcp_flag_invalid_rate": "0",
                    },
                ],
            )

            path = report.build_report(root, ["nb15", "2017"])
            text = path.read_text(encoding="utf-8-sig")

            self.assertIn("# 论文讨论版重点数据报告", text)
            self.assertNotIn("Decision Score", text)
            self.assertLess(text.index("## 核心结论"), text.index("## 指标速读"))
            self.assertIn("CIC NB15", text)
            self.assertIn("Stage2 生成阶段没有额外在线查询目标 IDS", text)
            self.assertIn("Stage1 Extraction Quality", text)
            self.assertIn("Stage2 Feature-Space Attack", text)
            self.assertIn("Stage2 Attack-Type Evidence", text)
            self.assertIn("Stage3 PCAP Evidence", text)
            self.assertIn("Baseline Comparison", text)
            self.assertIn("Ablation", text)
            self.assertIn("Failure Case Diagnostics", text)
            self.assertIn("Source PCAP Evasion", text)
            self.assertIn("Source p_mal Mean", text)
            self.assertIn("Adv Flow Evasion", text)
            self.assertIn("DoS", text)
            self.assertIn("fgsm", text)
            self.assertIn("amoeba_lite", text)
            self.assertIn("full", text)
            self.assertLess(
                text.index("| CIC NB15 | available | full |"), text.index("| CIC NB15 | available | backbone_gan |")
            )
            self.assertIn("| CIC-IDS2017 | missing | full | 0 |", text)
            self.assertIn("bad.pcap", text)
            self.assertIn("仍被检测", text)
            self.assertIn("**0.9500**", text)


if __name__ == "__main__":
    unittest.main()
