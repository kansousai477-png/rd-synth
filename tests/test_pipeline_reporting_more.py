from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.reporting import (
    baseline_fieldnames,
    collect_baseline_summary_records,
    leaderboard_fieldnames,
    make_leaderboard_records,
    make_wide_record,
    overview_fieldnames,
    pcap_eval_summary,
    print_baseline_table,
    print_table,
    select_overview_rows,
    wide_fieldnames,
    write_dict_csv,
)


class PipelineReportingMoreTest(unittest.TestCase):
    def test_pcap_eval_summary_aggregates_original_and_adv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pcap_eval.csv"
            path.write_text(
                "pcap,prob_malicious,pred_label\norig_flow.pcap,0.90,1\nadv_0000.pcap,0.40,0\nadv_0001.pcap,0.20,0\n",
                encoding="utf-8",
            )
            summary = pcap_eval_summary(path, Path("orig_flow.pcap"))

        self.assertEqual(summary["pcap_orig_prob_malicious"], 0.9)
        self.assertAlmostEqual(summary["pcap_adv_prob_malicious_mean"], 0.3)
        self.assertAlmostEqual(summary["pcap_adv_prob_malicious_min"], 0.2)
        self.assertAlmostEqual(summary["pcap_adv_prob_malicious_max"], 0.4)
        self.assertAlmostEqual(summary["pcap_adv_pred_malicious_rate"], 0.0)

    def test_write_dict_csv_respects_extra_field_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "records.csv"
            write_dict_csv(
                path, [{"a": "1", "b": "2"}, {"a": "3", "c": "4"}], fieldnames=["a"], append_extra_fields=True
            )
            with path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["a"], "1")
            self.assertIn("b", rows[0])
            self.assertIn("c", rows[1])

            write_dict_csv(
                path,
                [{"a": "1", "b": "2"}],
                fieldnames=["a"],
                append_extra_fields=False,
            )
            with path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0].keys()), ["a"])
            self.assertEqual(rows[0]["a"], "1")

    def test_summary_and_leaderboard_helpers_cover_baselines(self) -> None:
        stage2_metrics = {
            "asr_oracle": 0.96,
            "norm_FFD": 0.20,
            "norm_AdvToMal_L2": 1.0,
            "baseline_global_random_asr_oracle": 0.80,
            "baseline_global_random_norm_FFD": 0.40,
            "baseline_global_random_norm_AdvToMal_L2": 3.0,
            "baseline_gpmt_lite_asr_oracle": 0.98,
            "baseline_gpmt_lite_norm_FFD": 0.10,
            "baseline_gpmt_lite_norm_AdvToMal_L2": 1.5,
        }
        stage3_metrics = {
            "baseline_gpmt_lite_paper_pcap_attack_success_rate": 0.9,
            "baseline_global_random_paper_pcap_attack_success_rate": 0.3,
        }
        records = collect_baseline_summary_records(stage2_metrics, stage3_metrics)
        self.assertEqual([row["baseline"] for row in records], ["global_random", "gpmt_lite"])
        self.assertIn("paper_pcap_attack_success_rate", records[0])

        leaderboard = make_leaderboard_records(stage2_metrics)
        self.assertEqual(leaderboard[0]["baseline"], "gpmt_lite")
        self.assertEqual(leaderboard[0]["attack_tier"], "strong")
        self.assertEqual(leaderboard[-1]["attack_tier"], "control")
        self.assertEqual(leaderboard[0]["rank"], "1")
        self.assertIn("asr_oracle", leaderboard[0])
        self.assertIn("baseline", baseline_fieldnames(records))
        self.assertIn("rank", leaderboard_fieldnames(leaderboard))

    def test_wide_and_overview_helpers_format_and_filter_rows(self) -> None:
        rows = [
            ("Data", "rows", "100"),
            ("Stage2", "asr_oracle", "0.95"),
            ("Custom", "other metric", "x"),
        ]
        record = make_wide_record(rows, {"project": {"out_dir": "out"}, "data": {"dataset": "toy"}}, "mlp_small")
        self.assertEqual(record["project_out_dir"], "out")
        self.assertEqual(record["data__rows"], "100")
        self.assertEqual(record["stage2__asr_oracle"], "0.95")
        self.assertEqual(wide_fieldnames(record)[:3], ["project_out_dir", "dataset", "oracle_name"])
        self.assertIn("stage2__asr_oracle", overview_fieldnames(record))

        selected = select_overview_rows(rows)
        self.assertEqual(selected, [("Data", "rows", "100"), ("Stage2", "asr_oracle", "0.95")])
        self.assertIn("Stage", print_table([("Stage2", "asr_oracle", "0.95")]))
        rendered = print_baseline_table([("1", "gpmt_lite", "strong", "0.98", "0.10")])
        self.assertIn("Baseline", rendered)


if __name__ == "__main__":
    unittest.main()
