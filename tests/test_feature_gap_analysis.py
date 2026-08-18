from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from analyze_feature_gap import write_outputs


class FeatureGapAnalysisTest(unittest.TestCase):
    def test_write_outputs_summarizes_stage2_stage3_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "stage2").mkdir()
            (root / "stage3").mkdir()
            (root / "stage2" / "metrics.json").write_text(
                json.dumps({"asr_oracle": 1.0, "adv_prob_malicious_mean_oracle": 0.05, "norm_FFD": 12.0}),
                encoding="utf-8",
            )
            (root / "stage3" / "metrics.json").write_text(
                json.dumps(
                    {
                        "paper_pcap_attack_success_rate": 0.0,
                        "pcap_target_l2_mean": 25.0,
                        "pcap_eval_avg_alignment": 0.9,
                        "stage3_evidence_block_reason": "target_distance_outlier",
                    }
                ),
                encoding="utf-8",
            )
            (root / "stage3" / "pcap_eval.csv").write_text(
                "\n".join(
                    [
                        "pcap,is_original,prob_malicious,pred_label,target_l2,alignment_coverage,alignment_missing",
                        "src.pcap,1,0.9,1,,0.9,1",
                        "adv.pcap,0,0.7,1,25.0,0.9,1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            csv_path, md_path = write_outputs(root, root / "analysis")

            rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
            metrics = {row["metric"]: row["value"] for row in rows}
            self.assertEqual(metrics["stage2_asr_oracle"], "1.000000")
            self.assertEqual(metrics["stage3_pmal_delta_adv_minus_source"], "-0.200000")
            text = md_path.read_text(encoding="utf-8")
            self.assertIn("feature-space evasion is strong", text)


if __name__ == "__main__":
    unittest.main()
