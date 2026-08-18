from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.utils.paper_metrics import add_paper_attack_metrics, add_paper_pcap_metrics


class PaperMetricsTest(unittest.TestCase):
    def test_attack_metrics_derive_dr_and_eir(self) -> None:
        payload: dict[str, float] = {}
        add_paper_attack_metrics(
            payload,
            prefix="baseline_x_",
            asr=0.60,
            orig_benign_rate=0.10,
            adv_prob_malicious=0.25,
            ffd=1.2,
            swd=0.7,
            c2st_auc=0.4,
            c2st_acc=0.6,
            adv_to_ben_l2=2.0,
            adv_to_mal_l2=1.0,
            runtime_sec=0.5,
        )
        self.assertAlmostEqual(payload["baseline_x_paper_attack_success_rate"], 0.60)
        self.assertAlmostEqual(payload["baseline_x_paper_detection_rate"], 0.40)
        self.assertAlmostEqual(payload["baseline_x_paper_evasion_increase_rate"], 1.0 - 0.40 / 0.90)
        self.assertAlmostEqual(payload["baseline_x_paper_concealment_proxy"], 0.75)
        self.assertAlmostEqual(payload["baseline_x_paper_timeliness_sec"], 0.5)

    def test_pcap_metrics_derive_dr_and_eir(self) -> None:
        payload: dict[str, float] = {}
        add_paper_pcap_metrics(
            payload,
            prefix="baseline_y_",
            adv_pred_malicious_rate=0.30,
            orig_pred_malicious_rate=1.0,
            adv_prob_malicious=0.20,
            target_l2=1.5,
            target_mae=0.4,
            alignment_coverage=0.8,
            runtime_sec=0.25,
            pcaps_per_sec=4.0,
            packets_per_sec=4000.0,
        )
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_attack_success_rate"], 0.70)
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_detection_rate"], 0.30)
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_evasion_increase_rate"], 0.70)
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_concealment_proxy"], 0.80)
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_fidelity_target_l2"], 1.5)
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_alignment_coverage"], 0.8)
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_timeliness_sec"], 0.25)
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_pcaps_per_sec"], 4.0)
        self.assertAlmostEqual(payload["baseline_y_paper_pcap_packets_per_sec"], 4000.0)

    def test_eir_is_nan_when_original_detection_is_zero(self) -> None:
        payload: dict[str, float] = {}
        add_paper_attack_metrics(payload, asr=0.5, orig_benign_rate=1.0)
        self.assertTrue(math.isnan(payload["paper_evasion_increase_rate"]))


if __name__ == "__main__":
    unittest.main()
