from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_analysis import print_stage2_metric_tables


class Stage2AnalysisMoreTest(unittest.TestCase):
    def test_print_stage2_metric_tables_covers_norm_and_denorm_modes(self) -> None:
        metrics_norm = SimpleNamespace(
            as_dict=lambda: {
                "FFD": 0.1,
                "SWD": 0.2,
                "Coverage@1": 0.3,
                "Corr螖": 0.4,
                "PairDist-KS": 0.5,
            }
        )
        metrics_denorm = SimpleNamespace(
            as_dict=lambda: {
                "Violation_Range": 0.1,
                "Violation_NonNeg": 0.2,
                "Violation_Integer": 0.3,
            }
        )
        metrics_payload = {
            "denorm_nan_rate": 0.01,
            "denorm_inf_rate": 0.02,
            "asr_surrogate": 0.8,
            "adv_prob_malicious_mean": 0.3,
            "mal_prob_malicious_mean": 0.7,
            "asr_oracle": 0.9,
            "adv_prob_malicious_mean_oracle": 0.2,
            "mal_prob_malicious_mean_oracle": 0.8,
        }
        with patch("builtins.print") as print_mock:
            print_stage2_metric_tables(
                metrics_payload=metrics_payload,
                metrics_norm=metrics_norm,
                adv_ben_l2=1.0,
                adv_mal_l2=2.0,
                eval_denorm=True,
                metrics_denorm=metrics_denorm,
                adv_ben_l2_denorm=3.0,
                adv_mal_l2_denorm=4.0,
            )
        text = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("metrics (normalized)", text)
        self.assertIn("metrics (denormalized)", text)
        self.assertIn("surrogate attack success", text)
        self.assertIn("oracle attack success", text)
