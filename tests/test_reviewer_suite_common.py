from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.reviewer_suite import build_run_config  # noqa: E402


class ReviewerSuiteCommonTest(unittest.TestCase):
    def test_build_run_config_preserves_configured_pcap_path(self) -> None:
        base_cfg = {
            "project": {"seed": 42, "out_dir": "outputs/base"},
            "data": {"benign_labels": ["Benign"]},
            "stage1": {},
            "stage2": {"baselines": {}},
            "stage3": {
                "pcap_path": "data/PCAPs/example.pcap",
                "pcap_scan_dir": "data/PCAPs",
                "pcap_scan_limit": 0,
            },
        }
        out_dir = ROOT / "outputs" / "test_suite_cfg"
        cfg = build_run_config(
            base_cfg=base_cfg,
            attack="Bot",
            seed=7,
            out_dir=out_dir,
            profile="paper",
            stage2_baselines_enabled=True,
            stage3_baselines_enabled=True,
            stage2_baselines=["identity"],
        )
        self.assertEqual(cfg["stage3"]["pcap_path"], "")
        self.assertEqual(cfg["stage3"]["pcap_scan_limit"], 32)
        self.assertEqual(cfg["stage3"]["pcap_source_selection_mode"], "best")
        self.assertEqual(cfg["stage3"]["pcap_source_sample_n"], 1)
        self.assertTrue(cfg["stage3"]["pcap_compare_baselines"])

    def test_build_run_config_records_global_semantic_attack_labels(self) -> None:
        base_cfg = {
            "project": {"seed": 42, "out_dir": "outputs/base"},
            "data": {"dataset": "cic_ids2018", "benign_labels": ["Benign"]},
            "stage1": {},
            "stage2": {"baselines": {}},
            "stage3": {"pcap_scan_limit": 0},
            "oracle_models": [{"name": "mlp_small"}],
        }
        out_dir = ROOT / "outputs" / "test_suite_cfg_global"
        cfg = build_run_config(
            base_cfg=base_cfg,
            attack="GLOBAL",
            eval_attack_label="",
            semantic_attack_labels=["DDOS attack-HOIC", "SQL Injection"],
            seed=42,
            out_dir=out_dir,
            profile="paper",
            stage2_baselines_enabled=True,
            stage3_baselines_enabled=True,
            stage2_baselines=["identity"],
        )

        self.assertEqual(cfg["stage3"]["pcap_attack_label"], "GLOBAL")
        self.assertEqual(cfg["stage3"]["pcap_attack_labels"], ["DDOS attack-HOIC", "SQL Injection"])
        self.assertEqual(cfg["stage3"]["pcap_scan_limit"], 32)
        self.assertEqual(cfg["stage3"]["pcap_source_selection_mode"], "best")

    def test_build_run_config_reapplies_stage3_policy_after_patch(self) -> None:
        base_cfg = {
            "project": {"seed": 42, "out_dir": "outputs/base"},
            "data": {"dataset": "cic_ids2018", "benign_labels": ["Benign"]},
            "stage1": {},
            "stage2": {"baselines": {}},
            "stage3": {"pcap_path": "data/PCAPs/example.pcap", "pcap_scan_limit": 32},
            "oracle_models": [{"name": "mlp_small"}],
        }
        out_dir = ROOT / "outputs" / "test_suite_cfg_policy"
        cfg = build_run_config(
            base_cfg=base_cfg,
            attack="GLOBAL",
            eval_attack_label="",
            semantic_attack_labels=["Bot"],
            seed=42,
            out_dir=out_dir,
            profile="paper",
            stage2_baselines_enabled=True,
            stage3_baselines_enabled=True,
            stage2_baselines=["identity"],
            patch={"stage3": {"pcap_source_selection_mode": "all", "pcap_source_sample_n": 0, "pcap_scan_limit": 0}},
        )

        self.assertEqual(cfg["stage3"]["pcap_source_selection_mode"], "best")
        self.assertEqual(cfg["stage3"]["pcap_source_sample_n"], 1)
        self.assertEqual(cfg["stage3"]["pcap_scan_limit"], 32)
        self.assertEqual(cfg["stage3"]["pcap_path"], "")

    def test_build_run_config_allows_bounded_top_hard_carrier_diagnostics(self) -> None:
        base_cfg = {
            "project": {"seed": 42, "out_dir": "outputs/base"},
            "data": {"dataset": "cic_ids2018", "benign_labels": ["Benign"]},
            "stage1": {},
            "stage2": {"baselines": {}},
            "stage3": {"pcap_path": "data/PCAPs/example.pcap", "pcap_scan_limit": 32},
            "oracle_models": [{"name": "mlp_small"}],
        }
        out_dir = ROOT / "outputs" / "test_suite_cfg_top_hard"
        cfg = build_run_config(
            base_cfg=base_cfg,
            attack="GLOBAL",
            eval_attack_label="",
            semantic_attack_labels=["Bot"],
            seed=42,
            out_dir=out_dir,
            profile="paper",
            stage2_baselines_enabled=True,
            stage3_baselines_enabled=True,
            stage2_baselines=["identity"],
            pcap_source_selection_mode="top_hard",
            pcap_source_sample_n=3,
        )

        self.assertEqual(cfg["stage3"]["pcap_source_selection_mode"], "top_hard")
        self.assertEqual(cfg["stage3"]["pcap_source_sample_n"], 3)
        self.assertEqual(cfg["stage3"]["pcap_path"], "")


if __name__ == "__main__":
    unittest.main()
