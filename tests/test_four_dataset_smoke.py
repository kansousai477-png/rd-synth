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

import run_four_dataset_smoke as smoke  # noqa: E402


class FourDatasetSmokeTest(unittest.TestCase):
    def test_generated_smoke_configs_are_bounded_for_all_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir)
            config_paths = smoke.generate_configs(smoke.DEFAULT_DATASETS, out_root=out_root, seed=42)

            self.assertEqual(len(config_paths), 4)
            for dataset, cfg_path in zip(smoke.DEFAULT_DATASETS, config_paths):
                cfg = smoke.load_yaml(cfg_path)
                self.assertEqual(cfg["project"]["stage_timeout_sec"], 300)
                self.assertLessEqual(int(cfg["data"]["max_rows"]), 600)
                self.assertLessEqual(int(cfg["data"]["max_rows_per_label"]), 120)
                self.assertEqual(len(cfg["ids_models"]), 1)
                self.assertEqual(cfg["stage1"]["ids_names"], ["mlp_small"])
                self.assertFalse(cfg["stage1"]["compute_matrix"])
                self.assertFalse(cfg["stage1"]["compare_baseline"])
                self.assertFalse(cfg["stage2"]["baselines"]["enable"])
                self.assertEqual(cfg["stage2"]["baselines"]["methods"], [])
                self.assertFalse(cfg["stage2"]["attack_slice_eval_enabled"])
                self.assertEqual(int(cfg["stage2"]["eval_samples"]), 64)
                self.assertEqual(cfg["stage3"]["pcap_source_selection_mode"], "best")
                self.assertEqual(cfg["stage3"]["pcap_source_sample_n"], 1)
                self.assertEqual(cfg["stage3"]["pcap_path"], "")
                self.assertLessEqual(int(cfg["stage3"]["pcap_scan_limit"]), 4)
                self.assertLessEqual(int(cfg["stage3"]["pcap_scan_max_bytes"]), 1024 * 1024)
                self.assertEqual(int(cfg["stage3"]["pcap_apply_n"]), 1)
                self.assertFalse(cfg["stage3"]["pcap_compare_baselines"])
                self.assertTrue(cfg["stage3"]["pcap_attack_labels"], dataset)

    def test_validate_smoke_outputs_rejects_pcap_error_and_missing_skip_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir)
            run_root = out_root / "nb15" / "main" / "seed_42" / "global"
            (run_root / "pipeline").mkdir(parents=True)
            (run_root / "stage3").mkdir(parents=True)
            summary_path = run_root / "pipeline" / "summary_all_metrics.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "stage3_pcap__pcap_modified",
                        "stage3_pcap__pcap_skip_reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "stage3_pcap__pcap_modified": "False",
                        "stage3_pcap__pcap_skip_reason": "",
                    }
                )
            (run_root / "stage3" / "metrics.json").write_text(
                json.dumps({"pcap_error": "boom"}, ensure_ascii=False),
                encoding="utf-8",
            )

            rows = smoke.validate_smoke_outputs(["nb15"], out_root=out_root, seed=42)

        self.assertEqual(rows[0].status, "fail")
        self.assertIn("pcap_error=boom", rows[0].detail)
        self.assertIn("pcap_not_modified_without_skip_reason", rows[0].detail)

    def test_validate_smoke_outputs_rejects_explicit_not_modified_skip_reason_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir)
            run_root = out_root / "2017" / "main" / "seed_42" / "global"
            (run_root / "pipeline").mkdir(parents=True)
            (run_root / "stage3").mkdir(parents=True)
            with (run_root / "pipeline" / "summary_all_metrics.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "stage3_pcap__pcap_modified",
                        "stage3_pcap__pcap_skip_reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "stage3_pcap__pcap_modified": "False",
                        "stage3_pcap__pcap_skip_reason": "source_already_evasive",
                    }
                )
            (run_root / "stage3" / "metrics.json").write_text("{}", encoding="utf-8")

            rows = smoke.validate_smoke_outputs(["2017"], out_root=out_root, seed=42)

        self.assertEqual(rows[0].status, "fail")
        self.assertIn("pcap_not_modified=source_already_evasive", rows[0].detail)

    def test_validate_smoke_outputs_can_allow_explicit_not_modified_skip_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir)
            run_root = out_root / "2017" / "main" / "seed_42" / "global"
            (run_root / "pipeline").mkdir(parents=True)
            (run_root / "stage3").mkdir(parents=True)
            with (run_root / "pipeline" / "summary_all_metrics.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "stage3_pcap__pcap_modified",
                        "stage3_pcap__pcap_skip_reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "stage3_pcap__pcap_modified": "False",
                        "stage3_pcap__pcap_skip_reason": "source_already_evasive",
                    }
                )
            (run_root / "stage3" / "metrics.json").write_text("{}", encoding="utf-8")

            rows = smoke.validate_smoke_outputs(["2017"], out_root=out_root, seed=42, allow_pcap_skip=True)

        self.assertEqual(rows[0].status, "pass")


if __name__ == "__main__":
    unittest.main()
