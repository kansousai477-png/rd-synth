from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.reviewer_suite import (  # noqa: E402
    DATASET_SPECS,
    build_run_config,
    load_yaml,
    resolve_profile_overrides,
    selected_attacks,
    summarize_workload,
)

DATASETS = ["nb15", "2017", "2018", "iot23"]


def _suite_cfg() -> dict:
    return load_yaml(ROOT / "configs" / "reviewer_suite.yaml")


def _base_cfg(dataset: str) -> dict:
    return load_yaml(ROOT / str(DATASET_SPECS[dataset]["base_config"]))


def _selected_attack_map(profile: str = "paper") -> dict[str, list[str]]:
    suite_cfg = _suite_cfg()
    max_attacks = int(resolve_profile_overrides(profile).get("max_attacks_per_dataset", 0))
    return {
        dataset: selected_attacks(
            dataset,
            suite_cfg=suite_cfg,
            base_cfg=_base_cfg(dataset),
            override_attacks=[],
            max_attacks=max_attacks,
        )
        for dataset in DATASETS
    }


class FullRunContractTest(unittest.TestCase):
    def test_paper_workload_is_bounded_and_expected(self) -> None:
        profile = resolve_profile_overrides("paper")
        attacks = _selected_attack_map("paper")
        workload = summarize_workload(
            selected_attacks=attacks,
            global_binary_datasets={dataset for dataset in DATASETS if DATASET_SPECS[dataset].get("global_binary")},
            seeds=list(profile["seeds"]),
            stage2_baselines=list(profile["stage2_baselines"]),
            ablation_variants=list(profile["ablation_variants"]),
            transfer_oracles=list(profile["transfer_ids"]),
            stage2_baselines_enabled=bool(profile["stage2_baselines_enabled"]),
            skip_transfer=False,
        )

        self.assertEqual(workload["dataset_attack_counts"], {"nb15": 6, "2017": 6, "2018": 7, "iot23": 5})
        self.assertEqual(workload["total_attacks"], 24)
        self.assertEqual(workload["main_runs"], 4)
        self.assertEqual(workload["pipeline_invocations"], 20)
        self.assertLessEqual(workload["pipeline_invocations"], 24)
        self.assertLessEqual(workload["stage2_baseline_runs"], 64)

    def test_all_global_main_configs_follow_reviewer_contract(self) -> None:
        attacks = _selected_attack_map("paper")
        profile = resolve_profile_overrides("paper")
        out_dirs: set[str] = set()
        with tempfile.TemporaryDirectory() as tmp_dir:
            for dataset in DATASETS:
                out_dir = Path(tmp_dir) / dataset / "main" / "seed_42" / "global"
                cfg = build_run_config(
                    base_cfg=_base_cfg(dataset),
                    attack="GLOBAL",
                    eval_attack_label="",
                    semantic_attack_labels=attacks[dataset],
                    seed=42,
                    out_dir=out_dir,
                    profile="paper",
                    stage2_baselines_enabled=True,
                    stage3_baselines_enabled=True,
                    stage2_baselines=list(profile["stage2_baselines"]),
                )

                project = cfg["project"]
                data = cfg["data"]
                stage1 = cfg["stage1"]
                stage2 = cfg["stage2"]
                stage3 = cfg["stage3"]
                out_dir_text = str(project["out_dir"])
                self.assertNotIn(out_dir_text, out_dirs)
                out_dirs.add(out_dir_text)
                self.assertEqual(project["attack_type"], "GLOBAL")
                self.assertTrue(str(project["stage1_shared_root"]).endswith(f"{dataset}/_shared/stage1/seed_42"))
                self.assertNotIn("eval_attack_label", project)
                self.assertNotIn("include_labels", data)
                self.assertNotIn("eval_attack_label", data)
                self.assertTrue(stage1["compute_matrix"])
                self.assertGreaterEqual(len(stage1["ids_names"]), 2)
                self.assertEqual(stage2["ids_name"], stage3["main_ids_name"])
                self.assertTrue(stage2["save_samples"])
                self.assertTrue(stage2["baselines"]["enable"])
                self.assertEqual(stage2["baselines"]["methods"], list(profile["stage2_baselines"]))
                self.assertEqual(stage3["pcap_dataset"], str(data["dataset"]))
                self.assertEqual(stage3["pcap_attack_label"], "GLOBAL")
                self.assertEqual(stage3["pcap_attack_labels"], attacks[dataset])
                self.assertEqual(stage3["pcap_source_selection_mode"], "best")
                self.assertEqual(stage3["pcap_source_sample_n"], 1)
                self.assertEqual(stage3["pcap_path"], "")
                self.assertTrue(stage3["pcap_eval_use_ids"])
                self.assertEqual(int(stage3["pcap_scan_limit"]), 32)
                self.assertGreater(int(stage3["pcap_scan_max_bytes"]), 0)
                self.assertTrue(str(stage3["pcap_out_dir"]).endswith("/stage3/pcap"))

    def test_speed_profiles_keep_stage3_carrier_budget_bounded(self) -> None:
        expected_limits = {"paper": 32, "standard": 12, "quick": 8}
        base_cfg = _base_cfg("2018")
        for profile_name, expected_limit in expected_limits.items():
            profile = resolve_profile_overrides(profile_name)
            cfg = build_run_config(
                base_cfg=base_cfg,
                attack="GLOBAL",
                eval_attack_label="",
                semantic_attack_labels=["Bot", "SQL Injection"],
                seed=42,
                out_dir=ROOT / "outputs" / "test_contract" / profile_name,
                profile=profile_name,
                stage2_baselines_enabled=bool(profile["stage2_baselines_enabled"]),
                stage3_baselines_enabled=bool(profile["stage3_baselines_enabled"]),
                stage2_baselines=list(profile["stage2_baselines"]),
                patch={"stage3": {"pcap_source_selection_mode": "all", "pcap_scan_limit": 0, "pcap_path": "x.pcap"}},
            )

            self.assertEqual(cfg["stage3"]["pcap_source_selection_mode"], "best")
            self.assertEqual(cfg["stage3"]["pcap_source_sample_n"], 1)
            self.assertEqual(cfg["stage3"]["pcap_path"], "")
            self.assertEqual(int(cfg["stage3"]["pcap_scan_limit"]), expected_limit)
            self.assertGreater(int(cfg["stage3"]["pcap_scan_max_bytes"]), 0)

    def test_cross_dataset_entrypoint_exposes_quality_gate_controls(self) -> None:
        python_exe = ROOT / "venv" / "Scripts" / "python.exe"
        proc = subprocess.run(
            [str(python_exe), "scripts/run_cross_dataset_suite.py", "--help"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        help_text = proc.stdout
        for flag in (
            "--estimate-only",
            "--prebuild-data",
            "--require-prebuilt-data",
            "--two-phase-stage3",
            "--skip-transfer",
        ):
            self.assertIn(flag, help_text)


if __name__ == "__main__":
    unittest.main()
