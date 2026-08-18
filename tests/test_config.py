from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.runner import build_stage_env, run_stage
from rdsynth.pipeline.stage2 import Stage2Settings
from rdsynth.pipeline.stage3 import Stage3Settings
from rdsynth.utils.config import load_yaml, optional_section, require_section
from rdsynth.utils.pipeline_config import apply_pipeline_defaults, prepare_pipeline_config


class ConfigUtilsTest(unittest.TestCase):
    def test_load_yaml_rejects_non_mapping_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            path.write_text("- item\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yaml(path)

    def test_load_yaml_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_yaml("does-not-exist.yaml")

    def test_require_and_optional_section(self) -> None:
        cfg = {"project": {"seed": 1}}
        self.assertEqual(require_section(cfg, "project")["seed"], 1)
        self.assertEqual(optional_section(cfg, "stage1"), {})
        with self.assertRaises(ValueError):
            require_section(cfg, "stage1")

    def test_pipeline_defaults_include_decision_selection_knobs(self) -> None:
        cfg = apply_pipeline_defaults({"project": {}, "stage1": {}, "stage2": {}, "stage3": {}}, cli_oracle="")
        self.assertTrue(cfg["stage1"]["save_eval_snapshot"])
        self.assertEqual(cfg["stage1"]["query_label_noise"], 0.0)
        self.assertEqual(cfg["stage1"]["extraction_rounds"], 1)
        self.assertTrue(cfg["stage2"]["save_intermediate_results"])
        self.assertEqual(cfg["stage2"]["generator_backbone"], "ddpm")
        self.assertEqual(cfg["stage2"]["surrogate_guidance_mode"], "embedding")
        self.assertEqual(cfg["stage2"]["selection_eval_every"], 5)
        self.assertEqual(cfg["stage2"]["selection_eval_samples"], 256)
        self.assertEqual(cfg["stage2"]["selection_batch_size"], 256)
        self.assertEqual(cfg["stage2"]["selection_mal_anchor_alpha"], 0.1)
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["candidate_selection"], "global_pareto")
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["iterative_rounds"], 1)
        self.assertTrue(cfg["stage3"]["save_intermediate_results"])
        self.assertTrue(cfg["stage3"]["protocol_auto_fix"])
        self.assertEqual(cfg["stage3"]["pcap_scan_max_bytes"], 0)
        self.assertFalse(cfg["stage3"]["pcap_feature_fail_closed"])
        self.assertFalse(cfg["stage3"]["pcap_feature_fail_on_partial_alignment"])
        self.assertFalse(cfg["project"]["strict_repro"])
        self.assertFalse(cfg["data"]["strict_ingest"])

    def test_prepare_pipeline_config_normalizes_project_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_text = (
                "project:\n"
                "  seed: '7'\n"
                "  device: cpu\n"
                "  out_dir: outputs/test\n"
                "  num_threads: '2'\n"
                "  stage_timeout_sec: '30'\n"
                "stage1: {}\n"
                "stage2: {}\n"
                "stage3: {}\n"
            )
            config_path.write_text(config_text, encoding="utf-8")
            cfg = prepare_pipeline_config(load_yaml(config_path), config_path)
            project = cfg["project"]
            self.assertEqual(project["seed"], 7)
            self.assertEqual(project["out_dir"], "outputs/debug/test")
            self.assertEqual(project["num_threads"], 2)
            self.assertEqual(project["stage_timeout_sec"], 30)
            self.assertIn("runtime", project)
            self.assertEqual(project["runtime"]["config_path"], str(config_path.resolve()))
            self.assertEqual(
                project["runtime"]["config_sha256"],
                hashlib.sha256(config_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(project["runtime"]["output_bucket"], "debug")
            self.assertFalse(project["runtime"]["spec_output_path_valid"])
            self.assertEqual(project["runtime"]["failed_out_dir"], "outputs/failed/test")
            self.assertTrue(project["deterministic"])

    def test_prepare_pipeline_config_defaults_device_to_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: '7'",
                        "  out_dir: outputs/test",
                        "stage1: {}",
                        "stage2: {}",
                        "stage3: {}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = prepare_pipeline_config(load_yaml(config_path), config_path)
            self.assertEqual(cfg["project"]["device"], "auto")

    def test_prepare_pipeline_config_keeps_spec_output_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: '9'",
                        "  out_dir: outputs/reviewer_suite/demo",
                        "stage1: {}",
                        "stage2: {}",
                        "stage3: {}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = prepare_pipeline_config(load_yaml(config_path), config_path)
            self.assertEqual(cfg["project"]["out_dir"], "outputs/reviewer_suite/demo")
            self.assertEqual(cfg["project"]["runtime"]["output_bucket"], "reviewer_suite")
            self.assertTrue(cfg["project"]["runtime"]["spec_output_path_valid"])
            self.assertEqual(cfg["project"]["runtime"]["failed_out_dir"], "outputs/failed/reviewer_suite/demo")

    def test_prepare_pipeline_config_strict_repro_enables_strict_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: '7'",
                        "  out_dir: outputs/test",
                        "  strict_repro: true",
                        "stage1: {}",
                        "stage2: {}",
                        "stage3: {}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = prepare_pipeline_config(load_yaml(config_path), config_path)
            self.assertTrue(cfg["project"]["strict_repro"])
            self.assertTrue(cfg["data"]["strict_ingest"])
            self.assertEqual(cfg["data"]["encoding_errors"], "strict")

    def test_build_stage_env_propagates_seed_and_threads(self) -> None:
        env = build_stage_env(
            Path("configs/demo.yaml"),
            {"seed": 42, "num_threads": 3},
        )
        self.assertEqual(env["RDSYNTH_CONFIG"], "configs/demo.yaml")
        self.assertEqual(env["PYTHONHASHSEED"], "42")
        self.assertEqual(env["OMP_NUM_THREADS"], "3")
        self.assertEqual(env["MKL_NUM_THREADS"], "3")

    def test_run_stage_inline_invokes_entrypoint_with_config(self) -> None:
        recorded: dict[str, str] = {}

        def fake_entrypoint(config_path: str) -> None:
            recorded["config_path"] = config_path
            recorded["env_config"] = os.environ.get("RDSYNTH_CONFIG", "")
            recorded["seed"] = os.environ.get("PYTHONHASHSEED", "")

        from unittest.mock import patch

        with patch("rdsynth.pipeline.runner._resolve_stage_entrypoint", return_value=fake_entrypoint):
            run_stage(
                "run_stage1.py",
                Path("configs/demo.yaml"),
                {"seed": 11, "num_threads": 1},
                stage_name="stage1",
                execution_mode="inline",
            )
        self.assertEqual(recorded["config_path"], "configs\\demo.yaml" if os.name == "nt" else "configs/demo.yaml")
        self.assertEqual(recorded["env_config"], "configs/demo.yaml")
        self.assertEqual(recorded["seed"], "11")

    def test_run_stage_inline_writes_failure_record(self) -> None:
        from unittest.mock import patch

        def fake_entrypoint(config_path: str) -> None:
            raise RuntimeError(f"bad config: {config_path}")

        with patch("rdsynth.pipeline.runner._resolve_stage_entrypoint", return_value=fake_entrypoint):
            with patch("rdsynth.pipeline.runner.write_failure_record") as write_failure:
                with self.assertRaisesRegex(RuntimeError, "bad config"):
                    run_stage(
                        "run_stage2.py",
                        Path("configs/demo.yaml"),
                        {
                            "seed": 11,
                            "num_threads": 1,
                            "runtime": {"failed_out_dir": "outputs/failed/debug/demo"},
                        },
                        stage_name="stage2",
                        execution_mode="inline",
                    )
        write_failure.assert_called_once()

    def test_stage2_settings_preserve_branch_specific_defaults(self) -> None:
        settings = Stage2Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
            }
        )
        self.assertEqual(settings.sample_init_latent, "benign_sample")
        self.assertEqual(settings.sample_init_editor, "benign_mean")
        self.assertTrue(settings.post_clip_norm_range)

    def test_stage3_settings_parse_search_alphas_and_oracle(self) -> None:
        settings = Stage3Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
                "pcap_search_alphas": "0.1, 0.2",
            },
            {"oracle_name": "mlp_small"},
        )
        self.assertEqual(settings.pcap_search_alphas, [0.1, 0.2])
        self.assertEqual(settings.oracle_name, "mlp_small")
        self.assertEqual(settings.ids_name, "pcap_ids")
        self.assertEqual(settings.remap_mode, "auto")
        self.assertEqual(settings.pcap_apply_fields[0], "mean_iat_ms")
        self.assertEqual(settings.pcap_search_rounds, 1)
        self.assertEqual(settings.pcap_target_source, "stage2_saved_samples")
        self.assertTrue(settings.pcap_compare_baselines)
        self.assertFalse(settings.pcap_eval_use_ids)


if __name__ == "__main__":
    unittest.main()
