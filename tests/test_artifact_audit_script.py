from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

import audit_artifacts  # noqa: E402


class ArtifactAuditScriptTest(unittest.TestCase):
    def test_audit_outputs_reports_missing_declared_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir) / "outputs"
            stage_dir = outputs_root / "debug" / "demo" / "stage2"
            stage_dir.mkdir(parents=True)
            (stage_dir / "metrics.json").write_text("{}", encoding="utf-8")
            (stage_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "stage": "stage2",
                        "config_path": "configs/demo.yaml",
                        "metadata": {
                            "run_id": "demo",
                            "git_commit": "",
                            "created_at": "2026-04-19T00:00:00+00:00",
                            "config_path": "configs/demo.yaml",
                            "config_hash": "abc",
                            "dataset": "toy",
                            "attack_type": "",
                            "target_model": "mlp_small",
                            "variant": "",
                            "seed": 1,
                            "stage": "stage2",
                            "rq": "RQ2/RQ3/RQ4",
                            "status": "success",
                            "failure_reason": "",
                        },
                        "outputs": {"metrics_json": "metrics.json", "state": "stage2.pt"},
                    }
                ),
                encoding="utf-8",
            )
            report = audit_artifacts.audit_outputs(outputs_root)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any(issue["kind"] == "missing_declared_outputs" for issue in report["issues"]))

    def test_audit_outputs_ignores_missing_volatile_cache_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            outputs_root = root / "outputs"
            stage_dir = outputs_root / "debug" / "demo" / "data_prep"
            stage_dir.mkdir(parents=True)
            (stage_dir / "metrics.json").write_text("{}", encoding="utf-8")
            (stage_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "stage": "data_prep",
                        "config_path": "configs/demo.yaml",
                        "metadata": {
                            "run_id": "demo",
                            "git_commit": "",
                            "created_at": "2026-04-19T00:00:00+00:00",
                            "config_path": "configs/demo.yaml",
                            "config_hash": "abc",
                            "dataset": "toy",
                            "attack_type": "",
                            "target_model": "mlp_small",
                            "variant": "",
                            "seed": 1,
                            "stage": "data_prep",
                            "rq": "Stress",
                            "status": "success",
                            "failure_reason": "",
                        },
                        "outputs": {
                            "metrics_json": "metrics.json",
                            "data_artifact_dir": "missing_artifacts",
                            "data_cache": str(root / ".cache" / "rdsynth_data_context" / "missing.pkl"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = audit_artifacts.audit_outputs(outputs_root)
            self.assertEqual(report["status"], "ok")

    def test_audit_outputs_warns_for_non_spec_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir) / "outputs"
            rogue_dir = outputs_root / "scratch"
            rogue_dir.mkdir(parents=True)
            report = audit_artifacts.audit_outputs(outputs_root)
            self.assertEqual(report["status"], "warn")
            self.assertTrue(any(issue["kind"] == "non_spec_output_bucket" for issue in report["issues"]))

    def test_script_writes_reports_and_returns_zero_for_clean_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir) / "outputs"
            pipeline_dir = outputs_root / "debug" / "demo" / "pipeline"
            pipeline_dir.mkdir(parents=True)
            (pipeline_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "demo",
                        "git_commit": "",
                        "created_at": "2026-04-19T00:00:00+00:00",
                        "config_path": "configs/demo.yaml",
                        "config_hash": "abc",
                        "dataset": "toy",
                        "attack_type": "",
                        "target_model": "mlp_small",
                        "variant": "",
                        "seed": 1,
                        "stage": "pipeline",
                        "rq": "RQ1-RQ5",
                        "status": "success",
                        "failure_reason": "",
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_artifacts.py"),
                    "--root",
                    str(outputs_root),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((outputs_root / "reports" / "artifact_audit_report.json").exists())
            self.assertTrue((outputs_root / "reports" / "artifact_audit_report.md").exists())


if __name__ == "__main__":
    unittest.main()
