from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.utils.artifacts import (
    build_artifact_metadata,
    save_stage_manifest,
    summarize_array,
    write_failure_record,
)


class ArtifactUtilsTest(unittest.TestCase):
    def test_summarize_array_reports_basic_stats(self) -> None:
        summary = summarize_array(np.asarray([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32))
        self.assertEqual(summary["shape"], [2, 2])
        self.assertEqual(summary["size"], 4)
        self.assertAlmostEqual(summary["nan_rate"], 0.25)
        self.assertAlmostEqual(summary["min"], 1.0)
        self.assertAlmostEqual(summary["max"], 3.0)

    def test_save_stage_manifest_writes_json_with_array_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            config_path = out_dir / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: 42",
                        f"  out_dir: {out_dir.as_posix()}",
                        "data:",
                        "  dataset: toyset",
                        "stage2:",
                        "  oracle_name: mlp_small",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            path = save_stage_manifest(
                stage_name="stage2",
                out_dir=out_dir,
                config_path=config_path,
                inputs={"oracle_name": "mlp_small"},
                outputs={"metrics_json": "metrics.json"},
                arrays={"adv_pre": np.zeros((3, 4), dtype=np.float32)},
                metrics={"stage2_decision_score": 0.9},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["stage"], "stage2")
            self.assertEqual(payload["metadata"]["dataset"], "toyset")
            self.assertEqual(payload["metadata"]["seed"], 42)
            self.assertEqual(payload["metadata"]["target_model"], "mlp_small")
            self.assertEqual(payload["metadata"]["rq"], "RQ2/RQ3/RQ4")
            self.assertEqual(payload["inputs"]["oracle_name"], "mlp_small")
            self.assertEqual(payload["outputs"]["metrics_json"], "metrics.json")
            self.assertEqual(payload["arrays"]["adv_pre"]["shape"], [3, 4])
            self.assertAlmostEqual(payload["metrics"]["stage2_decision_score"], 0.9)

    def test_build_artifact_metadata_infers_attack_type_from_include_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: 7",
                        "  out_dir: outputs/reviewer_suite/demo/full",
                        "data:",
                        "  dataset: cic_ids2018",
                        "  include_labels: [BENIGN, SQLi]",
                        "stage3:",
                        "  oracle_name: cnn",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            metadata = build_artifact_metadata(config_path=config_path, stage_name="stage3")
            self.assertEqual(metadata["attack_type"], "SQLi")
            self.assertEqual(metadata["target_model"], "cnn")
            self.assertEqual(metadata["stage"], "stage3")
            self.assertEqual(metadata["rq"], "RQ5")

    def test_write_failure_record_uses_failed_out_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: 1",
                        "  out_dir: outputs/debug/demo",
                        "  runtime:",
                        f"    failed_out_dir: {(root / 'outputs' / 'failed' / 'debug' / 'demo').as_posix()}",
                        "data:",
                        "  dataset: toy",
                        "stage2: {}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            project_cfg = {
                "seed": 1,
                "out_dir": "outputs/debug/demo",
                "runtime": {
                    "failed_out_dir": str((root / "outputs" / "failed" / "debug" / "demo").as_posix()),
                },
            }
            record_path = write_failure_record(
                project_cfg=project_cfg,
                config_path=config_path,
                stage_name="stage2",
                error=RuntimeError("boom"),
            )
            self.assertIsNotNone(record_path)
            payload = json.loads(Path(record_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["status"], "failed")
            self.assertEqual(payload["metadata"]["failure_reason"], "boom")
            self.assertEqual(payload["stage"], "stage2")


if __name__ == "__main__":
    unittest.main()
