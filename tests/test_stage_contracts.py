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

from rdsynth.pipeline.stage_contracts import (
    StageManifestSpec,
    VersionedArtifactSpec,
    build_stage_output_files,
    build_versioned_artifact_payload,
    collect_manifest_arrays,
    output_filename,
    save_stage_manifest_spec,
)


class StageContractsTest(unittest.TestCase):
    def test_output_filename_uses_basename(self) -> None:
        self.assertEqual(output_filename(Path("a/b/c.txt")), "c.txt")
        self.assertEqual(output_filename("x\\y\\z.bin"), "z.bin")
        self.assertEqual(output_filename(Path("C:/tmp/demo.bin")), "C:\\tmp\\demo.bin")

    def test_build_stage_output_files_includes_base_and_optional_outputs(self) -> None:
        outputs = build_stage_output_files(
            primary_artifact_key="state",
            primary_artifact_name=Path("models/stage2.pt"),
            extra_outputs={
                "pareto": Path("runs/pareto.csv"),
                "skip_me": None,
            },
        )

        self.assertEqual(
            outputs,
            {
                "config": "config.yaml",
                "metrics_json": "metrics.json",
                "metrics_csv": "metrics.csv",
                "state": "stage2.pt",
                "pareto": "pareto.csv",
            },
        )

    def test_collect_manifest_arrays_omits_none_values(self) -> None:
        arrays = collect_manifest_arrays(
            {
                "adv_pre": np.zeros((2, 2), dtype=np.float32),
                "adv_norm": None,
                "meta": {"a": 1},
            }
        )

        self.assertIn("adv_pre", arrays)
        self.assertIn("meta", arrays)
        self.assertNotIn("adv_norm", arrays)

    def test_save_stage_manifest_spec_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = save_stage_manifest_spec(
                StageManifestSpec(
                    stage_name="stageX",
                    out_dir=Path(tmpdir),
                    config_path="configs/demo.yaml",
                    inputs={"rows": 10},
                    outputs={"state": "stageX.pt"},
                    arrays={"adv": np.zeros((2, 2), dtype=np.float32)},
                    metrics={"score": 0.5},
                )
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["stage"], "stageX")
            self.assertEqual(payload["outputs"]["state"], "stageX.pt")
            self.assertEqual(payload["metrics"]["score"], 0.5)

    def test_build_versioned_artifact_payload_supports_optional_fields(self) -> None:
        payload = build_versioned_artifact_payload(
            VersionedArtifactSpec(
                fields={"name": "demo"},
                optional_fields={"value": 3, "skip": None},
            )
        )

        self.assertEqual(payload["artifact_version"], 1)
        self.assertEqual(payload["name"], "demo")
        self.assertEqual(payload["value"], 3)
        self.assertNotIn("skip", payload)

    def test_build_versioned_artifact_payload_can_emit_array_version(self) -> None:
        payload = build_versioned_artifact_payload(
            VersionedArtifactSpec(
                version_as_array=True,
                fields={"feature_names": np.asarray(["a", "b"])},
            )
        )

        self.assertTrue(isinstance(payload["artifact_version"], np.ndarray))
        self.assertEqual(payload["artifact_version"].shape, ())


if __name__ == "__main__":
    unittest.main()
