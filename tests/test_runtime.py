from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.runtime import configure_torch_runtime, load_stage_runtime, resolve_torch_device


class RuntimeHelpersTest(unittest.TestCase):
    def test_configure_torch_runtime_sets_thread_limits(self) -> None:
        with patch("rdsynth.pipeline.runtime.torch.set_num_threads") as set_num_threads:
            with patch("rdsynth.pipeline.runtime.torch.set_num_interop_threads") as set_num_interop_threads:
                with patch("rdsynth.pipeline.runtime.torch.cuda.is_available", return_value=False):
                    configure_torch_runtime({"num_threads": 3, "num_interop_threads": 2})

        set_num_threads.assert_called_once_with(3)
        set_num_interop_threads.assert_called_once_with(2)

    def test_configure_torch_runtime_strict_repro_uses_highest_precision(self) -> None:
        with patch("rdsynth.pipeline.runtime.torch.set_float32_matmul_precision") as set_precision:
            with patch("rdsynth.pipeline.runtime.torch.cuda.is_available", return_value=False):
                configure_torch_runtime({"strict_repro": True, "matmul_precision": "high"})
        set_precision.assert_called_once_with("highest")

    def test_configure_torch_runtime_sets_cublas_workspace_for_deterministic_cuda(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("rdsynth.pipeline.runtime.torch.cuda.is_available", return_value=True):
                with patch("rdsynth.pipeline.runtime.torch.set_float32_matmul_precision"):
                    configure_torch_runtime({"deterministic": True})
                    self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")

    def test_configure_torch_runtime_skips_cublas_workspace_for_nondeterministic_cuda(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("rdsynth.pipeline.runtime.torch.cuda.is_available", return_value=True):
                with patch("rdsynth.pipeline.runtime.torch.set_float32_matmul_precision"):
                    configure_torch_runtime({"deterministic": False})
                    self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)

    def test_resolve_torch_device_uses_auto_cuda_when_available(self) -> None:
        with patch("rdsynth.pipeline.runtime.torch.cuda.is_available", return_value=True):
            device = resolve_torch_device({"device": "auto"})
        self.assertEqual(device.type, "cuda")

    def test_resolve_torch_device_keeps_cpu_when_requested(self) -> None:
        device = resolve_torch_device({"device": "cpu"})
        self.assertEqual(device.type, "cpu")

    def test_load_stage_runtime_builds_context_and_initializes_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_dir = (tmp_path / "outputs").resolve()
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: '7'",
                        "  device: cpu",
                        f"  out_dir: {out_dir.as_posix()}",
                        "stage2: {}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("rdsynth.pipeline.runtime.configure_torch_runtime") as configure_runtime:
                with patch("rdsynth.pipeline.runtime.set_seed") as set_seed:
                    runtime = load_stage_runtime(config_path, "stage2")
                    self.assertTrue(runtime.out_dir.exists())

        self.assertEqual(runtime.config_path, config_path.resolve())
        self.assertEqual(runtime.stage_name, "stage2")
        self.assertEqual(runtime.project_cfg["seed"], "7")
        self.assertEqual(runtime.stage_cfg, {})
        self.assertEqual(runtime.seed, 7)
        self.assertEqual(runtime.device.type, "cpu")
        self.assertEqual(runtime.out_dir, out_dir / "stage2")
        configure_runtime.assert_called_once_with(runtime.project_cfg)
        set_seed.assert_called_once_with(7, deterministic=True)


if __name__ == "__main__":
    unittest.main()
