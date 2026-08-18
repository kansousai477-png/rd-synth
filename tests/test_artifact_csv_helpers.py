from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.utils.artifacts import save_array_csv, save_records_csv, save_training_log_csv


class ArtifactCsvHelpersTest(unittest.TestCase):
    def test_save_records_csv_writes_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rows.csv"
            save_records_csv(path, [{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}], fieldnames=["a", "b"])
            text = path.read_text(encoding="utf-8")
            self.assertIn("a,b", text)
            self.assertIn("1,2.5", text)

    def test_save_training_log_csv_infers_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "train.csv"
            save_training_log_csv(path, [{"epoch": 1, "loss": 0.1}, {"epoch": 2, "loss": 0.05, "acc": 0.9}])
            text = path.read_text(encoding="utf-8")
            self.assertIn("epoch,loss,acc", text)
            self.assertIn("2,0.05,0.9", text)

    def test_save_array_csv_writes_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "array.csv"
            save_array_csv(path, [[1, 2], [3, 4]], header=["x", "y"])
            text = path.read_text(encoding="utf-8")
            self.assertIn("x,y", text)
            self.assertIn("1,2", text)


if __name__ == "__main__":
    unittest.main()
