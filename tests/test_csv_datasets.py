from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.data.csv_datasets import analyze_csv_collection, load_csv_dataset, resolve_dataset_profile


class CsvDatasetsTest(unittest.TestCase):
    def test_load_csv_dataset_merges_intersection_and_binary_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "part1.csv").write_text(
                "Dst Port,Flow Duration,Label\n80,10,BENIGN\n443,20,DoS\n",
                encoding="utf-8",
            )
            (root / "part2.csv").write_text(
                "Flow ID,Dst Port,Flow Duration,Label\n"
                "abc,53,5,BENIGN\n"
                "Flow ID,Dst Port,Flow Duration,Label\n"
                "def,8080,7,PortScan\n",
                encoding="utf-8",
            )

            features, labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*.csv",
                label_col="Label",
                task="binary",
                benign_labels=("BENIGN", "Benign"),
                drop_cols=(),
                merge_strategy="intersection",
            )

            self.assertEqual(list(features.columns), ["Dst Port", "Flow Duration"])
            self.assertEqual(features.shape, (4, 2))
            np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1], dtype=np.int64))

    def test_load_csv_dataset_supports_multiclass_and_max_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "part1.csv").write_text(
                "Feature A,Label\n1,BENIGN\n2,PortScan\n",
                encoding="utf-8",
            )
            (root / "part2.csv").write_text(
                "Feature A,Label\n3,DoS Hulk\n4,BENIGN\n",
                encoding="utf-8",
            )

            features, labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*.csv",
                label_col="Label",
                task="multiclass",
                benign_labels=("BENIGN",),
                max_rows=3,
            )

            self.assertEqual(features.shape, (3, 1))
            self.assertEqual(sorted(np.unique(labels).tolist()), [0, 1, 2])

    def test_load_csv_dataset_distributes_max_rows_across_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.csv").write_text(
                "Feature A,Label\n1,BENIGN\n2,BENIGN\n3,BENIGN\n",
                encoding="utf-8",
            )
            (root / "b.csv").write_text(
                "Feature A,Label\n4,Attack\n5,Attack\n6,Attack\n",
                encoding="utf-8",
            )

            _, labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*.csv",
                label_col="Label",
                task="binary",
                benign_labels=("BENIGN",),
                max_rows=2,
            )

            np.testing.assert_array_equal(labels, np.array([0, 1], dtype=np.int64))

    def test_analyze_csv_collection_reports_intersection_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.csv").write_text("A,B,Label\n1,2,BENIGN\n", encoding="utf-8")
            (root / "b.csv").write_text("B,C,Label\n2,3,Attack\n", encoding="utf-8")

            report = analyze_csv_collection(
                csv_paths=[root / "a.csv", root / "b.csv"],
                label_col="Label",
                drop_cols=(),
                merge_strategy="intersection",
            )

            self.assertEqual(report["selected_columns"], ["B"])
            self.assertEqual(report["file_count"], 2)

    def test_resolve_dataset_profile_supports_cic_dataset_selection(self) -> None:
        profile = resolve_dataset_profile({"dataset": "cic_ids2018"})
        self.assertEqual(profile.name, "cic_ids2018")
        self.assertEqual(profile.csv_dir, "data/CIC-IDS2018-CSV")
        self.assertEqual(profile.label_col, "Label")

    def test_resolve_dataset_profile_supports_iot2023_selection(self) -> None:
        profile = resolve_dataset_profile({"dataset": "cic_iot2023"})
        self.assertEqual(profile.name, "cic_iot2023")
        self.assertEqual(profile.csv_dir, "data/CIC_IOT_Dataset2023/CSV")
        self.assertEqual(profile.label_source, "parent_dir")
        self.assertEqual(profile.benign_labels, ("Benign_Final",))

    def test_load_csv_dataset_supports_parent_directory_labels_and_label_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benign_dir = root / "Benign_Final"
            attack_dir = root / "Recon-PortScan"
            benign_dir.mkdir(parents=True)
            attack_dir.mkdir(parents=True)
            (benign_dir / "a.csv").write_text(
                "Feature A,Feature B\n1,10\n2,20\n",
                encoding="utf-8",
            )
            (attack_dir / "b.csv").write_text(
                "Feature A,Feature B\n3,30\n4,40\n",
                encoding="utf-8",
            )

            features, labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*/*.csv",
                label_source="parent_dir",
                task="binary",
                benign_labels=("Benign_Final",),
                include_labels=("Benign_Final", "Recon-PortScan"),
            )

            self.assertEqual(features.shape, (4, 2))
            np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1], dtype=np.int64))

    def test_load_csv_dataset_parent_directory_intersection_uses_common_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benign_dir = root / "Benign_Final"
            attack_dir = root / "Recon-PortScan"
            benign_dir.mkdir(parents=True)
            attack_dir.mkdir(parents=True)
            (benign_dir / "a.csv").write_text(
                "Feature A,Feature B,Only Benign\n1,10,100\n2,20,200\n",
                encoding="utf-8",
            )
            (attack_dir / "b.csv").write_text(
                "Feature A,Feature B,Only Attack\n3,30,300\n4,40,400\n",
                encoding="utf-8",
            )

            features, labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*/*.csv",
                label_source="parent_dir",
                task="binary",
                benign_labels=("Benign_Final",),
                merge_strategy="intersection",
            )

            self.assertEqual(list(features.columns), ["Feature A", "Feature B"])
            self.assertEqual(features.shape, (4, 2))
            np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1], dtype=np.int64))

    def test_load_csv_dataset_parent_directory_reference_keeps_reference_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benign_dir = root / "Benign_Final"
            attack_dir = root / "Recon-PortScan"
            benign_dir.mkdir(parents=True)
            attack_dir.mkdir(parents=True)
            (benign_dir / "a.csv").write_text(
                "Feature A,Feature B,Only Benign\n1,10,100\n2,20,200\n",
                encoding="utf-8",
            )
            (attack_dir / "b.csv").write_text(
                "Feature A,Feature B,Only Attack\n3,30,300\n4,40,400\n",
                encoding="utf-8",
            )

            features, _ = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*/*.csv",
                label_source="parent_dir",
                task="binary",
                benign_labels=("Benign_Final",),
                merge_strategy="reference",
            )

            self.assertEqual(list(features.columns), ["Feature A", "Feature B", "Only Benign"])
            self.assertTrue(np.isnan(features.iloc[2]["Only Benign"]))
            self.assertTrue(np.isnan(features.iloc[3]["Only Benign"]))

    def test_load_csv_dataset_caps_rows_per_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benign_dir = root / "Benign_Final"
            attack_dir = root / "Recon-PortScan"
            benign_dir.mkdir(parents=True)
            attack_dir.mkdir(parents=True)
            (benign_dir / "a.csv").write_text(
                "Feature A\n1\n2\n3\n4\n",
                encoding="utf-8",
            )
            (attack_dir / "b.csv").write_text(
                "Feature A\n5\n6\n7\n8\n",
                encoding="utf-8",
            )

            _, labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*/*.csv",
                label_source="parent_dir",
                task="binary",
                benign_labels=("Benign_Final",),
                max_rows_per_label=2,
                seed=7,
            )

            counts = np.bincount(labels)
            self.assertEqual(counts.tolist(), [2, 2])

    def test_load_csv_dataset_label_quota_streaming_avoids_head_only_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "single.csv").write_text(
                "Feature A,Label\n"
                "1,BENIGN\n"
                "2,BENIGN\n"
                "3,BENIGN\n"
                "4,BENIGN\n"
                "5,BENIGN\n"
                "6,Attack\n"
                "7,Attack\n"
                "8,Attack\n"
                "9,Attack\n",
                encoding="utf-8",
            )

            features, labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*.csv",
                label_col="Label",
                task="binary",
                benign_labels=("BENIGN",),
                include_labels=("BENIGN", "Attack"),
                max_rows=4,
                max_rows_per_label=2,
                csv_chunk_size=3,
                seed=7,
            )

            self.assertEqual(features.shape[0], 4)
            counts = np.bincount(labels)
            self.assertEqual(counts.tolist(), [2, 2])

    def test_load_csv_dataset_global_label_quota_streams_all_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "single.csv").write_text(
                "Feature A,Label\n"
                "1,BENIGN\n"
                "2,BENIGN\n"
                "3,BENIGN\n"
                "4,AttackA\n"
                "5,AttackA\n"
                "6,AttackA\n"
                "7,AttackB\n"
                "8,AttackB\n"
                "9,AttackB\n",
                encoding="utf-8",
            )

            features, labels, raw_labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*.csv",
                label_col="Label",
                task="binary",
                benign_labels=("BENIGN",),
                max_rows_per_label=2,
                csv_chunk_size=2,
                seed=7,
                return_raw_labels=True,
            )

            self.assertEqual(features.shape[0], 6)
            self.assertEqual(
                {label: raw_labels.tolist().count(label) for label in set(raw_labels)},
                {
                    "BENIGN": 2,
                    "AttackA": 2,
                    "AttackB": 2,
                },
            )
            self.assertEqual(np.bincount(labels).tolist(), [2, 4])

    def test_load_csv_dataset_global_label_quota_respects_max_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "single.csv").write_text(
                "Feature A,Label\n1,BENIGN\n2,BENIGN\n3,AttackA\n4,AttackA\n5,AttackB\n6,AttackB\n",
                encoding="utf-8",
            )

            features, labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*.csv",
                label_col="Label",
                task="binary",
                benign_labels=("BENIGN",),
                max_rows=4,
                max_rows_per_label=2,
                csv_chunk_size=3,
                seed=7,
            )

            self.assertEqual(features.shape[0], 4)
            self.assertEqual(labels.shape[0], 4)

    def test_load_csv_dataset_parent_directory_quota_without_include_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, values in {
                "Benign_Final": [1, 2, 3],
                "Recon-PortScan": [4, 5, 6],
                "DDoS": [7, 8, 9],
            }.items():
                label_dir = root / label
                label_dir.mkdir(parents=True)
                rows = "\n".join(str(value) for value in values)
                (label_dir / "part.csv").write_text(f"Feature A\n{rows}\n", encoding="utf-8")

            features, labels, raw_labels = load_csv_dataset(
                csv_path=None,
                csv_dir=str(root),
                csv_glob="*/*.csv",
                label_source="parent_dir",
                task="binary",
                benign_labels=("Benign_Final",),
                max_rows_per_label=2,
                return_raw_labels=True,
            )

            self.assertEqual(features.shape[0], 6)
            self.assertEqual(
                {label: raw_labels.tolist().count(label) for label in set(raw_labels)},
                {
                    "Benign_Final": 2,
                    "Recon-PortScan": 2,
                    "DDoS": 2,
                },
            )
            self.assertEqual(np.bincount(labels).tolist(), [2, 4])

    def test_load_csv_dataset_strict_ingest_rejects_non_numeric_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.csv").write_text(
                "Feature A,Label\n1,BENIGN\noops,Attack\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Strict ingest failed"):
                load_csv_dataset(
                    csv_path=None,
                    csv_dir=str(root),
                    csv_glob="*.csv",
                    label_col="Label",
                    task="binary",
                    benign_labels=("BENIGN",),
                    strict_ingest=True,
                )


if __name__ == "__main__":
    unittest.main()
