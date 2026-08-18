from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.data.csv_datasets import load_csv_dataset, resolve_dataset_profile  # noqa: E402
from rdsynth.pipeline.reviewer_suite import DATASET_SPECS, load_yaml, normalize_label, selected_attacks  # noqa: E402
from rdsynth.pipeline.stage3_pcap_semantics import categories_for_attack, categories_for_attacks  # noqa: E402

DATASETS = ["nb15", "2017", "2018", "iot23"]
MAX_SMOKE_PCAP_BYTES = 1024 * 1024
SMOKE_ROWS_PER_RAW_LABEL = 2
REPRESENTATIVE_ATTACK_CSV = {
    "nb15": "data/CIC-NB15/CICFlowMeter_out.csv",
    "2017": "data/CIC-IDS-2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "2018": "data/CIC-IDS2018-CSV/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
}


class FourDatasetPrelaunchAssetsTest(unittest.TestCase):
    def _selected_attacks(self, dataset: str) -> list[str]:
        suite_cfg = load_yaml(ROOT / "configs" / "reviewer_suite.yaml")
        base_cfg = load_yaml(ROOT / str(DATASET_SPECS[dataset]["base_config"]))
        return selected_attacks(
            dataset,
            suite_cfg=suite_cfg,
            base_cfg=base_cfg,
            override_attacks=[],
            max_attacks=0,
        )

    def _csv_path_with_label(self, data_cfg: dict, label: str) -> Path:
        profile = resolve_dataset_profile(data_cfg)
        csv_paths: list[Path] = []
        if profile.csv_path:
            csv_paths.append(ROOT / profile.csv_path)
        if profile.csv_dir:
            csv_paths.extend(sorted((ROOT / profile.csv_dir).glob(profile.csv_glob)))
        wanted = normalize_label(label)
        label_col = normalize_label(profile.label_col)
        for path in csv_paths:
            if not path.is_file():
                continue
            with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                fields = [normalize_label(name) for name in header]
                if label_col not in fields:
                    continue
                label_idx = fields.index(label_col)
                for row in reader:
                    if label_idx < len(row) and normalize_label(row[label_idx]) == wanted:
                        return path
        raise AssertionError(f"No CSV row found for label {label!r}")

    def test_reviewer_suite_attacks_have_stage3_semantic_categories(self) -> None:
        missing: list[str] = []
        for dataset in DATASETS:
            for attack in self._selected_attacks(dataset):
                categories = categories_for_attack(dataset, attack)
                if not categories:
                    missing.append(f"{dataset}:{attack}")

        self.assertEqual(missing, [])

    def test_semantic_pcap_categories_exist_and_have_small_smoke_candidates(self) -> None:
        pcap_root = ROOT / "data" / "PCAPs" / "malicious"
        missing_dirs: list[str] = []
        missing_small_pcaps: list[str] = []
        for dataset in DATASETS:
            categories = categories_for_attacks(dataset, self._selected_attacks(dataset))
            for category in categories:
                category_dir = pcap_root / category
                if not category_dir.exists():
                    missing_dirs.append(f"{dataset}:{category}")
                    continue
                small_pcaps = [
                    path
                    for path in category_dir.rglob("*.pcap")
                    if path.is_file() and int(path.stat().st_size) <= MAX_SMOKE_PCAP_BYTES
                ]
                if not small_pcaps:
                    missing_small_pcaps.append(f"{dataset}:{category}")

        self.assertEqual(missing_dirs, [])
        self.assertEqual(missing_small_pcaps, [])

    def test_each_dataset_loads_representative_attack_with_benign_under_raw_label_cap(self) -> None:
        missing_labels: list[str] = []
        over_cap_labels: list[str] = []
        for dataset in DATASETS:
            base_cfg = load_yaml(ROOT / str(DATASET_SPECS[dataset]["base_config"]))
            data_cfg = dict(base_cfg["data"])
            profile = resolve_dataset_profile(data_cfg)
            representative_attack = self._selected_attacks(dataset)[0]
            requested_labels = [*profile.benign_labels, representative_attack]
            csv_path = profile.csv_path
            csv_dir = profile.csv_dir
            csv_glob = profile.csv_glob
            if profile.label_source == "column":
                csv_path = str(ROOT / REPRESENTATIVE_ATTACK_CSV.get(dataset, ""))
                if not Path(csv_path).is_file():
                    csv_path = str(self._csv_path_with_label(data_cfg, representative_attack))
                csv_dir = None
                csv_glob = "*.csv"
            features, labels, raw_labels = load_csv_dataset(
                csv_path=csv_path,
                csv_dir=csv_dir,
                csv_glob=csv_glob,
                label_col=profile.label_col,
                label_source=profile.label_source,
                task=str(data_cfg.get("task", "binary")),
                benign_labels=profile.benign_labels,
                drop_cols=profile.drop_cols,
                merge_strategy=str(data_cfg.get("merge_strategy", "intersection")),
                include_labels=requested_labels,
                max_rows_per_label=SMOKE_ROWS_PER_RAW_LABEL,
                csv_chunk_size=50000,
                seed=42,
                return_raw_labels=True,
            )

            observed = {str(label) for label in raw_labels.tolist()}
            benign_observed = {normalize_label(label) for label in observed} & {
                normalize_label(label) for label in profile.benign_labels
            }
            if not benign_observed:
                missing_labels.append(f"{dataset}:<benign>")
            if representative_attack not in observed:
                missing_labels.append(f"{dataset}:{representative_attack}")
            for label in observed:
                count = raw_labels.tolist().count(label)
                if count > SMOKE_ROWS_PER_RAW_LABEL:
                    over_cap_labels.append(f"{dataset}:{label}={count}")
            self.assertGreater(features.shape[0], 0, dataset)
            self.assertGreater(features.shape[1], 10, dataset)
            self.assertEqual(set(labels.tolist()), {0, 1}, dataset)

        self.assertEqual(missing_labels, [])
        self.assertEqual(over_cap_labels, [])


if __name__ == "__main__":
    unittest.main()
