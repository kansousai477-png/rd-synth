from __future__ import annotations

from pathlib import Path

import numpy as np

from rdsynth.pipeline.data import _data_artifact_dir, _data_cache_path, load_data_context
from rdsynth.pipeline.stage_contracts import StageManifestSpec, build_stage_output_files, save_stage_manifest_spec
from rdsynth.utils.artifacts import ensure_dir, save_config, save_metrics, save_metrics_csv
from rdsynth.utils.config import load_yaml
from rdsynth.utils.pipeline_config import prepare_pipeline_config


def run_data_prep(config_path: str | Path) -> None:
    resolved_config = Path(config_path).resolve()
    cfg = prepare_pipeline_config(load_yaml(resolved_config), resolved_config)
    project_cfg = cfg["project"]
    seed = int(project_cfg["seed"])
    out_dir = ensure_dir(Path(project_cfg["out_dir"]) / "data_prep")

    cache_path = _data_cache_path(cfg, seed)
    artifact_dir = _data_artifact_dir(cfg, seed)
    cache_hit_before = cache_path.exists()
    artifact_hit_before = artifact_dir.exists()

    context = load_data_context(cfg, seed)

    save_config(str(resolved_config), out_dir)
    metrics = {
        "seed": seed,
        "data_cache_path": str(cache_path),
        "data_artifact_dir": str(artifact_dir),
        "data_cache_hit_before": bool(cache_hit_before),
        "data_artifact_hit_before": bool(artifact_hit_before),
        "rows": int(context.features.shape[0]),
        "feature_count": int(context.features.shape[1]),
        "train_rows": int(np.asarray(context.bundle.x_train).shape[0]),
        "val_rows": int(np.asarray(context.bundle.x_val).shape[0]),
        "test_rows": int(np.asarray(context.bundle.x_test).shape[0]),
    }
    save_metrics(metrics, out_dir)
    save_metrics_csv(metrics, out_dir)

    save_stage_manifest_spec(
        StageManifestSpec(
            stage_name="data_prep",
            out_dir=out_dir,
            config_path=resolved_config,
            inputs={
                "dataset": str(cfg.get("data", {}).get("dataset", "")),
                "seed": seed,
                "cache_path": str(cache_path),
            },
            outputs=build_stage_output_files(
                extra_outputs={
                    "data_cache": cache_path,
                    "data_artifact_dir": artifact_dir,
                }
            ),
            arrays={
                "features": context.features.to_numpy(dtype=np.float64, copy=False),
                "labels": np.asarray(context.labels),
                "x_train": np.asarray(context.bundle.x_train),
                "x_val": np.asarray(context.bundle.x_val),
                "x_test": np.asarray(context.bundle.x_test),
            },
            metrics=metrics,
        )
    )
    print(f"[DataPrep] cache={cache_path}")
    print(f"[DataPrep] artifacts={artifact_dir}")
    print(
        f"[DataPrep] rows={metrics['rows']} features={metrics['feature_count']} "
        f"train={metrics['train_rows']} val={metrics['val_rows']} test={metrics['test_rows']}"
    )
