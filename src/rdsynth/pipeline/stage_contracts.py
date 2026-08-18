from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rdsynth.utils.artifacts import save_stage_manifest


@dataclass(frozen=True)
class StageManifestSpec:
    stage_name: str
    out_dir: Path
    config_path: str | Path
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VersionedArtifactSpec:
    fields: dict[str, Any] = field(default_factory=dict)
    optional_fields: dict[str, Any | None] = field(default_factory=dict)
    version: int = 1
    version_as_array: bool = False


def output_filename(path: str | Path) -> str:
    text = str(path)
    candidate = Path(text)
    if candidate.is_absolute():
        return str(candidate)
    return candidate.name


def build_stage_output_files(
    *,
    primary_artifact_key: str | None = None,
    primary_artifact_name: str | Path | None = None,
    include_config: bool = True,
    include_metrics: bool = True,
    extra_outputs: Mapping[str, str | Path | None] | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if include_config:
        outputs["config"] = "config.yaml"
    if include_metrics:
        outputs["metrics_json"] = "metrics.json"
        outputs["metrics_csv"] = "metrics.csv"
    if primary_artifact_key is not None and primary_artifact_name is not None:
        outputs[primary_artifact_key] = output_filename(primary_artifact_name)
    for key, value in (extra_outputs or {}).items():
        if value is not None:
            outputs[str(key)] = output_filename(value)
    return outputs


def collect_manifest_arrays(named_arrays: Mapping[str, Any | None]) -> dict[str, Any]:
    return {str(name): value for name, value in named_arrays.items() if value is not None}


def save_stage_manifest_spec(spec: StageManifestSpec) -> Path:
    return save_stage_manifest(
        stage_name=spec.stage_name,
        out_dir=spec.out_dir,
        config_path=spec.config_path,
        inputs=spec.inputs,
        outputs=spec.outputs,
        arrays=spec.arrays,
        metrics=spec.metrics,
    )


def build_versioned_artifact_payload(spec: VersionedArtifactSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_version": np.asarray(spec.version) if spec.version_as_array else int(spec.version),
        **spec.fields,
    }
    for key, value in spec.optional_fields.items():
        if value is not None:
            payload[str(key)] = value
    return payload
