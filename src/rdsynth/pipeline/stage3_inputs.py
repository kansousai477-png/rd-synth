from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from rdsynth.utils.checkpoints import load_torch_state


@dataclass(frozen=True)
class LoadedAdvSamples:
    path: Path
    adv: np.ndarray | None
    adv_norm: np.ndarray | None
    adv_space: str
    adv_mean: Any | None
    adv_std: Any | None
    loaded: bool
    count: int


def resolve_adv_samples_path(adv_samples_path: str, project_out_dir: str | Path) -> Path:
    if adv_samples_path:
        return Path(adv_samples_path)
    return Path(project_out_dir) / "stage2" / "adv_samples.npz"


def load_adv_samples(
    adv_path: Path,
    *,
    project_out_dir: str | Path,
    current_feature_names: Sequence[str],
    expected_feature_dim: int,
    copy_to: Path | None = None,
    load_torch_state_fn: Callable[..., Mapping[str, Any]] = load_torch_state,
    warn_fn: Callable[[str], None] = print,
) -> LoadedAdvSamples:
    if not adv_path.exists():
        return LoadedAdvSamples(
            path=adv_path,
            adv=None,
            adv_norm=None,
            adv_space="",
            adv_mean=None,
            adv_std=None,
            loaded=False,
            count=0,
        )

    with np.load(adv_path) as adv_npz:
        npz_data = {name: adv_npz[name] for name in adv_npz.files}
        adv, adv_space, adv_stats = _resolve_adv_samples(
            adv_npz,
            project_out_dir=project_out_dir,
            load_torch_state_fn=load_torch_state_fn,
            warn_fn=warn_fn,
        )
        saved_feature_names = adv_npz.get("feature_names")

    if adv is not None and saved_feature_names is not None:
        restored_names = [str(name) for name in np.asarray(saved_feature_names).tolist()]
        if restored_names != [str(name) for name in current_feature_names]:
            warn_fn("[Stage3][Warn] adv_samples feature_names mismatch; skipping loaded adversarial samples.")
            adv = None

    if adv is not None and adv.shape[1] != expected_feature_dim:
        warn_fn(
            "[Stage3][Warn] adv_samples feature dim mismatch "
            f"(got {adv.shape[1]}, expected {expected_feature_dim}). Skipping."
        )
        adv = None

    adv_mean = adv_stats.get("mean")
    adv_std = adv_stats.get("std")
    adv_norm = None
    if adv is not None and adv_mean is not None and adv_std is not None:
        adv_std_safe = np.asarray(adv_std, dtype=np.float64) + 1.0e-8
        adv_norm = (adv.astype(np.float64) - np.asarray(adv_mean, dtype=np.float64)) / adv_std_safe

    if copy_to is not None:
        np.savez_compressed(copy_to, **npz_data)

    return LoadedAdvSamples(
        path=adv_path,
        adv=adv,
        adv_norm=adv_norm,
        adv_space=adv_space,
        adv_mean=adv_mean,
        adv_std=adv_std,
        loaded=adv is not None,
        count=int(adv.shape[0]) if adv is not None else 0,
    )


def _resolve_adv_samples(
    npz: Any,
    *,
    project_out_dir: str | Path,
    load_torch_state_fn: Callable[..., Mapping[str, Any]],
    warn_fn: Callable[[str], None],
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    adv_space = _npz_str(npz.get("adv_space")).strip().lower()
    stats: dict[str, Any] = {}
    adv_pre = npz.get("adv_pre")
    if adv_pre is None:
        adv_pre = npz.get("adv")
    if adv_pre is None:
        return None, adv_space, stats

    if adv_space in ("benign_norm", "benign", "norm"):
        mean = npz.get("ben_stats_mean")
        std = npz.get("ben_stats_std")
        if mean is None or std is None:
            stage2_state_path = Path(project_out_dir) / "stage2" / "stage2.pt"
            if stage2_state_path.exists():
                stage2_state = load_torch_state_fn(stage2_state_path, map_location="cpu")
                stats = dict(stage2_state.get("ben_stats", {}))
                mean = stats.get("denorm_mean", stats.get("mean"))
                std = stats.get("denorm_std", stats.get("std"))
        stats["mean"] = mean
        stats["std"] = std
        if mean is not None and std is not None:
            adv_pre = adv_pre * std + mean
        else:
            warn_fn("[Stage3][Warn] adv_samples marked benign_norm but no stats found; using raw values.")

    if not adv_space:
        adv_space = "preprocessed" if "adv_pre" in npz.files else "unknown"
    adv_pre = np.nan_to_num(adv_pre, nan=0.0, posinf=0.0, neginf=0.0)
    if not stats:
        mean = npz.get("ben_stats_mean")
        std = npz.get("ben_stats_std")
        if mean is not None and std is not None:
            stats["mean"] = mean
            stats["std"] = std
    return adv_pre.astype(np.float32), adv_space, stats


def _npz_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.tolist())
        if value.size == 1:
            return str(value.ravel()[0])
    return str(value)
