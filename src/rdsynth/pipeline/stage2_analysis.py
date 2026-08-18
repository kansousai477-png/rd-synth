from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rdsynth.pipeline.stage_contracts import VersionedArtifactSpec, build_versioned_artifact_payload
from rdsynth.utils.artifacts import save_records_csv

PARETO_FIELDS = [
    "mal_anchor_alpha",
    "asr_oracle",
    "asr_surrogate",
    "norm_FFD",
    "norm_AdvToMal_L2",
    "selection_score",
    "remapability_score",
    "remap_projection_penalty",
    "remap_clip_penalty",
    "remap_center_penalty",
    "asr_surrogate",
    "asr_oracle",
    "adv_pmal_surrogate",
    "adv_pmal_oracle",
    "adv_to_ben_l2",
    "adv_to_mal_l2",
    "ffd",
    "swd",
    "energy",
    "c2st_auc",
    "c2st_acc",
]


def save_pareto_front(path: Path, rows: list[dict[str, float]]) -> Path:
    save_records_csv(path, rows, fieldnames=PARETO_FIELDS)
    return path


def print_stage2_metric_tables(
    *,
    metrics_payload: dict[str, Any],
    metrics_norm: Any,
    adv_ben_l2: float,
    adv_mal_l2: float,
    eval_denorm: bool,
    metrics_denorm: Any | None = None,
    adv_ben_l2_denorm: float | None = None,
    adv_mal_l2_denorm: float | None = None,
) -> None:
    def _print_group(title: str, metrics: dict[str, Any], keys: list[str]) -> None:
        print(f"\n{title}")
        for key in keys:
            if key in metrics:
                print(f"  {key:<14} {metrics[key]:.6f}")

    def _print_metrics_table(label: str, metrics: dict[str, Any], dist_ben: float, dist_mal: float, mode: str) -> None:
        print(f"\n[Stage2] metrics ({label}):")
        if mode == "denorm":
            print("  DenormStats")
            nan_rate = metrics_payload.get("denorm_nan_rate", float("nan"))
            inf_rate = metrics_payload.get("denorm_inf_rate", float("nan"))
            print(f"  {'NaN_Rate':<14} {nan_rate:.6f}")
            print(f"  {'Inf_Rate':<14} {inf_rate:.6f}")
            _print_group(
                "  Constraints",
                metrics,
                [
                    "Violation_Range",
                    "Violation_NonNeg",
                    "Violation_Integer",
                    "ΔMean_S",
                    "ΔMean_T",
                    "ΔMean_P",
                    "ΔStd_S",
                    "ΔStd_T",
                    "ΔStd_P",
                ],
            )
        else:
            _print_group("  Core", metrics, ["FFD", "SWD", "Energy", "C2ST-AUC", "C2ST-Acc"])
            _print_group(
                "  Coverage",
                metrics,
                [
                    "Coverage@1",
                    "Coverage@5",
                    "Coverage@10",
                    "kNN-P@1",
                    "kNN-R@1",
                    "kNN-P",
                    "kNN-R",
                    "kNN-P@10",
                    "kNN-R@10",
                ],
            )
            _print_group(
                "  Structure",
                metrics,
                [
                    "CorrΔ",
                    "CovSpec-L2",
                    "CovTrace",
                    "CorrΔ_ST",
                    "CorrΔ_SP",
                    "CorrΔ_TP",
                    "ΔMean_S",
                    "ΔMean_T",
                    "ΔMean_P",
                    "ΔStd_S",
                    "ΔStd_T",
                    "ΔStd_P",
                ],
            )
            _print_group(
                "  Validity",
                metrics,
                ["PairDist-KS", "PairMean", "Violation_Range", "Violation_NonNeg", "Violation_Integer"],
            )
        print("\n  Distances")
        print(f"  {'AdvToBen_L2':<14} {dist_ben:.6f}")
        print(f"  {'AdvToMal_L2':<14} {dist_mal:.6f}")

    _print_metrics_table("normalized", metrics_norm.as_dict(), adv_ben_l2, adv_mal_l2, mode="norm")
    if eval_denorm and metrics_denorm is not None and adv_ben_l2_denorm is not None and adv_mal_l2_denorm is not None:
        _print_metrics_table(
            "denormalized", metrics_denorm.as_dict(), adv_ben_l2_denorm, adv_mal_l2_denorm, mode="denorm"
        )
    else:
        print("\n[Stage2] denormalized metrics skipped (eval_denorm_metrics=false).")

    if "asr_surrogate" in metrics_payload:
        print(
            "\n[Stage2] surrogate attack success"
            f" asr={metrics_payload['asr_surrogate']:.6f}"
            f" adv_pmal={metrics_payload.get('adv_prob_malicious_mean', float('nan')):.6f}"
            f" mal_pmal={metrics_payload.get('mal_prob_malicious_mean', float('nan')):.6f}"
        )
    if "asr_oracle" in metrics_payload:
        print(
            "[Stage2] oracle attack success"
            f" asr={metrics_payload['asr_oracle']:.6f}"
            f" adv_pmal={metrics_payload.get('adv_prob_malicious_mean_oracle', float('nan')):.6f}"
            f" mal_pmal={metrics_payload.get('mal_prob_malicious_mean_oracle', float('nan')):.6f}"
        )


def build_stage2_artifact_payload(
    *,
    x_adv_pre: np.ndarray,
    x_adv_norm: np.ndarray,
    x_ben_norm: np.ndarray,
    x_mal_norm: np.ndarray,
    x_ben_pre: np.ndarray,
    x_mal_pre: np.ndarray,
    denorm_mean: np.ndarray,
    denorm_std: np.ndarray,
    feature_names: list[str],
    x_adv_denorm: np.ndarray | None = None,
    x_ben_denorm: np.ndarray | None = None,
    x_mal_denorm: np.ndarray | None = None,
) -> dict[str, Any]:
    return build_versioned_artifact_payload(
        VersionedArtifactSpec(
            version_as_array=True,
            fields={
                "adv": x_adv_pre,
                "adv_pre": x_adv_pre,
                "adv_ben_norm": x_adv_norm,
                "benign": x_ben_norm,
                "mal": x_mal_norm,
                "benign_pre": x_ben_pre,
                "mal_pre": x_mal_pre,
                "ben_stats_mean": denorm_mean,
                "ben_stats_std": denorm_std,
                "adv_space": np.asarray("preprocessed"),
                "feature_names": np.asarray(feature_names),
            },
            optional_fields={
                "adv_denorm": x_adv_denorm,
                "benign_denorm": x_ben_denorm,
                "mal_denorm": x_mal_denorm,
            },
        )
    )
