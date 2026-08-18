from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from rdsynth.pipeline.stage1_eval import TORCH_ORACLE_TYPES, batched_torch_preds
from rdsynth.stages.oracle import OracleWrapper


@dataclass(frozen=True)
class Stage1MatrixSummaryRow:
    name: str
    oracle_type: str
    agreement: float
    acc: float
    f1: float


@dataclass(frozen=True)
class Stage1AgreementMatrixResult:
    matrix: list[list[float]]
    matrix_path: Path
    summary_path: Path
    diag_mean: float
    diag_std: float
    within_mean: float
    within_std: float
    cross_mean: float
    cross_std: float


def write_stage1_agreement_matrix(
    *,
    bundle: Any,
    oracle_cfgs: list[dict[str, Any]],
    oracle_pool: Mapping[str, OracleWrapper],
    surrogate_pool: Mapping[str, nn.Module],
    out_dir: Path,
    matrix_max_rows: int | None,
    matrix_batch_size: int | None,
    seed: int,
    device: torch.device,
    summary_rows: Sequence[Stage1MatrixSummaryRow],
) -> Stage1AgreementMatrixResult:
    x_val = torch.tensor(bundle.x_val, dtype=torch.float32, device=device)
    if matrix_max_rows is not None and x_val.size(0) > matrix_max_rows:
        rng = np.random.default_rng(seed)
        sel = rng.choice(np.arange(x_val.size(0)), int(matrix_max_rows), replace=False)
        x_val = x_val[sel]

    oracle_names = [str(cfg["name"]) for cfg in oracle_cfgs]
    matrix: list[list[float]] = []
    oracle_preds_cache: dict[str, np.ndarray] = {}
    surrogate_preds_cache: dict[str, np.ndarray] = {}

    with torch.no_grad():
        print("\n[Stage1] computing mutual extraction matrix...")
        for name in oracle_names:
            oracle = oracle_pool[name]
            if oracle.model_type in TORCH_ORACLE_TYPES:
                oracle_preds_cache[name] = batched_torch_preds(oracle.model, x_val, matrix_batch_size)
            else:
                oracle_preds_cache[name] = oracle.model.predict(x_val.detach().cpu().numpy())
            surrogate_preds_cache[name] = batched_torch_preds(surrogate_pool[name], x_val, matrix_batch_size)

        for surrogate_name in oracle_names:
            row: list[float] = []
            s_preds = surrogate_preds_cache[surrogate_name]
            for oracle_name in oracle_names:
                o_preds = oracle_preds_cache[oracle_name]
                row.append(float((s_preds == o_preds).mean()))
            matrix.append(row)

    print("\n[Stage1] mutual extraction agreement matrix:")
    header = ["surrogate\\target"] + oracle_names
    print("\t".join(header))
    for surrogate_name, row in zip(oracle_names, matrix):
        print(f"{surrogate_name}\t" + "\t".join(f"{value:.4f}" for value in row))

    matrix_path = out_dir / "agreement_matrix.csv"
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for surrogate_name, row in zip(oracle_names, matrix):
            writer.writerow([surrogate_name] + [f"{value:.6f}" for value in row])
    print(f"[Stage1] agreement matrix saved to {matrix_path}")

    name_to_group = {str(cfg["name"]): oracle_group(str(cfg["type"])) for cfg in oracle_cfgs}
    diag_vals: list[float] = []
    within_vals: list[float] = []
    cross_vals: list[float] = []
    for row_index, surrogate_name in enumerate(oracle_names):
        for col_index, oracle_name in enumerate(oracle_names):
            value = matrix[row_index][col_index]
            if row_index == col_index:
                diag_vals.append(value)
            if name_to_group[surrogate_name] == name_to_group[oracle_name]:
                within_vals.append(value)
            else:
                cross_vals.append(value)

    diag_mean, diag_std = mean_std(diag_vals)
    within_mean, within_std = mean_std(within_vals)
    cross_mean, cross_std = mean_std(cross_vals)

    print("\n[Stage1] matrix summary:")
    print(f"diag_mean={diag_mean:.4f} diag_std={diag_std:.4f}")
    print(f"within_group_mean={within_mean:.4f} within_group_std={within_std:.4f}")
    print(f"cross_group_mean={cross_mean:.4f} cross_group_std={cross_std:.4f}")

    summary_path = out_dir / "agreement_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "oracle_type", "group", "diag_agreement"])
        for index, row in enumerate(summary_rows):
            writer.writerow([row.name, row.oracle_type, name_to_group[row.name], f"{matrix[index][index]:.6f}"])
        writer.writerow([])
        writer.writerow(["metric", "mean", "std"])
        writer.writerow(["diag", f"{diag_mean:.6f}", f"{diag_std:.6f}"])
        writer.writerow(["within_group", f"{within_mean:.6f}", f"{within_std:.6f}"])
        writer.writerow(["cross_group", f"{cross_mean:.6f}", f"{cross_std:.6f}"])
    print(f"[Stage1] agreement summary saved to {summary_path}")

    return Stage1AgreementMatrixResult(
        matrix=matrix,
        matrix_path=matrix_path,
        summary_path=summary_path,
        diag_mean=diag_mean,
        diag_std=diag_std,
        within_mean=within_mean,
        within_std=within_std,
        cross_mean=cross_mean,
        cross_std=cross_std,
    )


def oracle_group(model_type: str) -> str:
    if model_type in {"cnn", "rnn", "gru", "lstm"}:
        return "seq"
    if model_type in {"mlp", "logistic", "random_forest", "extra_trees", "linear_svm", "transformer"}:
        return "mlp_tree"
    return "other"


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.array(values, dtype=np.float64)
    return float(array.mean()), float(array.std())
