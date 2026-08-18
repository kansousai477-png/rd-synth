from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rdsynth.utils.artifacts import ensure_dir, save_metrics, save_metrics_csv
from rdsynth.utils.config import optional_section
from rdsynth.utils.data_quality import compute_data_quality


def run_stage1_data_quality(
    *,
    stage1_cfg: Mapping[str, Any],
    out_dir: Path,
    features: Any,
    labels: np.ndarray,
    seed: int,
) -> None:
    dq_cfg = optional_section(stage1_cfg, "data_quality")
    if not bool(dq_cfg.get("enable", True)):
        return
    dq = compute_data_quality(
        features,
        labels,
        max_rows=dq_cfg.get("max_rows"),
        seed=seed,
        high_missing_threshold=float(dq_cfg.get("high_missing_threshold", 0.1)),
        corr_topk=int(dq_cfg.get("corr_topk", 5)),
    )
    dq_dir = ensure_dir(out_dir / "data_quality")
    save_metrics(dq, dq_dir)
    save_metrics_csv(dq, dq_dir)
    print(
        "[Stage1] data_quality"
        f" rows={dq.get('rows')}"
        f" features={dq.get('features')}"
        f" pos_rate={dq.get('label_positive_rate', 0.0):.4f}"
        f" dup_rate={dq.get('duplicate_rate', 0.0):.4f}"
        f" max_abs_corr={dq.get('max_abs_corr_with_label', float('nan')):.4f}"
    )
