from __future__ import annotations

import numpy as np
import torch

from rdsynth.baselines.paper_attack_methods import (
    generate_amoeba,
    generate_digfupas,
    generate_gpmt,
    generate_idsgan,
    generate_netdiffusion,
    generate_progen,
    generate_vulnergan,
)
from rdsynth.baselines.paper_attack_methods.common import (
    PAPER_BASELINE_SPECS,
    PaperAttackContext,
    PaperBaselineSpec,
    as_numpy,
)
from rdsynth.baselines.paper_attack_methods.simple_controls import (
    generate_iat_jitter,
    generate_padding_only,
    generate_topk_perturb,
)

_PAPER_GENERATORS = {
    "idsgan_lite": generate_idsgan,
    "digfupas_lite": generate_digfupas,
    "gpmt_lite": generate_gpmt,
    "progen_lite": generate_progen,
    "amoeba_lite": generate_amoeba,
    "vulnergan_lite": generate_vulnergan,
    "netdiffusion_lite": generate_netdiffusion,
    "iat_jitter": generate_iat_jitter,
    "padding_only": generate_padding_only,
    "topk_perturb": generate_topk_perturb,
}


def get_paper_baseline_spec(name: str) -> PaperBaselineSpec | None:
    return PAPER_BASELINE_SPECS.get(str(name).lower())


def traffic_space_baseline_names() -> list[str]:
    return [spec.name for spec in PAPER_BASELINE_SPECS.values() if spec.traffic_space]


def stage3_policy_for_baseline(name: str) -> str:
    spec = get_paper_baseline_spec(name)
    return spec.stage3_policy if spec is not None else "feature_only_random_remap"


def generate_paper_attack_baseline(
    name: str,
    x_mal_pre: np.ndarray,
    x_ben_pre: np.ndarray,
    feature_names,
    score_fn,
    surrogate_model: torch.nn.Module | None = None,
    x_train_pre: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    device: torch.device | None = None,
    seed: int = 42,
    budget_scale: float = 1.0,
) -> np.ndarray:
    baseline_name = str(name).lower()
    if baseline_name not in _PAPER_GENERATORS:
        raise ValueError(f"Unknown paper baseline: {name}")

    x_mal = as_numpy(x_mal_pre)
    x_ben = as_numpy(x_ben_pre)
    if x_mal.ndim != 2 or x_ben.ndim != 2:
        raise ValueError("x_mal_pre and x_ben_pre must be 2D arrays.")
    if x_mal.shape[1] != x_ben.shape[1]:
        raise ValueError("Feature dimension mismatch between malicious and benign arrays.")
    if x_mal.shape[0] == 0:
        return x_mal.copy()

    context = PaperAttackContext(
        name=baseline_name,
        x_mal=x_mal,
        x_ben=x_ben,
        feature_names=tuple(str(item) for item in feature_names),
        score_fn=score_fn,
        surrogate_model=surrogate_model,
        x_train_pre=None if x_train_pre is None else as_numpy(x_train_pre),
        y_train=None if y_train is None else np.asarray(y_train, dtype=np.int64),
        device=device if device is not None else torch.device("cpu"),
        seed=int(seed),
        budget_scale=float(max(0.1, budget_scale)),
    )
    return _PAPER_GENERATORS[baseline_name](context)
