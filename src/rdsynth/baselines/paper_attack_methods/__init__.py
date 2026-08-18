from __future__ import annotations

from rdsynth.baselines.paper_attack_methods.amoeba import generate_amoeba
from rdsynth.baselines.paper_attack_methods.digfupas import generate_digfupas
from rdsynth.baselines.paper_attack_methods.gpmt import generate_gpmt
from rdsynth.baselines.paper_attack_methods.idsgan import generate_idsgan
from rdsynth.baselines.paper_attack_methods.netdiffusion import generate_netdiffusion
from rdsynth.baselines.paper_attack_methods.progen import generate_progen
from rdsynth.baselines.paper_attack_methods.simple_controls import (
    generate_iat_jitter,
    generate_padding_only,
    generate_topk_perturb,
)
from rdsynth.baselines.paper_attack_methods.vulnergan import generate_vulnergan

__all__ = [
    "generate_amoeba",
    "generate_digfupas",
    "generate_gpmt",
    "generate_iat_jitter",
    "generate_idsgan",
    "generate_netdiffusion",
    "generate_padding_only",
    "generate_progen",
    "generate_topk_perturb",
    "generate_vulnergan",
]
