# RDSynth

Adversarial network flow generation via diffusion models → protocol-valid PCAP traces.
Targets flow-level statistical-feature NIDS under hard-label black-box access.

## Setup

```powershell
.\scripts\bootstrap_windows.ps1 -Recreate
pip install -r requirements.txt -r requirements-dev.txt
```

## Quick Start

```powershell
# Dev smoke
.\scripts\python_in_venv.ps1 scripts\run_pipeline.py --config configs\demo_fast.yaml

# Full pipeline
.\scripts\python_in_venv.ps1 scripts\run_pipeline.py --config configs\paper_main.yaml
```

## Tests & Lint

```powershell
.\scripts\python_in_venv.ps1 -m pytest -q
.\scripts\check_windows.ps1 -WithLint
```

## Main Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | Full three-stage pipeline |
| `scripts/run_reviewer_suite.py` | Batch experiments across datasets |
| `scripts/run_cross_dataset_suite.py` | Four-dataset full run |
| `scripts/run_csv_experiments.py` | Per-attack CSV sweeps |
| `scripts/run_ablation_suite.py` | Module-level ablation |
| `scripts/run_quality_gate.py` | Quality gate before long runs |
| `scripts/run_full_preflight.py` | Full preflight check |

## Reports

HTML reports are regenerated from artifacts:

```powershell
.\scripts\python_in_venv.ps1 scripts\run_cross_dataset_suite.py --datasets nb15,2017,2018,iot23 --profile paper --report-only --out-root outputs/reviewer_suite/runs/<run_id>
```

Individual report scripts under `scripts/generate_*.py`.

## Layout

```
src/rdsynth/pipeline/   Orchestration, runtime, reporting
src/rdsynth/stages/     Stage-specific algorithms
src/rdsynth/models/     MLP, diffusion components
src/rdsynth/baselines/  Paper baselines (16 methods)
src/rdsynth/data/       Dataset loading, label mapping
src/rdsynth/utils/      Config, metrics, stats, schema
configs/                YAML experiment configs
outputs/                Generated artifacts (gitignored)
data/                   Local datasets and PCAPs (gitignored)
```

## Platform

Windows 11, Python 3.9, CUDA 12.6. `venv/` at repo root.
