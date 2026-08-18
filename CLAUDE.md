# CLAUDE.md

RDSynth generates adversarial network flow features via diffusion models and remaps them into protocol-valid PCAP traces, targeting **flow-level statistical-feature NIDS** under **hard-label black-box** access.

## Commands

```powershell
.\scripts\python_in_venv.ps1 -m pytest -q                          # Full test suite
.\scripts\python_in_venv.ps1 scripts\run_pipeline.py --config configs\demo_fast.yaml
.\scripts\check_windows.ps1                                         # pytest + coverage
.\scripts\check_windows.ps1 -WithLint                               # + ruff check/format
```

### Pipeline

```powershell
# Full pipeline
.\scripts\python_in_venv.ps1 scripts\run_pipeline.py --config configs\paper_main.yaml

# Individual stages
.\scripts\python_in_venv.ps1 scripts\run_stage1.py
.\scripts\python_in_venv.ps1 scripts\run_stage2.py
.\scripts\python_in_venv.ps1 scripts\run_stage3.py

# Reviewer suite (batch experiments)
.\scripts\python_in_venv.ps1 scripts\run_reviewer_suite.py --datasets nb15 --execution-mode inline
.\scripts\python_in_venv.ps1 scripts\run_cross_dataset_suite.py --datasets nb15,2017,2018,iot23 --profile paper --execution-mode subprocess

# CSV attack sweeps
.\scripts\python_in_venv.ps1 scripts\run_csv_experiments.py --dataset-preset 2017 --attacks "DDoS,PortScan,Bot"

# Quality gates & preflight
.\scripts\python_in_venv.ps1 scripts\run_quality_gate.py --level quick --datasets nb15,2017,2018,iot23 --profile paper
.\scripts\python_in_venv.ps1 scripts\run_full_preflight.py --datasets nb15,2017,2018,iot23 --profile paper

# Reports (HTML, regenerated from artifacts)
.\scripts\python_in_venv.ps1 scripts\run_cross_dataset_suite.py --datasets nb15,2017,2018,iot23 --profile paper --report-only --out-root <run_dir>
```

## Architecture

Three-stage pipeline: `scripts/` → `src/rdsynth/pipeline/` → `src/rdsynth/stages/` → `src/rdsynth/models/` + `src/rdsynth/data/`

| Stage | Pipeline | Algorithm |
|-------|----------|-----------|
| Stage1 — Surrogate extraction | `pipeline/stage1.py` | `stages/stage1_surrogate.py` |
| Stage2 — Diffusion generation | `pipeline/stage2.py` | `stages/stage2_diffusion.py` |
| Stage3 — PCAP remapping | `pipeline/stage3.py` | `stages/stage3_remap.py` |

| Directory | Purpose |
|-----------|---------|
| `src/rdsynth/pipeline/` | Orchestration, runtime, reporting |
| `src/rdsynth/stages/` | Stage-specific algorithms |
| `src/rdsynth/models/` | MLP, diffusion components |
| `src/rdsynth/baselines/` | Paper baselines (16 methods) |
| `src/rdsynth/data/` | Dataset loading, label mapping |
| `src/rdsynth/utils/` | Config, metrics, stats, schema |
| `configs/` | YAML experiment configs |
| `outputs/` | Artifacts: paper_main, reviewer_suite, ablations, debug, cache, failed, reports |

Config priority: CLI override → suite config → experiment config → dataset profile → defaults.

## Research Questions

- **RQ1**: Surrogate reliability under hard-label black-box
- **RQ2**: Diffusion stability vs. alternative generators
- **RQ3**: Statistical realism of adversarial features
- **RQ4**: Evasion effectiveness vs. baselines
- **RQ5**: PCAP protocol validity and offline deployability
- **RQ6**: Online deployment (future work only — do not claim)
- **RQ7**: Transfer IDS robustness and failure boundaries

## Key Principles

- **Fail-closed**: schema mismatches, missing labels, unparseable PCAPs, payload violations must explicitly fail — never silently skip.
- **Config-driven**: all experiment parameters in configs, never hardcoded.
- **Baseline fairness**: same splits, normalization, target model for all methods. `*_lite` = mechanism-aligned, not faithful reproduction, unless verified.
- **Claim discipline**: flow-level NIDS only. Offline deployability evidence, not online deployment. No raw-packet/embedding NIDS claims.
- **Threat model**: hard-label black-box. Attacker selects local feature extractor (structural alignment assumed, extractor mismatch stress-tested). Payload bytes never modified. Conservative protocol constraints.
- **Outputs**: gitignored, disposable. `outputs/` buckets: paper_main, reviewer_suite, ablations, debug, cache, failed, reports.

## Platform

Windows 11, Python 3.9, CUDA 12.6. `venv/` at repo root.
