# Reproducing and extending the experiments

## Environment

Run commands from the repository root. The package declares Python >=3.11,
torch >=2.3, and torchvision >=0.18. The recorded workstation used:

- Windows, Python 3.14;
- PyTorch 2.13.0+cu130 and torchvision 0.28.0+cu130;
- NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM.

These are recorded environment details, not a claim that every supported
Python/PyTorch combination has been tested. For a new machine, install a mutually
compatible torch/torchvision pair for that machine before installing this project.
`analysis` adds Matplotlib and `test` adds pytest. The optional Windows `compile`
extra pins the previously used Triton version; it is unnecessary for eager mode.

## Dataset

Use the official [PCam release](https://github.com/basveeling/pcam). The loader expects:

```text
data/pcam/
  camelyonpatch_level_2_split_train_x.h5
  camelyonpatch_level_2_split_train_y.h5
  camelyonpatch_level_2_split_valid_x.h5
  camelyonpatch_level_2_split_valid_y.h5
  camelyonpatch_level_2_split_test_x.h5
  camelyonpatch_level_2_split_test_y.h5
```

On Windows, the supplied downloader retrieves and checks the official mirror:

```powershell
.\scripts\download_pcam.ps1
pcam-repro inspect-data --root data/pcam
```

The downloader explicitly maps the mirror's validation/test names to canonical
PCam checksums. Do not identify those splits by filenames alone. No data is
downloaded by installing the package or running the unit tests.

Models trained from scratch use PCam normalization. ImageNet-initialized models
use ImageNet normalization. Augmentation choices are stored in each configuration;
aggressive random crops are avoided because they can change the central label.

## Experiment matrix and status

| Configurations | Purpose | Recorded status |
|---|---|---|
| E00–E08 | CNN baselines, initialization, D4, and gated C16 | Complete |
| E09–E10 | Supervised vs. Mean Teacher with 5% labels | Complete |
| E11 | SimCLR representation pretraining | Completed separately; no classifier test row |
| E11A–E11B | Frozen linear probe and full fine-tuning | Complete |
| E12–E16 | Transformers, noisy labels, tensor networks, multi-scale attention | Complete |
| E17, seed 17 | PCam-D4Loc | Complete |
| E17, seeds 43/71; E17A–E17C | Additional seeds and ablations | Configurations only; no completed results released |

Configuration IDs identify experiments, not an equal-budget architecture ranking.
E11A/E11B require `runs/e11-simclr-resnet50/best.pt`, as specified by
`model.checkpoint`. They have different learning protocols and should not be
treated as a controlled change of a single variable.

Examples, to be started explicitly after obtaining the data:

```bash
pcam-repro train --config configs/e07_gdensenet_d4.toml --device cuda
pcam-repro train --config configs/e11_simclr_resnet50.toml --device cuda
pcam-repro train --config configs/e11a_simclr_linear_probe.toml --device cuda
pcam-repro train --config configs/e11b_simclr_finetune.toml --device cuda
```

`scripts/run_all_experiments.py` can orchestrate selected runs, but its full matrix
can take days. The default invocation should not be used as an installation check.

## Checkpoints and execution modes

The best checkpoint is selected by validation ROC AUC. Each run stores its
resolved configuration, runtime metadata, epoch history, best and last checkpoints,
and test predictions under `runs/`. Saved runtime sessions distinguish eager,
fast, compiled, and resumed execution.

```bash
pcam-repro train --config configs/e17_pcam_d4loc_s17.toml --device cuda --resume
```

`--resume` continues from the existing run's `last.pt`. It restores model,
optimizer, scheduler, and gradient-scaler state. RNG and DataLoader-worker states
are not restored, so a resumed run is not bitwise equivalent to uninterrupted
training. `--fast` and `--compile` are explicit execution options and can affect
floating-point behavior and reproducibility. Compiled execution is optional.

Raw checkpoints and predictions are not distributed in this repository. Exact
metric recomputation from the original runs requires those artifacts. The
published summary tables can be inspected and plotted without them.

## Explainability tools

Mask-based analysis additionally needs PCam metadata and CAMELYON16 annotations.
Use `--help` to inspect the data and output arguments before running:

```bash
python scripts/download_camelyon16_annotations.py --help
python scripts/audit_pcam_annotation_mapping.py --help
python scripts/run_xai_analysis.py --help
```

Validate the coordinate mapping before interpreting localization scores. The XAI
runner also requires trained checkpoints and saved test predictions. It computes
attributions and perturbations; it does not fit a classifier.

## Validation without training

```bash
python -m compileall -q src scripts
pcam-repro list-models
python -m pytest -q -k "not one_epoch and not supervised_fit and not compile_happens"
python scripts/plot_results.py
```

The selected tests cover configuration parsing, HDF5 loading on synthetic files,
metrics, model forward/backward passes, D4 invariance, and XAI. The full suite
additionally contains three tiny synthetic training/resumption tests; they are
excluded by the command above.

## Interpretation limits

- Most configurations have one completed seed. Bootstrap intervals over source
  slides describe conditional test uncertainty, not variability across training runs.
- The test set was reused during broader comparison and architecture development;
  confirmatory claims require an independent set.
- No completed ablation evidence isolates the contribution of the three E17
  components. No comparison isolates BCE vs. SCE under identical noisy labels.
- Paper results use different datasets, splits, label budgets, and evaluation
  units. Keep those distinctions when using the literature table.
- The study does not validate whole-slide diagnosis, patient-level decisions,
  external generalization, or clinical deployment.
