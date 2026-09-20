# Recorded results

This directory publishes selected numerical summaries from the completed PCam
experiments and the September 2026 metric audit. It does not contain data,
trained checkpoints, individual patch predictions, or training logs.

| File | Contents |
|---|---|
| [classification.csv](classification.csv) | 19 completed classification runs: parameter counts, training-history summaries, and corrected test metrics |
| [prediction_audit.json](prediction_audit.json) | Checks on saved predictions; WSI bootstrap intervals for selected models and paired AUC differences |
| [xai_localization.csv](xai_localization.csv) | Per-model/per-method aggregate map metrics, 95% bootstrap intervals, and sample counts |

The plotting script reads `classification.csv` directly. It does not retrain
models, load checkpoints, or reconstruct results from rounded README values.

## Classification table

All `test_*` metrics refer to the official 32,768-patch PCam test split.
Rates use fractions in [0, 1], not percentages. Accuracy, F1, precision,
sensitivity, specificity, and confusion counts use a probability threshold of
0.5. `parameters` is the full model size; `trainable_parameters` is the optimized
subset (notably smaller for the E11A linear probe).

`best_auc_epoch` is the checkpoint selected by validation AUC;
`epochs_completed` can be larger because of early-stopping patience.
`minimum_nll_epoch` is diagnostic and does not identify the selected checkpoint.
`fast_mode` and `compile_enabled` record execution settings, not methodological
equivalence between runs.

`metrics_source` identifies the saved-prediction audit. Average precision was
recomputed with tied scores grouped at a common threshold. The legacy
`corrected_evaluation` flag marks a separate evaluation correction and does **not**
mean that rows marked `False` retain the old AP calculation. E08 is described
as C16 with relative-orientation gating; its historical identifier is unchanged.

## Uncertainty

The prediction audit resamples **54 source WSIs** with replacement, retaining
their patches together, for **2,000 paired bootstrap draws**. The saved 95%
percentile intervals condition on the observed slides and trained checkpoints.
They do not include training-seed variation or performance at a new center.

The audit retains SHA-256 hashes identifying the prediction files and metadata
used locally. Those hashes support provenance; the underlying files are not
bundled, so a fresh checkout alone cannot reproduce the original inference audit.

## Localization

The audit selected **48 patches** using E06 confusion groups. **25** had available,
non-empty tumor masks. Each row contains `model`, `method`, `metric`, `mean`,
`ci95_low`, `ci95_high`, and the actual sample count `n`.

`pixel_auprc` is a historical field name for pixel-level **average precision**.
Do not reinterpret it as trapezoidal PR AUC. `mass_inside` measures normalized
response inside the reference mask; pointing game tests the highest-scoring
location. Stability, deletion, and insertion evaluate different properties and
can produce different method rankings. High agreement with a mask does not
establish causal faithfulness or diagnostic validity.

The native E17 map obtained pixel AP 0.8275. Its parameter-randomization result
was undefined in the checked cases. The limited selected sample and incomplete
ablations preclude a general claim of superior explainability.

## Scope

No completed results are supplied for additional E17 seeds or E17A–E17C
ablations. Cross-paper results live separately in [references](../references/README.md)
because their evaluation protocols are not interchangeable with these runs.
