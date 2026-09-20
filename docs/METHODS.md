# Methods and implementation fidelity

## Task

PatchCamelyon contains 327,680 RGB histopathology patches of size 96 × 96.
A positive label means that the central 32 × 32 region contains tumor tissue.
Tumor elsewhere in a patch does not by itself define a positive label.
The official split contains 262,144 training, 32,768 validation, and 32,768 test
patches. Classification of these patches is distinct from localizing lesions or
assigning a lymph-node stage on a whole-slide image.

## Implementations

| Family | Implementation | Experiment IDs |
|---|---|---|
| SmallCNN | Four convolutional stages and global average pooling; control baseline | E00 |
| ResNet-18/50, DenseNet-121, Inception-v3 | torchvision definitions with a single-logit classification head | E01–E06, E09 |
| D4 G-DenseNet | Native group-convolution reconstruction, not the authors' code | E07 |
| C16 with relative-orientation gating | Inspired by partial-equivariance research; a different gating mechanism | E08 |
| Mean Teacher | EMA teacher and prediction consistency, with 5% labeled patches | E10 |
| SimCLR | NT-Xent pretraining, followed by linear probing or full fine-tuning | E11, E11A, E11B |
| ViT-S/16 | Transformer implementation adapted to 96 × 96 patches | E12 |
| Local-Global ViT | Reconstruction of a local convolutional/global attention design | E13 |
| ResNet-18 + SCE | Symmetric cross-entropy with 20% synthetic training-label noise | E14 |
| LoTeNet | Reconstruction of hierarchical locally orderless tensor networks | E15 |
| Multi-scale attention | Shared encoder and gated scale aggregation, inspired by MMSEN | E16 |
| PCam-D4Loc | Original center-aware D4 architecture | E17 |

The original methods and implementation references are listed in
[references.bib](../references/references.bib). The reconstructed architectures
are not bit-exact reproductions of author repositories; their measured results
should not be presented as the original papers' results.

### What E08 actually implements

The historical configuration/model identifier `partial_se2` is kept for compatibility.
The model uses 16 discrete orientations and learned gates on **relative**
orientations. These gates can be absorbed into the group kernel. They do not
learn the output domain used in Romero and Lohit's partial-equivariance method.
Filter rotations other than multiples of 90° use interpolation, which introduces
numerical approximation. This is a C16 orientation-gated model, not a reproduction
of the published partial-equivariance mechanism.

## PCam-D4Loc

The principal configuration has **42,151 trainable parameters** and three ingredients:

1. **D4 symmetry.** Lifting and group convolutions retain the eight square
   symmetries: four right-angle rotations and four reflected rotations.
2. **Two spatial scales.** Features at 24 × 24 and 12 × 12 are averaged over
   orientation, projected to a common channel width, and fused with a context gate.
3. **A central decision region.** A 1 × 1 head yields a 24 × 24 response map.
   The classifier uses LogSumExp over its center, matching PCam's label region.

For response map $E$, central cells $\Omega_c$, and $\beta=4$, the pooled value is:

$$
s = \frac{1}{\beta}\log\left(\frac{1}{|\Omega_c|}
\sum_{u\in\Omega_c}\exp(\beta E_u)\right).
$$

A learned scale and bias convert this pooled value into a classification logit.
The native map is available through `model.evidence_map(images)`. It is trained
with patch-level classification labels, without a pixel-level segmentation loss.
Its visual alignment with a tumor mask does not establish clinical validity.

The test suite checks D4 stability of maps and invariance of logits in evaluation
mode. Proposed controls remove D4 (E17A), replace central pooling with global
pooling (E17B), or remove multi-scale fusion (E17C). Their configurations are
included, but completed ablation measurements are not supplied.

## Classification metrics

Models return one logit; a sigmoid maps it to a probability. Threshold-based
metrics use **0.5**. The report includes ROC AUC, average precision (AP), accuracy,
balanced accuracy, precision, sensitivity, specificity, F1, negative log-likelihood,
Brier score, and expected calibration error.

AP groups equal scores at a shared threshold. It is not a trapezoidal area under
the precision–recall curve. ROC AUC likewise handles tied ranks. Historical AP
calculations were corrected in the released tables; older per-run JSON files are
not a substitute for those corrected summaries.

## Explainability audit

The tools implement Grad-CAM, Integrated Gradients, RISE, and native PCam-D4Loc
maps. Analyses include mask overlap, pixel-level AP, pointing-game accuracy,
deletion/insertion curves, D4 transformations, and parameter randomization.

The recorded audit sampled 48 test patches using the E06 confusion groups.
Only 25 had usable, non-empty tumor masks; metric-specific sample counts can be
smaller, and are stored in the `n` column. This selected sample is not a population
estimate. In particular, the native E17 map's parameter-randomization check was
undefined in the examined cases, so map stability alone is not evidence of
faithfulness. See [recorded results](../results/README.md).
