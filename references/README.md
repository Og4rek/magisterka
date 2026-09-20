# References and literature results

This is a compact export of the literature review supporting the implementations
and experiment comparisons. It contains bibliographic metadata and extracted
numerical evidence, not redistributed full-text papers or manuscript drafts.

- [references.bib](references.bib): selected dataset, model, optimization,
  calibration, and explainability references.
- [literature_results.csv](literature_results.csv): 104 model/configuration rows
  extracted from 14 publications, including source URLs and evidence locators.

## Read the protocol before the score

Use `dataset_variant`, `split_or_protocol`, `learning_regime`, `protocol_family`,
`comparability`, `verification`, and `evidence_locator` alongside each metric.
An empty metric is unavailable, not zero. Columns ending in `_pct` use percentages;
ROC AUC and NLL do not. A row describes a reported configuration, not a separate
paper. The table is not a leaderboard across incompatible settings.

The historical `study_id` column identifies individual result rows; group by
`study_title` to obtain the 14 publications. Some multi-dataset studies also
contribute contextual results outside PCam; inspect `dataset_variant` before
selecting rows for a PCam comparison.

In particular, distinguish official PCam from modified Kaggle splits, smaller
subsets, and higher-resolution variants. Published values are separate from this
repository's experiments in [results/classification.csv](../results/classification.csv).
Access to a source and the amount of implementation detail differ between papers.

## Starting points

- Veeling et al., *Rotation Equivariant CNNs for Digital Pathology*:
  [paper](https://arxiv.org/abs/1806.03962), [PCam dataset](https://github.com/basveeling/pcam).
- Cohen and Welling, *Group Equivariant Convolutional Networks*:
  [paper](https://proceedings.mlr.press/v48/cohenc16.html).
- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*:
  [paper](https://proceedings.mlr.press/v119/chen20j.html).
- Tarvainen and Valpola, *Mean teachers are better role models*:
  bibliographic record and source link in `references.bib`.
- Romero and Lohit, *Learning Partial Equivariances from Data*:
  bibliographic record in `references.bib`; see the explicit E08 fidelity caveat
  in [Methods](../docs/METHODS.md).

Please cite the original papers and dataset when using their methods or data.
The repository's MIT license does not relicense third-party publications or datasets.
