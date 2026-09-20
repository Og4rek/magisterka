# Contributing

Small, reproducible fixes are welcome. Please describe the observed behavior,
the expected behavior, your Python/PyTorch versions, and a minimal example.
Do not attach patient material, datasets, credentials, or large checkpoints.

For code changes, use an editable installation with the `test` extra and run:

```bash
python -m compileall -q src scripts
python -m pytest -q -k "not one_epoch and not supervised_fit and not compile_happens"
```

The complete `pytest` suite additionally runs tiny synthetic training/resumption
tests. Run those explicitly when changing training-loop behavior. Neither suite
requires the real PCam dataset. Dataset downloads and full experiments must be
started separately.

Keep metric definitions, official split semantics, and configuration identifiers
stable. If a result changes, explain the cause and preserve its provenance.
Do not silently overwrite recorded results with a new seed or a different protocol.

Research comparisons should state the dataset variant, split, label budget,
initialization, checkpoint-selection rule, and implementation fidelity. An
improvement on a reused test set should be treated as exploratory.
