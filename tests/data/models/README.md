# Synthetic Component 06 sample

`make_fixture.py` is the committed, deterministic sample definition. It writes the
published Component 02 CSV, Component 03 split/seed, and Component 04 serialized PyG
interfaces into a caller-selected directory. It is **software test data only** and is
not a substitute for any owner's production package.

The sample contains five invented, leading-zero station IDs, six UTC hours in 2022,
one deliberately absent observation, five explicit folds, and reversed graph node
ordering. The small complete graph and four fixed positional features are synthetic;
they do not claim to reproduce a scientific LPE or Sydney graph.

The test model uses four hidden channels, one layer, two epochs and a software-only
learning rate of `0.05`. This rate makes the short training fixture produce positive,
varying outputs, so exact replay tests cannot pass merely because negative predictions
were all clamped to zero. The production configuration retains its prescribed
`0.0003` learning rate and other inherited settings.

From the repository root:

```console
uv run --locked python tests/data/models/make_fixture.py experiments/models-fixture
uv run --locked python -m src.models.run experiments/models-fixture/config.yaml --check-inputs
uv run --locked python -m src.models.run experiments/models-fixture/config.yaml --output-dir experiments/models-fixture-run
uv run --locked python -m src.models.run experiments/models-fixture/config.yaml --predict-from experiments/models-fixture-run/best.pt --output-dir experiments/models-fixture-replay
```

Choose fresh output paths: existing directories are never overwritten. Generated
fixtures, checkpoints and runs belong under ignored `experiments/`.

Add `--modalities radar satellite` to the fixture command to include both sensors, or
name either sensor alone. Each adds two invented nodes with synthetic bidirectional
connections to the gauges. Satellite includes the unavailable payload `[999,0]` to
verify that its original availability flag survives while the computational input is
masked. These edges do not demonstrate geographic attachment or source-data alignment.

`tests/test_model_training.py` verifies exact CPU prediction and parameter repeatability,
actual optimizer parameter updates, a known nonzero checkpoint replay, input relocation,
the lowest-validation-loss checkpoint, all five folds, exact target rows, absent-hour
handling, stable identities through batching, target-feature masking, and rejection of
swapped fold checkpoints or time-varying positional channels.

Validation retains the inherited clamp-before-loss policy. On a different configuration,
negative outputs can therefore produce a tied zero plateau, retaining an early
checkpoint even while unclamped training loss improves. The run report records this
policy and flags an all-zero HGNN result; it is a review point, not scientific evidence
that the default settings are effective.
