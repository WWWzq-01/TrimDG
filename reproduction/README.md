# Bounded train/state reproduction

This is **not** the author's missing `evaluate_models_utils` module. The original
`trim_link.py` still fails clearly without that evaluator. `train_state.py` imports
the real `MemoryModel`, `GraphAttentionEmbedding`, `MultiHeadAttention` (through
the model), `MergeLayer`, negative sampler and optimizer. No validation, test,
AP/AUC or replacement TGN is supplied.

U0 is `cceffe826df3b8712c646ca9f46def5ea0f079e3`. U1 adds an explicit data root,
preprocessing input/output paths and saved outputs, the missing `bypass` option,
CPU device guard, and prevents `recent` from executing the unused `our` window
estimator. The shared positive entropy selector initially retains the published
top-k order. U2 is a separate change preserving that exact set while restoring
original event order before the memory model consumes it. Both real study
variants must use U2. The unsorted version is only a tiny counterexample, not a
performance baseline.

```sh
python preprocess_data/preprocess_data.py --dataset_name wikipedia \
  --input-csv /data/raw/wikipedia.csv \
  --output-dir /data/isolated/processed_data/wikipedia
python reproduction/train_state.py --data-root /data/isolated \
  --output-dir /results/T1 --variant T1 --device cuda:0 \
  --max-batches 128 --epochs 2 --ledger /results/shared-ledger.jsonl \
  --gpu-seconds-budget 5400 --source-u1 U1_COMMIT --source-u2 U2_COMMIT
OMP_NUM_THREADS=1 python -m unittest reproduction.test_train_state
```

Use the benchmark study runner for the actual paired execution and cumulative
GPU-time enforcement. It captures stdout separately. The driver additionally
limits its GPU lifetime and durably counts each batch start in the shared ledger
before doing training, including failed starts, up to 512. Optimizer completion
is recorded immediately after `step`, before detach/checking can fail. Existing
result files and processed data are never overwritten. Python 3.8 is used for
the author's `random.sample(set)` split without changing its node holdout.

The fixed configuration is seed 0, TGN/recent, one layer, 10 neighbors, two heads,
time dimension 100, batch 200, Adam lr 1e-4, dropout 0.1, cache 0. Features and
memory retain the author's 172 dimensions. T0 trains all prefix targets; T1
scores/trains all E0 targets and consumes entropy top-50% in E1. Every epoch
resets memories and pending messages, retains the selector table, and uses the
same complete train adjacency. The two epochs repeat the first 128 batches,
not a full first epoch. Negative forward precedes positive forward, BCELoss and
Adam perform a real update in every batch, then messages/memories are detached.

Outputs are `summary.json`, `epoch_metrics.csv`, `observations.json`. The
observations count actual sampler returns, node/memory/edge gather rows (including
padding), full global-memory clone work, raw-message event sources and stores,
and detach dictionary traversal. Bounded skipped-event probes query the same
full train adjacency at a legal later cutoff. Being in adjacency does not imply
that an event published a message; another event writing the same node does not
establish publication. Logical rows/bytes are not physical I/O. Instrumentation,
finite checks and durable ledger writes are included in the measured loop, so
these are not official performance timings. Whole feature tables remain resident.

Small tests use explicit synthetic fixtures only to validate code. They never
substitute for Wikipedia data or create study quality measurements. Full quality,
E5 selector refresh, cache/our/GIB/BaM and all other models are untested here.
