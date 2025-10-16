# Benchmark Inventory and Layout Plan

This note describes the expected benchmark layout for BIRD and Spider 2.0, how it maps to CHESS-style indices for the Information Retriever (IR), and what to inspect after extraction. It also records current workspace findings to ensure reproducibility.

## Expected Locations

- Root (EPFL): `EPFL_hyunjun/benchmark/`
- Archives expected (not tracked by Git):
  - `benchmark/bird.zip`
  - `benchmark/Spider2.zip`
- After extraction (example):
  - `benchmark/bird/`
  - `benchmark/Spider2/`

## CHESS Layout Requirements

For IR preprocessing (MinHash LSH + Chroma Vector DB), data should be arranged per database id:

```
<DB_ROOT_PATH>/<mode>_databases/<db_id>/
  <db_id>.sqlite
  preprocessed/
    <db_id>_lsh.pkl
    <db_id>_minhashes.pkl
  context_vector_db/
    (Chroma persisted index files)
```

Recommended EPFL settings:
- `DB_ROOT_PATH = EPFL_hyunjun/benchmark`
- `mode = dev` (for development runs)

## Spider 2.0 Ground Truth Tables

Spider 2.0 provides ground-truth table information per question. We will use this metadata to measure IR matching quality directly, e.g.:
- Table Recall@k: fraction of gold tables covered by IR-pruned schema
- Column Recall (optional): fraction of gold columns included
- Value/Context Support (optional): overlap between retrieved value examples/descriptions and those implied by gold queries

A small evaluation harness can compare CHESS IR artifacts (`similar_columns`, `schema_with_descriptions`) against the Spider 2.0 labels to produce summary metrics.

## Current Workspace Scan (auto-generated status)

- Archives found: none detected under `EPFL_hyunjun/benchmark/` at authoring time.
- Action: place `bird.zip` and `Spider2.zip` under `EPFL_hyunjun/benchmark/` and extract them:

```powershell
# From EPFL_hyunjun/
$bench = Join-Path (Get-Location) 'benchmark'
Expand-Archive -Force -LiteralPath (Join-Path $bench 'bird.zip') -DestinationPath (Join-Path $bench 'bird')
Expand-Archive -Force -LiteralPath (Join-Path $bench 'Spider2.zip') -DestinationPath (Join-Path $bench 'Spider2')
```

Then reorganize into CHESS layout and run preprocessing:

```bash
# Example: CHESS preprocessing (from ../CHESS)
python src/preprocess.py \
  --db_root_directory "<absolute path>/EPFL_hyunjun/benchmark/dev_databases" \
  --db_id all --use_value_description True --signature_size 20 --n_gram 3 --threshold 0.01
```

## Post-extraction Checklist

- BIRD
  - Presence of SQLite databases per `db_id`
  - JSON files (e.g., `dev.json`, tables) and schema metadata
  - After preprocessing: `preprocessed/` and `context_vector_db/` per `db_id`
- Spider 2.0
  - Dev/test splits, ground-truth SQL and table labels
  - Verify `db_id` names align with SQLite filenames

## Notes

- Large archives are excluded by `.gitignore` and should not be committed to the repository.
- Once indices are built, set `IRConfig.db_root_path = "benchmark"` and `IRConfig.data_mode = "dev"` to enable IR in the pipeline.
