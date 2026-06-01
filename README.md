# human_item_pair

Batch pairing pipeline for human-item multi-image editing data.

## Features

- Matches `gen` and `ref` images only when their metadata dimensions are identical.
- Randomizes `gen` traversal with a reproducible seed.
- Balances `ref` usage within each batch.
- Uses a VLM to judge pair suitability and generate hand-object interaction prompts.
- Writes both training JSON and audit JSONL outputs.

## Example

```bash
python human_item_pairing.py \
  --gen-dir sample/gen \
  --gen-metadata-dir sample/gen_metadata \
  --ref-dir sample/ref \
  --ref-metadata-dir sample/ref_metadata \
  --output-dir output \
  --target-count 100 \
  --batch-id exp_v1 \
  --seed 20260601 \
  --max-ref-attempts-per-gen 5 \
  --score-threshold 0.75 \
  --base-url http://10.154.39.57:8001/v1 \
  --api-key 123456 \
  --model-name gemma-4-31B-it
```

## Dry Run

Use dry-run mode to test indexing, pairing, and output writing without calling a VLM:

```bash
python human_item_pairing.py \
  --gen-dir sample/gen \
  --gen-metadata-dir sample/gen_metadata \
  --ref-dir sample/ref \
  --ref-metadata-dir sample/ref_metadata \
  --output-dir output \
  --target-count 1 \
  --batch-id dryrun_sample \
  --seed 20260601 \
  --dry-run-accept-all
```

## Outputs

- `output/human-item_<batch_id>.json`
- `output/human-item_<batch_id>.audit.jsonl`

