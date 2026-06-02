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

## Virtual Clothing

`virtual_clothing_pairing.py` is a separate script for virtual clothing pairing. It uses the same batch controls as the human-item script, but judges whether a reference garment can replace visible clothing on the person.

```bash
python virtual_clothing_pairing.py \
  --gen-dir sample/virtual-clothing/gen \
  --gen-metadata-dir sample/virtual-clothing/gen_metadata \
  --ref-dir sample/virtual-clothing/ref \
  --ref-metadata-dir sample/virtual-clothing/ref_metadata \
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

Outputs:

- `output/virtual-clothing_<batch_id>.json`
- `output/virtual-clothing_<batch_id>.audit.jsonl`

## Hairstyle

`hairstyle_pairing.py` is a separate script for hairstyle transfer pairing. It uses VLM only to judge whether both images have clear usable hair and no strong hat/headwear/accessory, occlusion, crop, blur, or background risk. Gender, face direction, and normal hair covering part of the face are not rejection reasons.

```bash
python hairstyle_pairing.py \
  --gen-dir sample/haitstyle/gen \
  --gen-metadata-dir sample/haitstyle/gen_metadata \
  --ref-dir sample/haitstyle/ref \
  --ref-metadata-dir sample/haitstyle/ref_metadata \
  --output-dir output \
  --target-count 100 \
  --batch-id exp_v1 \
  --seed 20260602 \
  --max-ref-attempts-per-gen 5 \
  --score-threshold 0.75 \
  --base-url http://10.154.39.57:8001/v1 \
  --api-key 123456 \
  --model-name gemma-4-31B-it
```

Outputs:

- `output/hairstyle_<batch_id>.json`
- `output/hairstyle_<batch_id>.audit.jsonl`

## Expression

`expression_pairing.py` is a separate script for facial expression transfer pairing. It uses VLM to judge whether the target face is editable, whether the reference expression regions are clear, and whether eyes, eyebrows, mouth, and facial contour are not strongly blocked, blurred, or cropped. Ambiguous emotion labels are allowed as long as the facial expression is visible.

```bash
python expression_pairing.py \
  --gen-dir sample/expression/gen \
  --gen-metadata-dir sample/expression/gen_metadata \
  --ref-dir sample/expression/ref \
  --ref-metadata-dir sample/expression/ref_metadata \
  --output-dir output \
  --target-count 100 \
  --batch-id exp_v1 \
  --seed 20260602 \
  --max-ref-attempts-per-gen 5 \
  --score-threshold 0.75 \
  --base-url http://10.154.39.57:8001/v1 \
  --api-key 123456 \
  --model-name gemma-4-31B-it
```

Outputs:

- `output/expression_<batch_id>.json`
- `output/expression_<batch_id>.audit.jsonl`
