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

`hairstyle_pairing.py` is a separate script for hairstyle transfer pairing. It uses VLM to judge whether both images have clear usable hair, whether the target and reference hairstyles are visibly different enough, and whether there is no strong hat/headwear/accessory, occlusion, crop, blur, or background risk. Gender, face direction, and normal hair covering part of the face are not rejection reasons.

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

`expression_pairing.py` is a separate script for facial expression transfer pairing. It uses VLM to judge whether the target face is editable, whether the reference expression regions are clear, whether eyes, eyebrows, mouth, and facial contour are not strongly blocked, blurred, or cropped, and whether the two expressions are visibly different enough to be useful. Pairing is ref-driven and softly balances accepted pairs across reference expression categories.

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

## Makeup

`makeup_pairing.py` is a separate script for makeup transfer pairing. It uses VLM to judge whether the target face and key makeup regions are clear, whether the reference makeup is visible and mainly on the face, and whether masks, sunglasses, hands, hair, props, strong shadow, blur, or overexposure create transfer risk. It also keeps accepted `gen` images near a 3:7 male:female ratio within the current batch; `ref` images are still selected randomly while balancing per-batch usage.

```bash
python makeup_pairing.py \
  --gen-dir sample/makeup/gen \
  --gen-metadata-dir sample/makeup/gen_metadata \
  --ref-dir sample/makeup/ref \
  --ref-metadata-dir sample/makeup/ref_metadata \
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

- `output/makeup_<batch_id>.json`
- `output/makeup_<batch_id>.audit.jsonl`

## Person Texture

`person_texture_pairing.py` is a separate script for transferring reference textures onto a garment worn by the person in image 1. It uses VLM to choose a clear editable target garment, confirm that image 2 is a usable reference texture, reject texture sources that are too similar to the original garment texture, and check that reference texture complexity is compatible with the target garment's editable surface. Slight blur on the target garment is allowed when the garment remains understandable.

```bash
python person_texture_pairing.py \
  --gen-dir sample/person-texture/gen \
  --gen-metadata-dir sample/person-texture/gen_metadata \
  --ref-dir sample/person-texture/ref \
  --ref-metadata-dir sample/person-texture/ref_metadata \
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

- `output/person-texture_<batch_id>.json`
- `output/person-texture_<batch_id>.audit.jsonl`
