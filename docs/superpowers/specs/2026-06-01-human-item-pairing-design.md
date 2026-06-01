# Human-Item Pairing Design

Date: 2026-06-01

## Goal

Build a batch pairing pipeline for the human-item multi-image editing data pool. The pipeline pairs `gen` person images with `ref` object images, then uses a VLM to decide whether each candidate pair is suitable for hand-object interaction training. Accepted pairs produce image-editing prompts and are written in the existing training JSON format.

The pipeline should balance:

- strict training constraints, especially matching image dimensions;
- random and diverse pairing;
- balanced `ref` usage within a single batch;
- bounded VLM cost and runtime;
- auditable decisions for later quality analysis.

## Inputs

The expected data layout follows the current sample structure:

```text
sample/
  gen/
  gen_metadata/
  ref/
  ref_metadata/
```

Each metadata JSON is keyed by image filename stem. For example:

```text
gen/000000.jpg
gen_metadata/000000.json
ref/000000.jpg
ref_metadata/000000.json
```

The implementation should not assume `gen` and `ref` are paired by identical filename. Filenames only identify individual images and metadata.

## Required Pairing Rules

### Dimension Match

A `gen` image and a `ref` image may be paired only when their dimensions match.

The dimension key should be read from metadata:

```text
resized_width x resized_height
```

If resized dimensions are missing, the implementation may fall back to `preferred_resolution`, then to `original_width` and `original_height`. Any fallback should be recorded in the audit file.

### Metadata Pre-Filter

The metadata pre-filter keeps VLM calls focused on likely useful pairs.

For `gen`, require:

- `original_annotation.annotation.hand_hold_feasible == "yes"` when present;
- `filter_rules` already indicate the image belongs to the curated pool;
- valid dimensions.

For `ref`, require:

- `suitable_for_holding == true`;
- `should_use == true`;
- `confidence == "high"` when present;
- `holdability` is one of the allowed hand-object interaction types, such as `handle`, `carryable`, or `pet_holdable`;
- valid dimensions.

These checks should be configurable enough to relax later, but the default should be conservative.

## Batch Parameters

The script should expose these parameters:

```text
--target-count
--batch-id
--seed
--max-ref-attempts-per-gen
--score-threshold
--workers
--allow-gen-reuse
--output-dir
```

Recommended defaults:

```text
target-count: required
batch-id: auto timestamp when omitted
seed: auto generated when omitted
max-ref-attempts-per-gen: 5
score-threshold: 0.75
workers: 8 or 12
allow-gen-reuse: false
```

The effective seed must be written to the audit output so a batch can be reproduced.

## Output Naming

Each run should produce a versioned training JSON and audit file.

If `--batch-id exp_v1` is provided:

```text
human-item_exp_v1.json
human-item_exp_v1.audit.jsonl
```

If no batch ID is provided:

```text
human-item_YYYYMMDD-HHMMSS.json
human-item_YYYYMMDD-HHMMSS.audit.jsonl
```

The main JSON must follow the existing format:

```json
[
  {
    "cond_1": "path/to/gen.jpg",
    "cond_2": "path/to/ref.jpg",
    "prompt": "Let the person in image 1 hold ..."
  }
]
```

The audit JSONL should include accepted and rejected attempts. Each line should contain at least:

```json
{
  "batch_id": "exp_v1",
  "seed": 20260601,
  "size_key": "1248x832",
  "gen_path": "path/to/gen.jpg",
  "ref_path": "path/to/ref.jpg",
  "attempt_index_for_gen": 1,
  "suitable": true,
  "score": 0.86,
  "reason": "The person has visible hands and the object can be naturally held.",
  "prompt": "Let the person in image 1 hold ..."
}
```

Failures caused by missing metadata, missing images, invalid model output, or VLM request errors should also be recorded with an error field.

## Pairing Algorithm

### Indexing

1. Load all `gen` metadata and all `ref` metadata.
2. Resolve the corresponding image path for each metadata item.
3. Apply metadata pre-filters.
4. Group valid `gen` and `ref` items by `size_key`.
5. Discard size buckets that do not contain at least one valid `gen` and one valid `ref`.

### Gen Selection

For each batch:

1. Build a list of eligible `gen` items from all valid size buckets.
2. Shuffle the list with the run RNG.
3. Process each `gen` at most once by default.
4. If `allow-gen-reuse` is true and `target-count` cannot be reached, start another shuffled pass over the same eligible gen pool.

Default behavior should prioritize unique `gen` images within the batch.

### Ref Selection With Balanced Usage

Within each size bucket, keep a per-batch `ref_usage_count`.

When selecting candidates for one `gen`:

1. Consider only refs from the same `size_key`.
2. Exclude refs already attempted for that gen.
3. Find the minimum current usage count among remaining refs.
4. Randomly choose among refs with that minimum usage count.
5. Submit the candidate pair to VLM.
6. Increase `ref_usage_count` only if the pair is accepted.
7. If rejected, record the rejection and try another ref until `max-ref-attempts-per-gen` is reached.

This "least-used first, random within the least-used set" strategy keeps `ref` usage balanced within the batch while preserving randomness.

## VLM Decision Contract

The VLM should receive:

- `gen` image;
- `ref` image;
- compact `gen` metadata;
- compact `ref` metadata;
- a system prompt describing suitability criteria and output schema.

The VLM should return strict JSON:

```json
{
  "suitable": true,
  "score": 0.86,
  "reason": "Short reason.",
  "action": "hold",
  "object_description": "a clear glass jar filled with snack mix",
  "prompt": "Let the person in image 1 hold a clear glass jar filled with snack mix shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged."
}
```

The script accepts a pair only when:

- `suitable == true`;
- `score >= score_threshold`;
- `prompt` is non-empty;
- `prompt` follows the required hand-object interaction format.

Rejected pairs should not appear in the main training JSON, but they should appear in the audit JSONL.

## VLM Suitability Criteria

The VLM should judge whether the person in `gen` can naturally interact with the object in `ref`.

The decision should consider:

- whether at least one hand or arm is visible and usable;
- whether the person's pose can naturally support the object;
- whether the object scale is plausible relative to the person;
- whether the required interaction is simple enough for image editing;
- whether adding the object would cause severe occlusion or unrealistic contact;
- whether the correct interaction type is clear, such as holding, carrying, gripping, holding by a handle, holding in both hands, or carrying over the shoulder.

The VLM should be conservative. If the pair would require major pose changes, impossible hand placement, or unclear contact, it should reject the pair.

## Efficiency Controls

The pipeline should bound VLM usage with:

- `max-ref-attempts-per-gen`;
- `target-count`;
- worker concurrency;
- metadata pre-filtering;
- strict JSON parsing and retry limits;
- optional per-run limit for total VLM calls.

Recommended first-run settings:

```text
max-ref-attempts-per-gen = 5
score-threshold = 0.75
workers = 8
```

If acceptance rate is low, first inspect audit reasons before increasing retries.

## Error Handling

The script should continue processing when a single pair fails.

Record these cases in audit:

- metadata file missing;
- image file missing;
- invalid or missing dimensions;
- no same-size ref candidates;
- VLM request failure after retries;
- invalid VLM JSON;
- VLM rejection below threshold.

The final log should report:

- target count;
- accepted count;
- rejected VLM attempts;
- skipped gen count;
- failed request count;
- output paths;
- effective seed.

## Open Decisions

The current design assumes:

- ref usage should be balanced only within the current batch;
- gen images should be unique by default within the current batch;
- ref images may be reused if the target count is larger than available same-size refs, but the least-used strategy keeps reuse balanced;
- rejected ref attempts do not increase ref usage count.

These assumptions match the current requirements and can be changed later through parameters if needed.
