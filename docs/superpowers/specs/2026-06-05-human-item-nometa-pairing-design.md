# Human-Item No-Metadata Pairing Design

Date: 2026-06-05

## Goal

Create a separate human-item pairing pipeline for datasets that only have two image folders:

- `gen`: person images
- `ref`: reference object images

The image filenames are not coordinated and there is no metadata. The script should use a VLM to judge candidate pairs directly from images, center-crop each accepted reference image to the paired gen image size, copy the paired outputs to user-specified locations, and write a training JSON containing `cond_1`, `cond_2`, and `prompt`.

The existing metadata-based human-item script remains unchanged.

## CLI

The new script should be named `human_item_pairing_nometa.py`.

Required inputs:

```text
--gen-dir
--ref-dir
--target-count
```

Output path mode C is supported:

- If explicit paths are provided, use them:
  - `--output-gen-dir`
  - `--output-ref-dir`
  - `--output-json`
  - optional `--audit-jsonl`
- Otherwise use `--output-root` and derive:
  - `<output-root>/gen`
  - `<output-root>/ref`
  - `<output-root>/human-item_<batch_id>.json`
  - `<output-root>/human-item_<batch_id>.audit.jsonl`

Batch and VLM controls should mirror the previous script where useful:

```text
--batch-id
--seed
--max-ref-attempts-per-gen
--score-threshold
--workers
--allow-gen-reuse
--base-url
--api-key
--model-name
--max-retries
--temperature
--max-tokens
--dry-run-accept-all
```

## Image Handling

The script scans `gen` and `ref` directories directly for common image extensions:

```text
.jpg .jpeg .png .bmp .webp .tif .tiff
```

Each `gen` image uses its real pixel size. For a candidate pair, the `ref` image is transformed to exactly the `gen` size before VLM judgment and before accepted output:

1. Open the ref image as RGB.
2. If the ref image is smaller than the gen canvas in either dimension, resize it proportionally until it covers the gen width and height.
3. Center-crop to the gen width and height.

Accepted outputs are saved as PNG files, starting from:

```text
00000.png
00001.png
00002.png
```

The saved `gen` file is the original gen image converted to PNG. The saved `ref` file is the cropped reference image.

## Pairing Algorithm

The script should not rely on filenames or metadata.

1. Build sorted image item lists for `gen` and `ref`.
2. Shuffle gen traversal with the run seed.
3. For each gen, choose ref candidates by balanced per-batch ref usage.
4. Exclude refs already attempted for the current gen.
5. Submit gen plus cropped ref to the VLM.
6. Accept when the VLM decision passes the score threshold and prompt validation.
7. Save paired output images and append a JSON row.
8. Stop at `target-count`, or earlier if no more acceptable pairs exist.

Default behavior processes each gen once. `--allow-gen-reuse` may start additional shuffled passes.

## VLM Decision Contract

The VLM receives:

- Image 1: the gen person image
- Image 2: the center-cropped ref object image, already matching image 1 size

The standard is intentionally lighter than the metadata-based script. The VLM should mainly decide:

- whether image 1 contains a person with enough visible body, arm, hand, or nearby support context for a plausible human-object interaction;
- whether image 2 contains a clear main object that can reasonably be held, carried, supported, worn in hand, or used by the person;
- whether the interaction can be achieved with minor hand, wrist, forearm, or placement changes while keeping the rest of image 1 stable.

Hard rejections should still cover close-up headshots with no usable hand or arm context, objects that are clearly huge/fixed/immovable, unreadable reference objects, and interactions requiring a full pose rebuild.

The prompt format must remain consistent with the previous human-item script:

```text
Let the person in image 1 [hand-object action phrase] [object description] shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged.
```

The VLM returns strict JSON:

```json
{
  "suitable": true,
  "score": 0.86,
  "reason": "short reason",
  "action": "short hand-object action phrase, or empty string if unsuitable",
  "object_description": "main object description, or empty string if unsuitable",
  "prompt": "final prompt, or empty string if unsuitable"
}
```

The script accepts a pair only when:

- `suitable == true`
- `score >= score_threshold`
- `prompt` is non-empty
- `prompt` follows the previous human-item prompt format

## Outputs

The main JSON contains paths to the newly written output images, not the original source paths:

```json
[
  {
    "cond_1": "path/to/output/gen/00000.png",
    "cond_2": "path/to/output/ref/00000.png",
    "prompt": "Let the person in image 1 hold ..."
  }
]
```

The audit JSONL records all major events:

- image scan failures
- crop/resize failures
- pair accepted or rejected
- VLM errors
- original gen/ref paths
- output gen/ref paths for accepted pairs
- gen size and original/cropped ref size
- score, reason, action, object description, prompt
- batch id and seed

## Tests

Focused tests should cover:

- directory image scanning without metadata
- center crop when ref is larger than gen
- resize-to-cover then center crop when ref is smaller than gen
- deterministic `00000.png` output naming
- output JSON points to copied/cropped output files
- prompt acceptance keeps the previous human-item format
- dry-run accept-all can produce a small batch without VLM credentials
