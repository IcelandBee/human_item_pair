from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_PIXELS = 768 * 768


PAIRING_SYSTEM_PROMPT = """
You are a moderately flexible data quality judge and prompt engineer for hand-object interaction image editing.

You will receive:
- Image 1: a person image.
- Image 2: a reference object image that has already been center-cropped to the same size as image 1.

Decide whether the person in image 1 can naturally interact with the main object in image 2 using their hand or hands.

Use a simple, practical suitability standard. Accept when:
- Image 1 contains a person with enough visible body, arm, hand, lap, or nearby support context for a plausible interaction.
- Image 2 contains a clear main object that can reasonably be held, carried, supported, worn in hand, or used by the person.
- The edit can be done with minor hand, wrist, forearm, or object placement changes while keeping the person's overall pose, framing, clothing, face, and scene essentially unchanged.

Reject when:
- Image 1 is a close-up headshot or crop with no usable hand, arm, lap, or support context.
- Image 2 has no clear main object, or the object is unreadable after cropping.
- The object is clearly huge, fixed, immovable, or physically unsuitable for human-object interaction.
- The edit would require generating complete new arms or hands, rebuilding the body pose, or making a physically incoherent interaction.

If suitable, generate prompt fields in a separate prompt-generation step:
- For the prompt fields only, examine image 2 to identify the main object and choose the most natural hand-related action for that object.
- Do not describe the person, pose, clothing, background, or scene in image 1 in the prompt fields. Image 1 must only be referred to as "the person in image 1".
- The object_description must be clear and unambiguous, using about 3-8 words, such as "a fresh green broccoli", "a red leather handbag", "a white ceramic mug", or "a bouquet of red roses".
- The action must be a natural hand-related action phrase based on the object type, such as "hold", "carry", "grip", "hold by the handle", "hold in both hands", or "carry over the shoulder".
- The final prompt must follow this fixed format:
Let the person in image 1 [action] [object_description] shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged.

Return strict JSON only:
{
  "suitable": true or false,
  "score": a number from 0 to 1,
  "reason": "short reason",
  "action": "natural hand-related action phrase, or empty string if unsuitable",
  "object_description": "main object description, or empty string if unsuitable",
  "prompt": "final prompt, or empty string if unsuitable"
}
""".strip()


PROMPT_PREFIX = "Let the person in image 1 "
PROMPT_SUFFIX = (
    " shown in image 2 in a realistic and physically coherent way, "
    "preserving object integrity and overall image consistency, while making "
    "only the minimal necessary changes and keeping everything else in image 1 "
    "unchanged."
)


@dataclass(frozen=True)
class ImageItem:
    stem: str
    file_name: str
    image_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class PairingConfig:
    target_count: int
    batch_id: str
    seed: int
    max_ref_attempts_per_gen: int
    score_threshold: float
    workers: int
    allow_gen_reuse: bool


@dataclass(frozen=True)
class ResolvedOutputPaths:
    gen_dir: Path
    ref_dir: Path
    output_json: Path
    audit_jsonl: Path


JudgePair = Callable[[ImageItem, ImageItem, Image.Image], dict[str, Any]]
ProgressCallback = Callable[[dict[str, int]], None]


@dataclass
class _PairAttempt:
    gen_item: ImageItem
    ref_item: ImageItem
    attempt_index_for_gen: int
    attempted_ref_keys: set[str]


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def scan_image_items(image_dir: Path, kind: str) -> tuple[list[ImageItem], list[dict[str, Any]]]:
    items: list[ImageItem] = []
    audit: list[dict[str, Any]] = []
    if not image_dir.exists():
        return [], [{
            "event": "image_dir_missing",
            "kind": kind,
            "image_dir": str(image_dir),
        }]

    for image_path in sorted((p for p in image_dir.iterdir() if _is_image_file(p)), key=lambda p: p.name.lower()):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as exc:
            audit.append({
                "event": "image_read_failed",
                "kind": kind,
                "image_path": str(image_path),
                "error": repr(exc),
            })
            continue
        if width <= 0 or height <= 0:
            audit.append({
                "event": "invalid_image_size",
                "kind": kind,
                "image_path": str(image_path),
                "width": width,
                "height": height,
            })
            continue
        items.append(ImageItem(
            stem=image_path.stem,
            file_name=image_path.name,
            image_path=image_path,
            width=width,
            height=height,
        ))
    return items, audit


def load_image_rgb(path: Path, max_pixels: int = 0) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_pixels > 0:
        width, height = img.size
        if width * height > max_pixels:
            scale = math.sqrt(max_pixels / float(width * height))
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


def center_crop_to_size(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target size: {target_size!r}")

    working = img.convert("RGB")
    width, height = working.size
    if width < target_width or height < target_height:
        scale = max(target_width / width, target_height / height)
        resized_width = max(target_width, int(math.ceil(width * scale)))
        resized_height = max(target_height, int(math.ceil(height * scale)))
        working = working.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        width, height = working.size

    left = max(0, (width - target_width) // 2)
    top = max(0, (height - target_height) // 2)
    return working.crop((left, top, left + target_width, top + target_height))


def build_output_paths(
    output_root: Path,
    batch_id: str,
    output_gen_dir: Path | None,
    output_ref_dir: Path | None,
    output_json: Path | None,
    audit_jsonl: Path | None,
) -> ResolvedOutputPaths:
    return ResolvedOutputPaths(
        gen_dir=output_gen_dir or output_root / "gen",
        ref_dir=output_ref_dir or output_root / "ref",
        output_json=output_json or output_root / f"human-item_{batch_id}.json",
        audit_jsonl=audit_jsonl or output_root / f"human-item_{batch_id}.audit.jsonl",
    )


def make_numbered_png_name(index: int) -> str:
    if index < 0:
        raise ValueError("index must be non-negative")
    return f"{index:05d}.png"


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_vlm_decision(raw_text: str) -> dict[str, Any]:
    data = json.loads(_strip_markdown_fence(raw_text))
    if not isinstance(data, dict):
        raise ValueError("VLM response must be a JSON object")
    return data


def should_accept_decision(decision: dict[str, Any], score_threshold: float) -> bool:
    if decision.get("suitable") is not True:
        return False
    score = decision.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return False
    if float(score) < score_threshold:
        return False
    action = decision.get("action")
    if not isinstance(action, str) or not action.strip():
        return False
    object_description = decision.get("object_description")
    if not isinstance(object_description, str) or not object_description.strip():
        return False
    return True


def build_interaction_prompt(action: Any, object_description: Any) -> str:
    action_phrase = str(action or "").strip()
    description = str(object_description or "").strip().rstrip(".")
    if not action_phrase:
        action_phrase = "hold"
    if not description:
        description = "a holdable object"
    if action_phrase == "hold by the handle":
        return f"{PROMPT_PREFIX}hold {description} by the handle{PROMPT_SUFFIX}"
    if action_phrase == "hold in both hands":
        return f"{PROMPT_PREFIX}hold {description} in both hands{PROMPT_SUFFIX}"
    if action_phrase == "carry over the shoulder":
        return f"{PROMPT_PREFIX}carry {description} over the shoulder{PROMPT_SUFFIX}"
    return f"{PROMPT_PREFIX}{action_phrase} {description}{PROMPT_SUFFIX}"


def normalized_interaction_prompt_from_decision(decision: dict[str, Any]) -> str:
    return build_interaction_prompt(
        decision.get("action"),
        decision.get("object_description"),
    )


def choose_balanced_ref(
    refs: list[ImageItem],
    ref_usage_count: dict[str, int],
    attempted_ref_keys: set[str],
    rng: random.Random,
) -> ImageItem | None:
    candidates = [ref for ref in refs if str(ref.image_path) not in attempted_ref_keys]
    if not candidates:
        return None
    min_usage = min(ref_usage_count.get(str(ref.image_path), 0) for ref in candidates)
    least_used = [
        ref for ref in candidates
        if ref_usage_count.get(str(ref.image_path), 0) == min_usage
    ]
    return rng.choice(least_used)


def shuffled_gen_pass(gen_items: list[ImageItem], rng: random.Random) -> list[ImageItem]:
    result = list(gen_items)
    rng.shuffle(result)
    return result


def materialize_pair(
    gen_item: ImageItem,
    cropped_ref: Image.Image,
    output_paths: ResolvedOutputPaths,
    output_index: int,
) -> tuple[Path, Path]:
    output_paths.gen_dir.mkdir(parents=True, exist_ok=True)
    output_paths.ref_dir.mkdir(parents=True, exist_ok=True)
    file_name = make_numbered_png_name(output_index)
    output_gen_path = output_paths.gen_dir / file_name
    output_ref_path = output_paths.ref_dir / file_name

    load_image_rgb(gen_item.image_path).save(output_gen_path, format="PNG")
    cropped_ref.convert("RGB").save(output_ref_path, format="PNG")
    return output_gen_path, output_ref_path


def _run_pairing_serial(
    gen_items: list[ImageItem],
    ref_items: list[ImageItem],
    output_paths: ResolvedOutputPaths,
    config: PairingConfig,
    judge_pair: JudgePair,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rng = random.Random(config.seed)
    ref_usage_count = {str(ref.image_path): 0 for ref in ref_items}
    results: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    processed_unique_gen: set[str] = set()
    processed_gen = 0
    attempts = 0

    def report_progress() -> None:
        if progress_callback is None:
            return
        progress_callback({
            "accepted": len(results),
            "target": config.target_count,
            "processed_gen": processed_gen,
            "attempts": attempts,
        })

    while len(results) < config.target_count:
        gen_pass = shuffled_gen_pass(gen_items, rng)
        accepted_in_pass = 0

        for gen_item in gen_pass:
            if len(results) >= config.target_count:
                break
            gen_key = str(gen_item.image_path)
            if not config.allow_gen_reuse and gen_key in processed_unique_gen:
                continue
            processed_unique_gen.add(gen_key)

            attempted_ref_keys: set[str] = set()
            accepted_for_gen = False
            for attempt_index in range(1, config.max_ref_attempts_per_gen + 1):
                ref_item = choose_balanced_ref(ref_items, ref_usage_count, attempted_ref_keys, rng)
                if ref_item is None:
                    audit.append({
                        "event": "no_ref_candidates_left",
                        "batch_id": config.batch_id,
                        "seed": config.seed,
                        "source_gen_path": str(gen_item.image_path),
                    })
                    break
                attempted_ref_keys.add(str(ref_item.image_path))

                try:
                    original_ref = load_image_rgb(ref_item.image_path)
                    cropped_ref = center_crop_to_size(original_ref, (gen_item.width, gen_item.height))
                    decision = judge_pair(gen_item, ref_item, cropped_ref)
                    accepted = should_accept_decision(decision, config.score_threshold)
                except Exception as exc:
                    audit.append({
                        "event": "pair_error",
                        "batch_id": config.batch_id,
                        "seed": config.seed,
                        "source_gen_path": str(gen_item.image_path),
                        "source_ref_path": str(ref_item.image_path),
                        "attempt_index_for_gen": attempt_index,
                        "error": repr(exc),
                    })
                    continue

                audit_row: dict[str, Any] = {
                    "event": "pair_accepted" if accepted else "pair_rejected",
                    "batch_id": config.batch_id,
                    "seed": config.seed,
                    "source_gen_path": str(gen_item.image_path),
                    "source_ref_path": str(ref_item.image_path),
                    "attempt_index_for_gen": attempt_index,
                    "gen_size": [gen_item.width, gen_item.height],
                    "ref_original_size": [ref_item.width, ref_item.height],
                    "ref_cropped_size": [cropped_ref.width, cropped_ref.height],
                    "suitable": decision.get("suitable"),
                    "score": decision.get("score"),
                    "reason": decision.get("reason"),
                    "action": decision.get("action", ""),
                    "object_description": decision.get("object_description", ""),
                    "prompt": decision.get("prompt", ""),
                }

                if accepted:
                    ref_usage_count[str(ref_item.image_path)] += 1
                    output_gen_path, output_ref_path = materialize_pair(
                        gen_item=gen_item,
                        cropped_ref=cropped_ref,
                        output_paths=output_paths,
                        output_index=len(results),
                    )
                    result = {
                        "cond_1": str(output_gen_path),
                        "cond_2": str(output_ref_path),
                        "prompt": normalized_interaction_prompt_from_decision(decision),
                    }
                    results.append(result)
                    audit_row["output_gen_path"] = str(output_gen_path)
                    audit_row["output_ref_path"] = str(output_ref_path)
                    accepted_for_gen = True
                    accepted_in_pass += 1
                    audit.append(audit_row)
                    break

                audit.append(audit_row)

            if not accepted_for_gen:
                audit.append({
                    "event": "gen_skipped_after_attempts",
                    "batch_id": config.batch_id,
                    "seed": config.seed,
                    "source_gen_path": str(gen_item.image_path),
                    "attempted_count": len(attempted_ref_keys),
                })

            processed_gen += 1
            attempts += len(attempted_ref_keys)
            report_progress()

        if len(results) >= config.target_count:
            break
        if not config.allow_gen_reuse:
            break
        if accepted_in_pass == 0:
            break

    return results, audit


def _judge_pair_with_crop(
    gen_item: ImageItem,
    ref_item: ImageItem,
    judge_pair: JudgePair,
) -> tuple[Image.Image, dict[str, Any]]:
    original_ref = load_image_rgb(ref_item.image_path)
    cropped_ref = center_crop_to_size(original_ref, (gen_item.width, gen_item.height))
    decision = judge_pair(gen_item, ref_item, cropped_ref)
    return cropped_ref, decision


def _run_pairing_concurrent(
    gen_items: list[ImageItem],
    ref_items: list[ImageItem],
    output_paths: ResolvedOutputPaths,
    config: PairingConfig,
    judge_pair: JudgePair,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rng = random.Random(config.seed)
    ref_usage_count = {str(ref.image_path): 0 for ref in ref_items}
    results: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    processed_unique_gen: set[str] = set()
    processed_gen = 0
    attempts = 0

    def report_progress() -> None:
        if progress_callback is None:
            return
        progress_callback({
            "accepted": len(results),
            "target": config.target_count,
            "processed_gen": processed_gen,
            "attempts": attempts,
        })

    def finish_gen(gen_item: ImageItem, attempted_ref_keys: set[str], accepted_for_gen: bool) -> None:
        nonlocal processed_gen, attempts
        if not accepted_for_gen:
            audit.append({
                "event": "gen_skipped_after_attempts",
                "batch_id": config.batch_id,
                "seed": config.seed,
                "source_gen_path": str(gen_item.image_path),
                "attempted_count": len(attempted_ref_keys),
            })
        processed_gen += 1
        attempts += len(attempted_ref_keys)
        report_progress()

    def submit_attempt(
        executor: ThreadPoolExecutor,
        active: dict[Future[tuple[Image.Image, dict[str, Any]]], _PairAttempt],
        gen_item: ImageItem,
        attempted_ref_keys: set[str],
        attempt_index_for_gen: int,
    ) -> bool:
        ref_item = choose_balanced_ref(ref_items, ref_usage_count, attempted_ref_keys, rng)
        if ref_item is None:
            audit.append({
                "event": "no_ref_candidates_left",
                "batch_id": config.batch_id,
                "seed": config.seed,
                "source_gen_path": str(gen_item.image_path),
            })
            finish_gen(gen_item, attempted_ref_keys, accepted_for_gen=False)
            return False

        attempted_ref_keys.add(str(ref_item.image_path))
        future = executor.submit(_judge_pair_with_crop, gen_item, ref_item, judge_pair)
        active[future] = _PairAttempt(
            gen_item=gen_item,
            ref_item=ref_item,
            attempt_index_for_gen=attempt_index_for_gen,
            attempted_ref_keys=attempted_ref_keys,
        )
        return True

    def start_more_gens(
        executor: ThreadPoolExecutor,
        active: dict[Future[tuple[Image.Image, dict[str, Any]]], _PairAttempt],
        gen_pass: list[ImageItem],
        next_gen_index: int,
    ) -> int:
        while (
            len(active) < config.workers
            and next_gen_index < len(gen_pass)
            and len(results) < config.target_count
        ):
            gen_item = gen_pass[next_gen_index]
            next_gen_index += 1
            gen_key = str(gen_item.image_path)
            if not config.allow_gen_reuse and gen_key in processed_unique_gen:
                continue
            processed_unique_gen.add(gen_key)
            submit_attempt(
                executor=executor,
                active=active,
                gen_item=gen_item,
                attempted_ref_keys=set(),
                attempt_index_for_gen=1,
            )
        return next_gen_index

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        while len(results) < config.target_count:
            gen_pass = shuffled_gen_pass(gen_items, rng)
            accepted_in_pass = 0
            next_gen_index = 0
            active: dict[Future[tuple[Image.Image, dict[str, Any]]], _PairAttempt] = {}
            next_gen_index = start_more_gens(executor, active, gen_pass, next_gen_index)

            while active and len(results) < config.target_count:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    attempt = active.pop(future)
                    gen_item = attempt.gen_item
                    ref_item = attempt.ref_item

                    try:
                        cropped_ref, decision = future.result()
                        accepted = should_accept_decision(decision, config.score_threshold)
                    except Exception as exc:
                        audit.append({
                            "event": "pair_error",
                            "batch_id": config.batch_id,
                            "seed": config.seed,
                            "source_gen_path": str(gen_item.image_path),
                            "source_ref_path": str(ref_item.image_path),
                            "attempt_index_for_gen": attempt.attempt_index_for_gen,
                            "error": repr(exc),
                        })
                        if attempt.attempt_index_for_gen < config.max_ref_attempts_per_gen:
                            submit_attempt(
                                executor=executor,
                                active=active,
                                gen_item=gen_item,
                                attempted_ref_keys=attempt.attempted_ref_keys,
                                attempt_index_for_gen=attempt.attempt_index_for_gen + 1,
                            )
                        else:
                            finish_gen(gen_item, attempt.attempted_ref_keys, accepted_for_gen=False)
                        continue

                    audit_row: dict[str, Any] = {
                        "event": "pair_accepted" if accepted else "pair_rejected",
                        "batch_id": config.batch_id,
                        "seed": config.seed,
                        "source_gen_path": str(gen_item.image_path),
                        "source_ref_path": str(ref_item.image_path),
                        "attempt_index_for_gen": attempt.attempt_index_for_gen,
                        "gen_size": [gen_item.width, gen_item.height],
                        "ref_original_size": [ref_item.width, ref_item.height],
                        "ref_cropped_size": [cropped_ref.width, cropped_ref.height],
                        "suitable": decision.get("suitable"),
                        "score": decision.get("score"),
                        "reason": decision.get("reason"),
                        "action": decision.get("action", ""),
                        "object_description": decision.get("object_description", ""),
                        "prompt": decision.get("prompt", ""),
                    }

                    if accepted:
                        ref_usage_count[str(ref_item.image_path)] += 1
                        output_gen_path, output_ref_path = materialize_pair(
                            gen_item=gen_item,
                            cropped_ref=cropped_ref,
                            output_paths=output_paths,
                            output_index=len(results),
                        )
                        results.append({
                            "cond_1": str(output_gen_path),
                            "cond_2": str(output_ref_path),
                            "prompt": normalized_interaction_prompt_from_decision(decision),
                        })
                        audit_row["output_gen_path"] = str(output_gen_path)
                        audit_row["output_ref_path"] = str(output_ref_path)
                        accepted_in_pass += 1
                        audit.append(audit_row)
                        finish_gen(gen_item, attempt.attempted_ref_keys, accepted_for_gen=True)
                    elif attempt.attempt_index_for_gen < config.max_ref_attempts_per_gen:
                        audit.append(audit_row)
                        submit_attempt(
                            executor=executor,
                            active=active,
                            gen_item=gen_item,
                            attempted_ref_keys=attempt.attempted_ref_keys,
                            attempt_index_for_gen=attempt.attempt_index_for_gen + 1,
                        )
                    else:
                        audit.append(audit_row)
                        finish_gen(gen_item, attempt.attempted_ref_keys, accepted_for_gen=False)

                next_gen_index = start_more_gens(executor, active, gen_pass, next_gen_index)

            for future in active:
                future.cancel()

            if len(results) >= config.target_count:
                break
            if not config.allow_gen_reuse:
                break
            if accepted_in_pass == 0:
                break

    return results[:config.target_count], audit


def run_pairing(
    gen_items: list[ImageItem],
    ref_items: list[ImageItem],
    output_paths: ResolvedOutputPaths,
    config: PairingConfig,
    judge_pair: JudgePair,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if config.workers <= 1:
        return _run_pairing_serial(
            gen_items=gen_items,
            ref_items=ref_items,
            output_paths=output_paths,
            config=config,
            judge_pair=judge_pair,
            progress_callback=progress_callback,
        )
    return _run_pairing_concurrent(
        gen_items=gen_items,
        ref_items=ref_items,
        output_paths=output_paths,
        config=config,
        judge_pair=judge_pair,
        progress_callback=progress_callback,
    )


def write_outputs(
    output_json_path: Path,
    audit_jsonl_path: Path,
    results: list[dict[str, str]],
    audit: list[dict[str, Any]],
) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    audit_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    with audit_jsonl_path.open("w", encoding="utf-8") as f:
        for row in audit:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pil_to_data_url(img: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_user_text() -> str:
    return (
        "Judge whether image 1 and image 2 form a suitable hand-object interaction pair. "
        "Image 2 has already been center-cropped to image 1 size. Return strict JSON only."
    )


def infer_pair_decision(
    client: OpenAI,
    model_name: str,
    gen_item: ImageItem,
    ref_item: ImageItem,
    cropped_ref: Image.Image,
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    del ref_item
    img1_url = pil_to_data_url(load_image_rgb(gen_item.image_path, max_pixels=MAX_PIXELS))
    img2_url = pil_to_data_url(cropped_ref, image_format="PNG")
    messages = [
        {"role": "system", "content": PAIRING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_text()},
                {"type": "image_url", "image_url": {"url": img1_url}},
                {"type": "image_url", "image_url": {"url": img2_url}},
            ],
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return parse_vlm_decision(content)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(float(attempt))

    raise RuntimeError(f"VLM inference failed after {max_retries} retries: {last_error!r}")


def make_mock_accept_decision(
    gen_item: ImageItem,
    ref_item: ImageItem,
    cropped_ref: Image.Image | None = None,
) -> dict[str, Any]:
    del gen_item, cropped_ref
    object_description = ref_item.stem.replace("_", " ").replace("-", " ").strip() or "a holdable object"
    prompt = build_interaction_prompt("hold", object_description)
    return {
        "suitable": True,
        "score": 1.0,
        "reason": "dry-run accept-all mode",
        "action": "hold",
        "object_description": object_description,
        "prompt": prompt,
    }


def render_progress_bar(
    stage: str,
    accepted: int,
    target: int,
    processed_gen: int,
    attempts: int,
    width: int = 30,
) -> str:
    safe_target = max(1, target)
    ratio = min(1.0, max(0.0, accepted / safe_target))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100
    return (
        f"[{stage}] [{bar}] {percent:5.1f}% "
        f"accepted={accepted}/{target}, processed_gen={processed_gen}, attempts={attempts}"
    )


class ConsoleProgress:
    def __init__(self, stream: Any | None = None, bar_width: int = 30) -> None:
        self.stream = stream or sys.stderr
        self.bar_width = bar_width

    def stage(self, name: str) -> None:
        self.stream.write(f"[{name}] start\n")
        self.stream.flush()

    def pairing(self, snapshot: dict[str, int]) -> None:
        self.stream.write(render_progress_bar(
            stage="pairing",
            accepted=snapshot["accepted"],
            target=snapshot["target"],
            processed_gen=snapshot["processed_gen"],
            attempts=snapshot["attempts"],
            width=self.bar_width,
        ) + "\n")
        self.stream.flush()


def format_summary(
    batch_id: str,
    seed: int,
    target_count: int,
    accepted_count: int,
    output_json_path: Path,
    audit_jsonl_path: Path,
) -> str:
    return "\n".join([
        "Summary:",
        f"batch_id={batch_id}",
        f"seed={seed}",
        f"target={target_count}",
        f"accepted={accepted_count}",
        f"output={output_json_path}",
        f"audit={audit_jsonl_path}",
    ])


def make_batch_id(batch_id: str | None) -> str:
    if batch_id:
        return batch_id
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return int(datetime.now().strftime("%Y%m%d%H%M%S"))


def _configure_no_proxy(base_url: str | None) -> None:
    no_proxy_hosts = ["localhost", "127.0.0.1"]
    if base_url:
        try:
            host = base_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
            if host:
                no_proxy_hosts.append(host)
        except Exception:
            pass
    value = ",".join(dict.fromkeys(no_proxy_hosts))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value
    for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(proxy_key, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLM-judged no-metadata human-item pairings.")
    parser.add_argument("--gen-dir", type=Path, required=True)
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--output-gen-dir", type=Path, default=None)
    parser.add_argument("--output-ref-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--audit-jsonl", type=Path, default=None)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-ref-attempts-per-gen", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--allow-gen-reuse", action="store_true")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dry-run-accept-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    batch_id = make_batch_id(args.batch_id)
    seed = make_seed(args.seed)
    _configure_no_proxy(args.base_url)
    progress = ConsoleProgress()

    output_paths = build_output_paths(
        output_root=args.output_root,
        batch_id=batch_id,
        output_gen_dir=args.output_gen_dir,
        output_ref_dir=args.output_ref_dir,
        output_json=args.output_json,
        audit_jsonl=args.audit_jsonl,
    )
    config = PairingConfig(
        target_count=args.target_count,
        batch_id=batch_id,
        seed=seed,
        max_ref_attempts_per_gen=args.max_ref_attempts_per_gen,
        score_threshold=args.score_threshold,
        workers=args.workers,
        allow_gen_reuse=args.allow_gen_reuse,
    )

    progress.stage("scan")
    gen_items, gen_audit = scan_image_items(args.gen_dir, "gen")
    ref_items, ref_audit = scan_image_items(args.ref_dir, "ref")
    if not gen_items:
        raise ValueError(f"No valid gen images found in {args.gen_dir}")
    if not ref_items:
        raise ValueError(f"No valid ref images found in {args.ref_dir}")

    progress.stage("pairing")
    if args.dry_run_accept_all:
        judge_pair = make_mock_accept_decision
    else:
        if not args.base_url or not args.api_key or not args.model_name:
            raise ValueError("--base-url, --api-key and --model-name are required outside dry-run mode")
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

        def judge_pair(gen_item: ImageItem, ref_item: ImageItem, cropped_ref: Image.Image) -> dict[str, Any]:
            return infer_pair_decision(
                client=client,
                model_name=args.model_name or "",
                gen_item=gen_item,
                ref_item=ref_item,
                cropped_ref=cropped_ref,
                max_retries=args.max_retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

    results, pairing_audit = run_pairing(
        gen_items=gen_items,
        ref_items=ref_items,
        output_paths=output_paths,
        config=config,
        judge_pair=judge_pair,
        progress_callback=progress.pairing,
    )
    audit = gen_audit + ref_audit + pairing_audit

    progress.stage("write")
    write_outputs(output_paths.output_json, output_paths.audit_jsonl, results, audit)

    progress.stage("done")
    progress.stream.write(format_summary(
        batch_id=batch_id,
        seed=seed,
        target_count=args.target_count,
        accepted_count=len(results),
        output_json_path=output_paths.output_json,
        audit_jsonl_path=output_paths.audit_jsonl,
    ) + "\n")
    progress.stream.flush()


if __name__ == "__main__":
    main()
