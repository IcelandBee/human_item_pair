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
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_PIXELS = 768 * 768

ALLOWED_GEN_SHOT_TYPES = {"half_body", "full_body", "near_full_body"}
ALLOWED_REF_DISPLAY_MODES = {"flat", "hanging"}
ALLOWED_REF_BACKGROUNDS = {"white", "light_solid", "transparent"}


VIRTUAL_CLOTHING_SYSTEM_PROMPT = """
You are a strict data quality judge and prompt engineer for virtual clothing image editing.

You will receive:
- Image 1: a person image.
- Image 2: a reference garment image.
- Metadata for image 1.
- Metadata for image 2.

Decide whether image 2 is suitable as a garment reference to edit the person in image 1.

Judge these five points carefully:
1. Whether the corresponding body region is visible in image 1. A top needs the upper body visible, a bottom needs the lower body visible, and a one-piece garment needs enough torso and leg/body region visible.
2. Whether the reference garment can replace visible clothing on the person in image 1. Do not accept if the target garment would replace a region that is hidden, cropped out, or not clearly clothed.
3. Whether the top, bottom, or one-piece garment type matches the person's framing and visible body area.
4. Treat target user group, gender, and style as a soft compatibility signal, not an absolute rejection rule. Reject only when the mismatch would clearly look unnatural or impossible.
5. Whether the edit can preserve the original person pose without major body pose changes, severe occlusion, unrealistic layering, or impossible garment fit.

Accept reasonably simple replacements, such as replacing a visible shirt with another shirt, pants with pants, a dress with a dress, or an outerwear/top garment when the relevant body region is visible.

Hard reject these cases:
- The required body region is not visible or is too cropped.
- The garment is incomplete, unclear, or not a single usable garment.
- The edit would require major body pose changes or changing the person's body shape.
- The garment type is incompatible with the visible framing, such as applying pants to a tight upper-body crop.
- The prompt would need to add a new person, new pose, or a target image.

If suitable, generate exactly one prompt in this format:
Replace the [clothes description in image 1 to be replaced] worn by the person in image 1 with the [clothes description in image 2] in image 2, while making minimal changes and preserving the original pose of the person.

Return strict JSON only:
{
  "suitable": true or false,
  "score": a number from 0 to 1,
  "reason": "short reason",
  "source_clothes": "short description of clothes in image 1 to replace, or empty string if unsuitable",
  "reference_clothes": "short description of garment in image 2, or empty string if unsuitable",
  "prompt": "final prompt, or empty string if unsuitable"
}
""".strip()


@dataclass(frozen=True)
class ImageItem:
    stem: str
    file_name: str
    image_path: Path
    metadata_path: Path
    size_key: str
    metadata: dict[str, Any]
    dimension_source: str


@dataclass(frozen=True)
class PairingConfig:
    target_count: int
    batch_id: str
    seed: int
    max_ref_attempts_per_gen: int
    score_threshold: float
    workers: int
    allow_gen_reuse: bool


JudgePair = Callable[[ImageItem, ImageItem], dict[str, Any]]
ProgressCallback = Callable[[dict[str, int]], None]


def _annotation(metadata: dict[str, Any]) -> dict[str, Any]:
    annotation = metadata.get("original_annotation", {}).get("annotation", {})
    return annotation if isinstance(annotation, dict) else {}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def resolve_size_key(metadata: dict[str, Any]) -> tuple[str, str]:
    resized_width = _positive_int(metadata.get("resized_width"))
    resized_height = _positive_int(metadata.get("resized_height"))
    if resized_width and resized_height:
        return f"{resized_width}x{resized_height}", "resized_width_height"

    preferred = metadata.get("preferred_resolution")
    if (
        isinstance(preferred, list)
        and len(preferred) == 2
        and _positive_int(preferred[0])
        and _positive_int(preferred[1])
    ):
        return f"{preferred[0]}x{preferred[1]}", "preferred_resolution"

    original_width = _positive_int(metadata.get("original_width"))
    original_height = _positive_int(metadata.get("original_height"))
    if original_width and original_height:
        return f"{original_width}x{original_height}", "original_width_height"

    return "", "missing"


def is_valid_gen_metadata(metadata: dict[str, Any]) -> bool:
    size_key, _ = resolve_size_key(metadata)
    if not size_key:
        return False

    ann = _annotation(metadata)
    if ann.get("person_count") is not None and ann.get("person_count") != "1":
        return False
    if ann.get("head_visible") is not None and ann.get("head_visible") != "yes":
        return False
    if ann.get("clothes_visible") != "yes":
        return False
    if ann.get("shot_type") is not None and ann.get("shot_type") not in ALLOWED_GEN_SHOT_TYPES:
        return False

    return True


def is_valid_ref_metadata(metadata: dict[str, Any]) -> bool:
    size_key, _ = resolve_size_key(metadata)
    if not size_key:
        return False

    ann = _annotation(metadata)
    if ann.get("is_valid_garment_image") != "yes":
        return False
    if ann.get("garment_count") != "single":
        return False
    if ann.get("display_mode") not in ALLOWED_REF_DISPLAY_MODES:
        return False
    if ann.get("image_quality") != "high":
        return False
    if ann.get("garment_completeness") != "complete":
        return False
    if ann.get("background_type") not in ALLOWED_REF_BACKGROUNDS:
        return False

    return True


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def _metadata_is_valid(kind: str, metadata: dict[str, Any]) -> bool:
    if kind == "gen":
        return is_valid_gen_metadata(metadata)
    if kind == "ref":
        return is_valid_ref_metadata(metadata)
    raise ValueError(f"Unsupported kind: {kind}")


def _resolve_image_path(image_dir: Path, metadata: dict[str, Any], stem: str) -> tuple[str, Path]:
    file_name = metadata.get("file_name") or f"{stem}.jpg"
    image_path = image_dir / file_name
    if image_path.exists():
        return file_name, image_path

    metadata_image_path = metadata.get("image_path")
    if isinstance(metadata_image_path, str):
        candidate = Path(metadata_image_path)
        if candidate.exists():
            return candidate.name, candidate

    return file_name, image_path


def build_valid_items(
    image_dir: Path,
    metadata_dir: Path,
    kind: str,
) -> tuple[list[ImageItem], list[dict[str, Any]]]:
    items: list[ImageItem] = []
    audit: list[dict[str, Any]] = []

    for metadata_path in sorted(metadata_dir.glob("*.json")):
        stem = metadata_path.stem
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            audit.append({
                "event": "metadata_read_failed",
                "kind": kind,
                "metadata_path": str(metadata_path),
                "error": repr(exc),
            })
            continue

        file_name, image_path = _resolve_image_path(image_dir, metadata, stem)
        if not image_path.exists() or not _is_image_file(image_path):
            audit.append({
                "event": "image_missing",
                "kind": kind,
                "metadata_path": str(metadata_path),
                "image_path": str(image_path),
            })
            continue

        size_key, dimension_source = resolve_size_key(metadata)
        if not size_key:
            audit.append({
                "event": "invalid_dimensions",
                "kind": kind,
                "metadata_path": str(metadata_path),
            })
            continue

        if not _metadata_is_valid(kind, metadata):
            audit.append({
                "event": "metadata_prefilter_rejected",
                "kind": kind,
                "metadata_path": str(metadata_path),
                "size_key": size_key,
            })
            continue

        items.append(ImageItem(
            stem=stem,
            file_name=file_name,
            image_path=image_path,
            metadata_path=metadata_path,
            size_key=size_key,
            metadata=metadata,
            dimension_source=dimension_source,
        ))

    return items, audit


def build_size_buckets(
    gen_items: list[ImageItem],
    ref_items: list[ImageItem],
) -> dict[str, dict[str, list[ImageItem]]]:
    gen_by_size: dict[str, list[ImageItem]] = defaultdict(list)
    ref_by_size: dict[str, list[ImageItem]] = defaultdict(list)

    for item in gen_items:
        gen_by_size[item.size_key].append(item)
    for item in ref_items:
        ref_by_size[item.size_key].append(item)

    common_size_keys = sorted(set(gen_by_size) & set(ref_by_size))
    return {
        size_key: {
            "gen": gen_by_size[size_key],
            "ref": ref_by_size[size_key],
        }
        for size_key in common_size_keys
    }


def shuffled_gen_pass(gen_items: list[ImageItem], rng: random.Random) -> list[ImageItem]:
    result = list(gen_items)
    rng.shuffle(result)
    return result


def choose_balanced_ref(
    refs: list[ImageItem],
    ref_usage_count: dict[str, int],
    attempted_ref_stems: set[str],
    rng: random.Random,
) -> ImageItem | None:
    candidates = [ref for ref in refs if ref.stem not in attempted_ref_stems]
    if not candidates:
        return None

    min_usage = min(ref_usage_count.get(ref.stem, 0) for ref in candidates)
    least_used = [
        ref for ref in candidates
        if ref_usage_count.get(ref.stem, 0) == min_usage
    ]
    return rng.choice(least_used)


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
    prompt = decision.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    prompt = prompt.strip()
    if not prompt.startswith("Replace the "):
        return False
    required_parts = [
        " worn by the person in image 1 ",
        " with the ",
        " in image 2",
        "preserving the original pose of the person",
    ]
    return all(part in prompt for part in required_parts)


def compact_gen_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    ann = _annotation(metadata)
    return {
        "file_name": metadata.get("file_name"),
        "resized_width": metadata.get("resized_width"),
        "resized_height": metadata.get("resized_height"),
        "gender": ann.get("gender"),
        "shot_type": ann.get("shot_type"),
        "clothes_visible": ann.get("clothes_visible"),
        "person_size_in_frame": ann.get("person_size_in_frame"),
        "person_prominence": ann.get("person_prominence"),
        "face_direction": ann.get("face_direction"),
        "holding_object": ann.get("holding_object"),
    }


def compact_ref_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    ann = _annotation(metadata)
    return {
        "file_name": metadata.get("file_name"),
        "resized_width": metadata.get("resized_width"),
        "resized_height": metadata.get("resized_height"),
        "target_user_group": ann.get("target_user_group"),
        "display_mode": ann.get("display_mode"),
        "background_type": ann.get("background_type"),
        "pattern_type": ann.get("pattern_type"),
        "garment_position_type": ann.get("garment_position_type"),
        "garment_category": ann.get("garment_category"),
    }


def load_image_rgb(path: Path, max_pixels: int = MAX_PIXELS) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_pixels > 0:
        width, height = img.size
        if width * height > max_pixels:
            scale = math.sqrt(max_pixels / float(width * height))
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


def pil_to_data_url(img: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_user_text(gen_item: ImageItem, ref_item: ImageItem) -> str:
    payload = {
        "image_1_gen_metadata": compact_gen_metadata(gen_item.metadata),
        "image_2_ref_metadata": compact_ref_metadata(ref_item.metadata),
    }
    return (
        "Judge whether image 1 and image 2 form a suitable virtual clothing editing pair. "
        "Use both images and metadata. Return strict JSON only.\n\n"
        f"Metadata:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def infer_pair_decision(
    client: OpenAI,
    model_name: str,
    gen_item: ImageItem,
    ref_item: ImageItem,
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    img1_url = pil_to_data_url(load_image_rgb(gen_item.image_path))
    img2_url = pil_to_data_url(load_image_rgb(ref_item.image_path))
    messages = [
        {"role": "system", "content": VIRTUAL_CLOTHING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_text(gen_item, ref_item)},
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


def _all_gen_items_from_buckets(
    buckets: dict[str, dict[str, list[ImageItem]]],
) -> list[ImageItem]:
    items: list[ImageItem] = []
    for size_key in sorted(buckets):
        items.extend(buckets[size_key]["gen"])
    return items


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
        f"accepted={accepted}/{target}, "
        f"processed_gen={processed_gen}, attempts={attempts}"
    )


class ConsoleProgress:
    def __init__(
        self,
        stream: Any | None = None,
        bar_width: int = 30,
        non_tty_percent_step: int = 5,
    ) -> None:
        self.stream = stream or sys.stderr
        self.bar_width = bar_width
        self.non_tty_percent_step = non_tty_percent_step
        self._last_line_length = 0
        self._last_non_tty_percent = -non_tty_percent_step

    def stage(self, name: str) -> None:
        self.stream.write(f"[{name}] start\n")
        self.stream.flush()

    def pairing(self, snapshot: dict[str, int]) -> None:
        line = render_progress_bar(
            stage="pairing",
            accepted=snapshot["accepted"],
            target=snapshot["target"],
            processed_gen=snapshot["processed_gen"],
            attempts=snapshot["attempts"],
            width=self.bar_width,
        )
        if self.stream.isatty():
            padding = " " * max(0, self._last_line_length - len(line))
            self.stream.write("\r" + line + padding)
            self._last_line_length = len(line)
        else:
            target = max(1, snapshot["target"])
            current_percent = int((snapshot["accepted"] / target) * 100)
            should_emit = (
                current_percent >= self._last_non_tty_percent + self.non_tty_percent_step
                or snapshot["accepted"] >= snapshot["target"]
            )
            if not should_emit:
                return
            self._last_non_tty_percent = current_percent
            self.stream.write(line + "\n")
        self.stream.flush()

    def finish_pairing_line(self) -> None:
        if self.stream.isatty() and self._last_line_length:
            self.stream.write("\n")
            self.stream.flush()
            self._last_line_length = 0


def run_pairing(
    buckets: dict[str, dict[str, list[ImageItem]]],
    config: PairingConfig,
    judge_pair: JudgePair,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rng = random.Random(config.seed)
    ref_usage_by_size: dict[str, dict[str, int]] = {
        size_key: {ref.stem: 0 for ref in bucket["ref"]}
        for size_key, bucket in buckets.items()
    }
    results: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    gen_pool = _all_gen_items_from_buckets(buckets)
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
        gen_pass = shuffled_gen_pass(gen_pool, rng)
        accepted_in_pass = 0

        for gen_item in gen_pass:
            if len(results) >= config.target_count:
                break
            if not config.allow_gen_reuse and gen_item.stem in processed_unique_gen:
                continue
            processed_unique_gen.add(gen_item.stem)

            refs = buckets[gen_item.size_key]["ref"]
            attempted_ref_stems: set[str] = set()
            accepted_for_gen = False

            for attempt_index in range(1, config.max_ref_attempts_per_gen + 1):
                ref_item = choose_balanced_ref(
                    refs=refs,
                    ref_usage_count=ref_usage_by_size[gen_item.size_key],
                    attempted_ref_stems=attempted_ref_stems,
                    rng=rng,
                )
                if ref_item is None:
                    audit.append({
                        "event": "no_ref_candidates_left",
                        "batch_id": config.batch_id,
                        "seed": config.seed,
                        "size_key": gen_item.size_key,
                        "gen_path": str(gen_item.image_path),
                    })
                    break

                attempted_ref_stems.add(ref_item.stem)
                try:
                    decision = judge_pair(gen_item, ref_item)
                    accepted = should_accept_decision(decision, config.score_threshold)
                except Exception as exc:
                    audit.append({
                        "event": "pair_error",
                        "batch_id": config.batch_id,
                        "seed": config.seed,
                        "size_key": gen_item.size_key,
                        "gen_path": str(gen_item.image_path),
                        "ref_path": str(ref_item.image_path),
                        "attempt_index_for_gen": attempt_index,
                        "error": repr(exc),
                    })
                    continue

                audit.append({
                    "event": "pair_accepted" if accepted else "pair_rejected",
                    "batch_id": config.batch_id,
                    "seed": config.seed,
                    "size_key": gen_item.size_key,
                    "gen_path": str(gen_item.image_path),
                    "ref_path": str(ref_item.image_path),
                    "attempt_index_for_gen": attempt_index,
                    "suitable": decision.get("suitable"),
                    "score": decision.get("score"),
                    "reason": decision.get("reason"),
                    "source_clothes": decision.get("source_clothes", ""),
                    "reference_clothes": decision.get("reference_clothes", ""),
                    "prompt": decision.get("prompt", ""),
                })

                if accepted:
                    ref_usage_by_size[gen_item.size_key][ref_item.stem] += 1
                    results.append({
                        "cond_1": str(gen_item.image_path),
                        "cond_2": str(ref_item.image_path),
                        "prompt": str(decision["prompt"]).strip(),
                    })
                    accepted_for_gen = True
                    accepted_in_pass += 1
                    break

            if not accepted_for_gen:
                audit.append({
                    "event": "gen_skipped_after_attempts",
                    "batch_id": config.batch_id,
                    "seed": config.seed,
                    "size_key": gen_item.size_key,
                    "gen_path": str(gen_item.image_path),
                    "attempted_count": len(attempted_ref_stems),
                })

            processed_gen += 1
            attempts += len(attempted_ref_stems)
            report_progress()

        if len(results) >= config.target_count:
            break
        if not config.allow_gen_reuse:
            break
        if accepted_in_pass == 0:
            break

    return results, audit


def build_output_paths(output_dir: Path, batch_id: str) -> tuple[Path, Path]:
    return (
        output_dir / f"virtual-clothing_{batch_id}.json",
        output_dir / f"virtual-clothing_{batch_id}.audit.jsonl",
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


def _garment_description_from_metadata(metadata: dict[str, Any]) -> str:
    ann = _annotation(metadata)
    pattern = ann.get("pattern_type")
    category = ann.get("garment_category")
    position = ann.get("garment_position_type")
    parts = [str(x).strip() for x in (pattern, category) if x and str(x).strip() != "other"]
    if parts:
        return " ".join(parts)
    if category and str(category).strip():
        return str(category).strip()
    if position and str(position).strip():
        return str(position).strip() + " garment"
    return "reference garment"


def make_mock_accept_decision(gen_item: ImageItem, ref_item: ImageItem) -> dict[str, Any]:
    reference_clothes = _garment_description_from_metadata(ref_item.metadata)
    source_clothes = "visible clothing"
    prompt = (
        f"Replace the {source_clothes} worn by the person in image 1 with the "
        f"{reference_clothes} in image 2, while making minimal changes and "
        "preserving the original pose of the person."
    )
    return {
        "suitable": True,
        "score": 1.0,
        "reason": "dry-run accept-all mode",
        "source_clothes": source_clothes,
        "reference_clothes": reference_clothes,
        "prompt": prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLM-judged virtual clothing pairings.")
    parser.add_argument("--gen-dir", type=Path, required=True)
    parser.add_argument("--gen-metadata-dir", type=Path, required=True)
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--ref-metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-ref-attempts-per-gen", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-gen-reuse", action="store_true")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dry-run-accept-all", action="store_true")
    return parser.parse_args()


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


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    batch_id = make_batch_id(args.batch_id)
    seed = make_seed(args.seed)
    _configure_no_proxy(args.base_url)
    progress = ConsoleProgress()

    config = PairingConfig(
        target_count=args.target_count,
        batch_id=batch_id,
        seed=seed,
        max_ref_attempts_per_gen=args.max_ref_attempts_per_gen,
        score_threshold=args.score_threshold,
        workers=args.workers,
        allow_gen_reuse=args.allow_gen_reuse,
    )

    progress.stage("prepare-data")
    gen_items, gen_audit = build_valid_items(args.gen_dir, args.gen_metadata_dir, "gen")
    ref_items, ref_audit = build_valid_items(args.ref_dir, args.ref_metadata_dir, "ref")

    progress.stage("build-size-buckets")
    buckets = build_size_buckets(gen_items, ref_items)

    progress.stage("pairing")
    if args.dry_run_accept_all:
        judge_pair = make_mock_accept_decision
    else:
        if not args.base_url or not args.api_key or not args.model_name:
            raise ValueError("--base-url, --api-key and --model-name are required outside dry-run mode")
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

        def judge_pair(gen_item: ImageItem, ref_item: ImageItem) -> dict[str, Any]:
            return infer_pair_decision(
                client=client,
                model_name=args.model_name or "",
                gen_item=gen_item,
                ref_item=ref_item,
                max_retries=args.max_retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

    results, pairing_audit = run_pairing(
        buckets=buckets,
        config=config,
        judge_pair=judge_pair,
        progress_callback=progress.pairing,
    )
    progress.finish_pairing_line()
    audit = gen_audit + ref_audit + pairing_audit

    progress.stage("write-output")
    output_json_path, audit_jsonl_path = build_output_paths(args.output_dir, batch_id)
    write_outputs(output_json_path, audit_jsonl_path, results, audit)

    progress.stage("done")
    progress.stream.write(format_summary(
        batch_id=batch_id,
        seed=seed,
        target_count=args.target_count,
        accepted_count=len(results),
        output_json_path=output_json_path,
        audit_jsonl_path=audit_jsonl_path,
    ) + "\n")
    progress.stream.flush()


if __name__ == "__main__":
    main()
