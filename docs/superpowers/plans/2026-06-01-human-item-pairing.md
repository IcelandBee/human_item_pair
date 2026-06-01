# Human-Item Pairing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可批量运行的人物-物品配对脚本：先按尺寸和 metadata 规则筛选候选，再随机且均衡地选择同尺寸 `ref`，最后调用 VLM 判断配对是否适合并生成训练 prompt。

**Architecture:** 实现拆成一个主脚本和一组可测试的纯函数。纯函数负责 metadata 读取、尺寸解析、预筛、索引、随机配对和 VLM JSON 解析；主流程负责参数解析、并发调用 VLM、写入主输出 JSON 和 audit JSONL。这样可以先用 sample metadata 和 mock VLM 完成测试，再接入真实图片与 OpenAI-compatible VLM 服务。

**Tech Stack:** Python 3、标准库 `argparse/json/pathlib/random/concurrent.futures/dataclasses`、Pillow、OpenAI Python SDK、pytest。

---

## 文件结构

- Create: `D:/Project/training_pair/human_item_pairing.py`
  - 命令行入口。
  - 读取 gen/ref 图片目录与 metadata 目录。
  - 按尺寸建立候选池。
  - 执行随机配对、VLM 判断、prompt 生成、输出主 JSON 和 audit JSONL。
- Create: `D:/Project/training_pair/tests/test_human_item_pairing.py`
  - 单元测试文件。
  - 覆盖尺寸解析、metadata 预筛、索引分桶、ref 均衡选择、VLM 响应解析、输出命名。
- Keep: `D:/Project/training_pair/output/testing_prompt_gen.py`
  - 作为 VLM 图片编码、OpenAI-compatible 调用和 prompt 风格参考。
  - 不直接修改，避免破坏原始参考脚本。
- Keep: `D:/Project/training_pair/docs/superpowers/specs/2026-06-01-human-item-pairing-design.md`
  - 已确认的设计 spec。
- Create/Use: `D:/Project/training_pair/output/`
  - 默认输出目录。
  - 生成 `human-item_<batch_id>.json` 和 `human-item_<batch_id>.audit.jsonl`。

## Task 1: 建立脚本骨架和数据结构

**Files:**
- Create: `D:/Project/training_pair/human_item_pairing.py`
- Test: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 创建失败测试，验证核心数据结构可以导入**

在 `D:/Project/training_pair/tests/test_human_item_pairing.py` 中加入：

```python
from pathlib import Path

from human_item_pairing import ImageItem, PairingConfig


def test_core_dataclasses_can_be_created():
    item = ImageItem(
        stem="000001",
        file_name="000001.jpg",
        image_path=Path("gen/000001.jpg"),
        metadata_path=Path("gen_metadata/000001.json"),
        size_key="1248x832",
        metadata={"file_name": "000001.jpg"},
        dimension_source="resized_width_height",
    )

    config = PairingConfig(
        target_count=3,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=5,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=False,
    )

    assert item.stem == "000001"
    assert item.size_key == "1248x832"
    assert config.seed == 123
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py::test_core_dataclasses_can_be_created -v
```

Expected:

```text
ModuleNotFoundError: No module named 'human_item_pairing'
```

- [ ] **Step 3: 创建最小脚本骨架**

在 `D:/Project/training_pair/human_item_pairing.py` 中加入：

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py::test_core_dataclasses_can_be_created -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: 提交**

如果当前目录还不是 git 仓库，先准备仓库：

```bash
cd D:/Project/training_pair
git init
git remote add origin https://github.com/IcelandBee/human_item_pair.git
```

然后提交：

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "feat: add pairing script skeleton"
```

## Task 2: 实现尺寸解析和 metadata 预筛

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py`
- Modify: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 写尺寸解析测试**

追加到 `D:/Project/training_pair/tests/test_human_item_pairing.py`：

```python
from human_item_pairing import resolve_size_key


def test_resolve_size_key_prefers_resized_dimensions():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "preferred_resolution": [512, 512],
        "original_width": 640,
        "original_height": 480,
    }

    size_key, source = resolve_size_key(metadata)

    assert size_key == "1248x832"
    assert source == "resized_width_height"


def test_resolve_size_key_falls_back_to_preferred_resolution():
    metadata = {
        "preferred_resolution": [944, 1104],
        "original_width": 640,
        "original_height": 480,
    }

    size_key, source = resolve_size_key(metadata)

    assert size_key == "944x1104"
    assert source == "preferred_resolution"


def test_resolve_size_key_returns_empty_for_invalid_metadata():
    size_key, source = resolve_size_key({"file_name": "bad.jpg"})

    assert size_key == ""
    assert source == "missing"
```

- [ ] **Step 2: 写 metadata 预筛测试**

追加：

```python
from human_item_pairing import is_valid_gen_metadata, is_valid_ref_metadata


def test_valid_gen_metadata_requires_hand_hold_feasible_when_present():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "hand_hold_feasible": "yes",
            }
        },
    }

    assert is_valid_gen_metadata(metadata) is True


def test_invalid_gen_metadata_rejects_hand_hold_no():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "hand_hold_feasible": "no",
            }
        },
    }

    assert is_valid_gen_metadata(metadata) is False


def test_valid_ref_metadata_requires_holdable_flags():
    metadata = {
        "resized_width": 944,
        "resized_height": 1104,
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "high",
        "holdability": "carryable",
    }

    assert is_valid_ref_metadata(metadata) is True


def test_invalid_ref_metadata_rejects_low_confidence():
    metadata = {
        "resized_width": 944,
        "resized_height": 1104,
        "suitable_for_holding": True,
        "should_use": True,
        "confidence": "low",
        "holdability": "carryable",
    }

    assert is_valid_ref_metadata(metadata) is False
```

- [ ] **Step 3: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
FAILED ... cannot import name 'resolve_size_key'
```

- [ ] **Step 4: 实现尺寸解析和预筛函数**

追加到 `D:/Project/training_pair/human_item_pairing.py`：

```python
ALLOWED_HOLDABILITY = {"handle", "carryable", "pet_holdable"}


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

    annotation = (
        metadata.get("original_annotation", {})
        .get("annotation", {})
    )
    hand_hold_feasible = annotation.get("hand_hold_feasible")
    if hand_hold_feasible is not None and hand_hold_feasible != "yes":
        return False

    return True


def is_valid_ref_metadata(metadata: dict[str, Any]) -> bool:
    size_key, _ = resolve_size_key(metadata)
    if not size_key:
        return False

    if metadata.get("suitable_for_holding") is not True:
        return False
    if metadata.get("should_use") is not True:
        return False
    if metadata.get("confidence") is not None and metadata.get("confidence") != "high":
        return False
    if metadata.get("holdability") not in ALLOWED_HOLDABILITY:
        return False

    return True
```

- [ ] **Step 5: 运行测试并确认通过**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
8 passed
```

- [ ] **Step 6: 提交**

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "feat: add metadata validation"
```

## Task 3: 实现 metadata 加载和尺寸分桶索引

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py`
- Modify: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 写加载与分桶测试**

追加：

```python
import json

from human_item_pairing import build_valid_items, build_size_buckets


def test_build_valid_items_loads_matching_metadata_and_images(tmp_path):
    image_dir = tmp_path / "gen"
    metadata_dir = tmp_path / "gen_metadata"
    image_dir.mkdir()
    metadata_dir.mkdir()
    image_path = image_dir / "000001.jpg"
    image_path.write_bytes(b"fake")
    metadata_path = metadata_dir / "000001.json"
    metadata_path.write_text(
        json.dumps(
            {
                "file_name": "000001.jpg",
                "resized_width": 1248,
                "resized_height": 832,
                "original_annotation": {
                    "annotation": {
                        "hand_hold_feasible": "yes",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    items, audit = build_valid_items(
        image_dir=image_dir,
        metadata_dir=metadata_dir,
        kind="gen",
    )

    assert len(items) == 1
    assert items[0].image_path == image_path
    assert items[0].size_key == "1248x832"
    assert audit == []


def test_build_size_buckets_keeps_only_common_size_keys():
    gen = [
        ImageItem("g1", "g1.jpg", Path("g1.jpg"), Path("g1.json"), "100x100", {}, "resized_width_height"),
        ImageItem("g2", "g2.jpg", Path("g2.jpg"), Path("g2.json"), "200x200", {}, "resized_width_height"),
    ]
    ref = [
        ImageItem("r1", "r1.jpg", Path("r1.jpg"), Path("r1.json"), "100x100", {}, "resized_width_height"),
    ]

    buckets = build_size_buckets(gen, ref)

    assert sorted(buckets.keys()) == ["100x100"]
    assert buckets["100x100"]["gen"] == [gen[0]]
    assert buckets["100x100"]["ref"] == [ref[0]]
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
FAILED ... cannot import name 'build_valid_items'
```

- [ ] **Step 3: 实现加载与分桶**

追加到 `D:/Project/training_pair/human_item_pairing.py`：

```python
import json
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def _metadata_is_valid(kind: str, metadata: dict[str, Any]) -> bool:
    if kind == "gen":
        return is_valid_gen_metadata(metadata)
    if kind == "ref":
        return is_valid_ref_metadata(metadata)
    raise ValueError(f"Unsupported kind: {kind}")


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

        file_name = metadata.get("file_name") or f"{stem}.jpg"
        image_path = image_dir / file_name
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
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
10 passed
```

- [ ] **Step 5: 提交**

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "feat: index metadata by image size"
```

## Task 4: 实现随机 gen 顺序和 ref 均衡选择

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py`
- Modify: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 写 ref 最少使用优先测试**

追加：

```python
import random

from human_item_pairing import choose_balanced_ref, shuffled_gen_pass


def test_choose_balanced_ref_prefers_least_used_ref():
    refs = [
        ImageItem("r1", "r1.jpg", Path("r1.jpg"), Path("r1.json"), "100x100", {}, "resized_width_height"),
        ImageItem("r2", "r2.jpg", Path("r2.jpg"), Path("r2.json"), "100x100", {}, "resized_width_height"),
        ImageItem("r3", "r3.jpg", Path("r3.jpg"), Path("r3.json"), "100x100", {}, "resized_width_height"),
    ]
    usage = {"r1": 3, "r2": 1, "r3": 1}
    rng = random.Random(5)

    selected = choose_balanced_ref(
        refs=refs,
        ref_usage_count=usage,
        attempted_ref_stems=set(),
        rng=rng,
    )

    assert selected.stem in {"r2", "r3"}


def test_choose_balanced_ref_excludes_attempted_refs():
    refs = [
        ImageItem("r1", "r1.jpg", Path("r1.jpg"), Path("r1.json"), "100x100", {}, "resized_width_height"),
        ImageItem("r2", "r2.jpg", Path("r2.jpg"), Path("r2.json"), "100x100", {}, "resized_width_height"),
    ]
    usage = {"r1": 0, "r2": 0}
    rng = random.Random(7)

    selected = choose_balanced_ref(
        refs=refs,
        ref_usage_count=usage,
        attempted_ref_stems={"r1"},
        rng=rng,
    )

    assert selected.stem == "r2"


def test_shuffled_gen_pass_is_reproducible_with_seed():
    gen_items = [
        ImageItem("g1", "g1.jpg", Path("g1.jpg"), Path("g1.json"), "100x100", {}, "resized_width_height"),
        ImageItem("g2", "g2.jpg", Path("g2.jpg"), Path("g2.json"), "100x100", {}, "resized_width_height"),
        ImageItem("g3", "g3.jpg", Path("g3.jpg"), Path("g3.json"), "100x100", {}, "resized_width_height"),
    ]

    first = shuffled_gen_pass(gen_items, random.Random(123))
    second = shuffled_gen_pass(gen_items, random.Random(123))

    assert [item.stem for item in first] == [item.stem for item in second]
    assert sorted(item.stem for item in first) == ["g1", "g2", "g3"]
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
FAILED ... cannot import name 'choose_balanced_ref'
```

- [ ] **Step 3: 实现随机和均衡选择函数**

追加到 `D:/Project/training_pair/human_item_pairing.py`：

```python
import random


def shuffled_gen_pass(
    gen_items: list[ImageItem],
    rng: random.Random,
) -> list[ImageItem]:
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
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
13 passed
```

- [ ] **Step 5: 提交**

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "feat: balance reference usage in batches"
```

## Task 5: 实现 VLM 响应解析与验收规则

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py`
- Modify: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 写 VLM JSON 解析测试**

追加：

```python
from human_item_pairing import parse_vlm_decision, should_accept_decision


def test_parse_vlm_decision_accepts_json_inside_markdown_fence():
    raw = '''```json
{
  "suitable": true,
  "score": 0.86,
  "reason": "Natural hand-object interaction.",
  "action": "hold",
  "object_description": "a clear glass jar",
  "prompt": "Let the person in image 1 hold a clear glass jar shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged."
}
```'''

    decision = parse_vlm_decision(raw)

    assert decision["suitable"] is True
    assert decision["score"] == 0.86
    assert decision["action"] == "hold"


def test_should_accept_decision_requires_threshold_and_prompt_format():
    decision = {
        "suitable": True,
        "score": 0.8,
        "reason": "ok",
        "prompt": "Let the person in image 1 hold a clear glass jar shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged.",
    }

    assert should_accept_decision(decision, score_threshold=0.75) is True
    assert should_accept_decision(decision, score_threshold=0.9) is False


def test_should_accept_decision_rejects_non_prompt_text():
    decision = {
        "suitable": True,
        "score": 0.99,
        "reason": "ok",
        "prompt": "The pair looks good.",
    }

    assert should_accept_decision(decision, score_threshold=0.75) is False
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
FAILED ... cannot import name 'parse_vlm_decision'
```

- [ ] **Step 3: 实现解析与验收函数**

追加到 `D:/Project/training_pair/human_item_pairing.py`：

```python
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
    cleaned = _strip_markdown_fence(raw_text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("VLM response must be a JSON object")
    return data


def should_accept_decision(
    decision: dict[str, Any],
    score_threshold: float,
) -> bool:
    if decision.get("suitable") is not True:
        return False
    score = decision.get("score")
    if not isinstance(score, int | float):
        return False
    if float(score) < score_threshold:
        return False
    prompt = decision.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    prompt = prompt.strip()
    if not prompt.startswith("Let the person in image 1 "):
        return False
    if " shown in image 2 " not in prompt:
        return False
    if "keeping everything else in image 1 unchanged" not in prompt:
        return False
    return True
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
16 passed
```

- [ ] **Step 5: 提交**

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "feat: parse vlm pairing decisions"
```

## Task 6: 实现 VLM prompt、图片编码和单对推理

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py`
- Modify: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 写 compact metadata 测试**

追加：

```python
from human_item_pairing import compact_gen_metadata, compact_ref_metadata


def test_compact_gen_metadata_keeps_pose_related_fields():
    metadata = {
        "file_name": "000001.jpg",
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "gender": "female",
                "shot_type": "half_body",
                "face_direction": "frontal",
                "hand_hold_feasible": "yes",
                "person_size_in_frame": "large",
            }
        },
        "large_unused_field": "x" * 1000,
    }

    compact = compact_gen_metadata(metadata)

    assert compact["file_name"] == "000001.jpg"
    assert compact["shot_type"] == "half_body"
    assert compact["hand_hold_feasible"] == "yes"
    assert "large_unused_field" not in compact


def test_compact_ref_metadata_keeps_object_fields():
    metadata = {
        "file_name": "000002.jpg",
        "object_name": "glass jar",
        "object_category": "container",
        "object_description": "A large clear glass jar.",
        "holdability": "carryable",
        "bbox_norm": {"x_min": 0.1},
    }

    compact = compact_ref_metadata(metadata)

    assert compact["object_name"] == "glass jar"
    assert compact["object_category"] == "container"
    assert compact["holdability"] == "carryable"
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
FAILED ... cannot import name 'compact_gen_metadata'
```

- [ ] **Step 3: 实现 compact metadata 和 VLM prompt 常量**

追加到 `D:/Project/training_pair/human_item_pairing.py`：

```python
PAIRING_SYSTEM_PROMPT = """
You are a strict data quality judge and prompt engineer for hand-object interaction image editing.

You will receive:
- Image 1: a person image.
- Image 2: a reference object image.
- Metadata for image 1.
- Metadata for image 2.

Decide whether the person in image 1 can naturally interact with the main object in image 2 using their hand or hands.

Be conservative. Reject the pair if the interaction would require major pose changes, impossible hand placement, severe occlusion, unrealistic object scale, unclear contact, or an unclear hand-object action.

If suitable, generate exactly one prompt in this format:
Let the person in image 1 [hand-object action phrase] [object description] shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged.

Return strict JSON only:
{
  "suitable": true or false,
  "score": a number from 0 to 1,
  "reason": "short reason",
  "action": "short hand-object action phrase, or empty string if unsuitable",
  "object_description": "main object description, or empty string if unsuitable",
  "prompt": "final prompt, or empty string if unsuitable"
}
""".strip()


def compact_gen_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    annotation = (
        metadata.get("original_annotation", {})
        .get("annotation", {})
    )
    return {
        "file_name": metadata.get("file_name"),
        "resized_width": metadata.get("resized_width"),
        "resized_height": metadata.get("resized_height"),
        "shot_type": annotation.get("shot_type"),
        "face_direction": annotation.get("face_direction"),
        "hand_hold_feasible": annotation.get("hand_hold_feasible"),
        "person_size_in_frame": annotation.get("person_size_in_frame"),
        "person_prominence": annotation.get("person_prominence"),
        "holding_object": annotation.get("holding_object"),
        "head_visible": annotation.get("head_visible"),
    }


def compact_ref_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": metadata.get("file_name"),
        "resized_width": metadata.get("resized_width"),
        "resized_height": metadata.get("resized_height"),
        "object_name": metadata.get("object_name"),
        "object_category": metadata.get("object_category"),
        "object_description": metadata.get("object_description"),
        "holdability": metadata.get("holdability"),
        "bbox_norm": metadata.get("bbox_norm"),
        "reason": metadata.get("reason"),
    }
```

- [ ] **Step 4: 实现图片编码和真实 VLM 调用函数**

继续追加：

```python
import base64
import io
import math
import time

from PIL import Image
from openai import OpenAI

MAX_PIXELS = 768 * 768


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
        "Judge whether image 1 and image 2 form a suitable hand-object interaction pair. "
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
        {"role": "system", "content": PAIRING_SYSTEM_PROMPT},
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
```

- [ ] **Step 5: 运行测试并确认通过**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
18 passed
```

- [ ] **Step 6: 提交**

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "feat: add vlm decision prompt"
```

## Task 7: 实现批处理 pairing 主循环，支持 mock VLM 测试

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py`
- Modify: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 写批处理主循环测试**

追加：

```python
from human_item_pairing import run_pairing


def test_run_pairing_balances_refs_and_stops_at_target_count():
    gen_items = [
        ImageItem("g1", "g1.jpg", Path("g1.jpg"), Path("g1.json"), "100x100", {}, "resized_width_height"),
        ImageItem("g2", "g2.jpg", Path("g2.jpg"), Path("g2.json"), "100x100", {}, "resized_width_height"),
        ImageItem("g3", "g3.jpg", Path("g3.jpg"), Path("g3.json"), "100x100", {}, "resized_width_height"),
    ]
    ref_items = [
        ImageItem("r1", "r1.jpg", Path("r1.jpg"), Path("r1.json"), "100x100", {}, "resized_width_height"),
        ImageItem("r2", "r2.jpg", Path("r2.jpg"), Path("r2.json"), "100x100", {}, "resized_width_height"),
    ]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=3,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=2,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=False,
    )

    def fake_judge(gen_item, ref_item):
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "action": "hold",
            "object_description": "a mock object",
            "prompt": f"Let the person in image 1 hold a mock object shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged.",
        }

    results, audit = run_pairing(
        buckets=buckets,
        config=config,
        judge_pair=fake_judge,
    )

    assert len(results) == 3
    assert len([row for row in audit if row["event"] == "pair_accepted"]) == 3
    ref_counts = {}
    for result in results:
        ref_counts[result["cond_2"]] = ref_counts.get(result["cond_2"], 0) + 1
    assert sorted(ref_counts.values()) in ([1, 2], [1, 1, 1])
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
FAILED ... cannot import name 'run_pairing'
```

- [ ] **Step 3: 实现批处理主循环**

追加到 `D:/Project/training_pair/human_item_pairing.py`：

```python
from collections.abc import Callable

JudgePair = Callable[[ImageItem, ImageItem], dict[str, Any]]


def _all_gen_items_from_buckets(
    buckets: dict[str, dict[str, list[ImageItem]]],
) -> list[ImageItem]:
    items: list[ImageItem] = []
    for size_key in sorted(buckets):
        items.extend(buckets[size_key]["gen"])
    return items


def run_pairing(
    buckets: dict[str, dict[str, list[ImageItem]]],
    config: PairingConfig,
    judge_pair: JudgePair,
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

                audit_row = {
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
                    "prompt": decision.get("prompt", ""),
                }
                audit.append(audit_row)

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

        if len(results) >= config.target_count:
            break
        if not config.allow_gen_reuse:
            break
        if accepted_in_pass == 0:
            break

    return results, audit
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
19 passed
```

- [ ] **Step 5: 提交**

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "feat: add batch pairing loop"
```

## Task 8: 实现输出命名、写文件和 CLI 参数

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py`
- Modify: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 写输出命名测试**

追加：

```python
from human_item_pairing import build_output_paths, write_outputs


def test_build_output_paths_uses_batch_id():
    output_json, audit_jsonl = build_output_paths(
        output_dir=Path("output"),
        batch_id="exp_v1",
    )

    assert output_json == Path("output") / "human-item_exp_v1.json"
    assert audit_jsonl == Path("output") / "human-item_exp_v1.audit.jsonl"


def test_write_outputs_writes_json_and_jsonl(tmp_path):
    output_json = tmp_path / "human-item_unit.json"
    audit_jsonl = tmp_path / "human-item_unit.audit.jsonl"

    write_outputs(
        output_json_path=output_json,
        audit_jsonl_path=audit_jsonl,
        results=[{"cond_1": "g.jpg", "cond_2": "r.jpg", "prompt": "prompt"}],
        audit=[{"event": "pair_accepted", "score": 0.9}],
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["cond_1"] == "g.jpg"
    assert json.loads(audit_jsonl.read_text(encoding="utf-8").strip())["event"] == "pair_accepted"
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
FAILED ... cannot import name 'build_output_paths'
```

- [ ] **Step 3: 实现输出函数和 CLI**

追加到 `D:/Project/training_pair/human_item_pairing.py`：

```python
import argparse
import logging
import os
from datetime import datetime


def build_output_paths(output_dir: Path, batch_id: str) -> tuple[Path, Path]:
    return (
        output_dir / f"human-item_{batch_id}.json",
        output_dir / f"human-item_{batch_id}.audit.jsonl",
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


def make_batch_id(batch_id: str | None) -> str:
    if batch_id:
        return batch_id
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return int(datetime.now().strftime("%Y%m%d%H%M%S"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLM-judged human-item pairings.")
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
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    batch_id = make_batch_id(args.batch_id)
    seed = make_seed(args.seed)

    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    os.environ["no_proxy"] = "localhost,127.0.0.1"

    config = PairingConfig(
        target_count=args.target_count,
        batch_id=batch_id,
        seed=seed,
        max_ref_attempts_per_gen=args.max_ref_attempts_per_gen,
        score_threshold=args.score_threshold,
        workers=args.workers,
        allow_gen_reuse=args.allow_gen_reuse,
    )

    gen_items, gen_audit = build_valid_items(args.gen_dir, args.gen_metadata_dir, "gen")
    ref_items, ref_audit = build_valid_items(args.ref_dir, args.ref_metadata_dir, "ref")
    buckets = build_size_buckets(gen_items, ref_items)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    def judge_pair(gen_item: ImageItem, ref_item: ImageItem) -> dict[str, Any]:
        return infer_pair_decision(
            client=client,
            model_name=args.model_name,
            gen_item=gen_item,
            ref_item=ref_item,
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    results, audit = run_pairing(
        buckets=buckets,
        config=config,
        judge_pair=judge_pair,
    )
    audit = gen_audit + ref_audit + audit

    output_json_path, audit_jsonl_path = build_output_paths(args.output_dir, batch_id)
    write_outputs(output_json_path, audit_jsonl_path, results, audit)

    logging.info("batch_id=%s", batch_id)
    logging.info("seed=%s", seed)
    logging.info("accepted=%s target=%s", len(results), args.target_count)
    logging.info("output_json=%s", output_json_path)
    logging.info("audit_jsonl=%s", audit_jsonl_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
21 passed
```

- [ ] **Step 5: 提交**

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "feat: add pairing cli outputs"
```

## Task 9: 使用 sample 做无真实 VLM 的 smoke test

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py`
- Modify: `D:/Project/training_pair/tests/test_human_item_pairing.py`

- [ ] **Step 1: 添加 `--dry-run-accept-all` 参数测试入口**

追加测试：

```python
from human_item_pairing import make_mock_accept_decision


def test_make_mock_accept_decision_returns_valid_prompt():
    gen_item = ImageItem("g", "g.jpg", Path("g.jpg"), Path("g.json"), "100x100", {}, "resized_width_height")
    ref_item = ImageItem(
        "r",
        "r.jpg",
        Path("r.jpg"),
        Path("r.json"),
        "100x100",
        {"object_description": "A red leather handbag.", "object_name": "handbag"},
        "resized_width_height",
    )

    decision = make_mock_accept_decision(gen_item, ref_item)

    assert decision["suitable"] is True
    assert should_accept_decision(decision, 0.75) is True
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
FAILED ... cannot import name 'make_mock_accept_decision'
```

- [ ] **Step 3: 实现 mock judge 并接入 CLI**

追加函数：

```python
def make_mock_accept_decision(gen_item: ImageItem, ref_item: ImageItem) -> dict[str, Any]:
    object_description = (
        ref_item.metadata.get("object_description")
        or ref_item.metadata.get("object_name")
        or "a holdable object"
    )
    object_description = str(object_description).strip().rstrip(".")
    prompt = (
        f"Let the person in image 1 hold {object_description} shown in image 2 "
        "in a realistic and physically coherent way, preserving object integrity "
        "and overall image consistency, while making only the minimal necessary "
        "changes and keeping everything else in image 1 unchanged."
    )
    return {
        "suitable": True,
        "score": 1.0,
        "reason": "dry-run accept-all mode",
        "action": "hold",
        "object_description": object_description,
        "prompt": prompt,
    }
```

在 `parse_args()` 中加入：

```python
    parser.add_argument("--dry-run-accept-all", action="store_true")
```

在 `main()` 中创建 `client` 前加入分支：

```python
    if args.dry_run_accept_all:
        judge_pair = make_mock_accept_decision
    else:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

        def judge_pair(gen_item: ImageItem, ref_item: ImageItem) -> dict[str, Any]:
            return infer_pair_decision(
                client=client,
                model_name=args.model_name,
                gen_item=gen_item,
                ref_item=ref_item,
                max_retries=args.max_retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
```

同时将 `--base-url`、`--api-key`、`--model-name` 改成非 required，并在非 dry-run 分支显式检查：

```python
        if not args.base_url or not args.api_key or not args.model_name:
            raise ValueError("--base-url, --api-key and --model-name are required outside dry-run mode")
```

- [ ] **Step 4: 运行单元测试**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
22 passed
```

- [ ] **Step 5: 运行 sample dry-run**

Run:

```bash
python D:/Project/training_pair/human_item_pairing.py \
  --gen-dir D:/Project/training_pair/sample/gen \
  --gen-metadata-dir D:/Project/training_pair/sample/gen_metadata \
  --ref-dir D:/Project/training_pair/sample/ref \
  --ref-metadata-dir D:/Project/training_pair/sample/ref_metadata \
  --output-dir D:/Project/training_pair/output \
  --target-count 1 \
  --batch-id dryrun_sample \
  --seed 20260601 \
  --dry-run-accept-all
```

Expected:

```text
accepted=0 target=1
```

当前 sample 的 `gen` 和 `ref` 图片目录为空，因此 dry-run 只能验证脚本不会崩溃，并会在 audit 中记录 `image_missing`。如果后续放入 sample 图片，预期应变为 `accepted=1 target=1`。

- [ ] **Step 6: 提交**

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py D:/Project/training_pair/output/human-item_dryrun_sample.audit.jsonl
git commit -m "test: add dry-run pairing mode"
```

## Task 10: 真实 VLM 小批次验证

**Files:**
- Modify: `D:/Project/training_pair/human_item_pairing.py` only if real VLM reveals response-format issues

- [ ] **Step 1: 设置真实服务参数**

使用与 `D:/Project/training_pair/output/testing_prompt_gen.py` 一致的 OpenAI-compatible 服务参数，例如：

```bash
set BASE_URL=http://10.154.39.57:8001/v1
set API_KEY=123456
set MODEL_NAME=gemma-4-31B-it
```

- [ ] **Step 2: 运行小批次**

在有真实图片的环境中运行：

```bash
python D:/Project/training_pair/human_item_pairing.py \
  --gen-dir D:/Project/training_pair/sample/gen \
  --gen-metadata-dir D:/Project/training_pair/sample/gen_metadata \
  --ref-dir D:/Project/training_pair/sample/ref \
  --ref-metadata-dir D:/Project/training_pair/sample/ref_metadata \
  --output-dir D:/Project/training_pair/output \
  --target-count 3 \
  --batch-id vlm_smoke \
  --seed 20260601 \
  --max-ref-attempts-per-gen 5 \
  --score-threshold 0.75 \
  --workers 1 \
  --base-url %BASE_URL% \
  --api-key %API_KEY% \
  --model-name %MODEL_NAME%
```

Expected:

```text
accepted=3 target=3
output_json=...human-item_vlm_smoke.json
audit_jsonl=...human-item_vlm_smoke.audit.jsonl
```

如果 accepted 小于 3，检查 `human-item_vlm_smoke.audit.jsonl` 中的 `pair_rejected`、`pair_error`、`invalid JSON` 类记录。只有当模型响应格式不稳定时，才修改 prompt 或 JSON 清洗逻辑。

- [ ] **Step 3: 验证输出 JSON 结构**

Run:

```bash
python -m json.tool D:/Project/training_pair/output/human-item_vlm_smoke.json
```

Expected:

```text
格式化后的 JSON 正常输出，无 JSONDecodeError。
```

- [ ] **Step 4: 提交真实验证后的必要修正**

如果没有代码修正，只提交生成的计划和脚本即可。如果有修正：

```bash
git add D:/Project/training_pair/human_item_pairing.py D:/Project/training_pair/tests/test_human_item_pairing.py
git commit -m "fix: harden vlm response handling"
```

## Task 11: 文档、最终测试和推送到 GitHub

**Files:**
- Create: `D:/Project/training_pair/README.md`
- Modify: `D:/Project/training_pair/docs/superpowers/plans/2026-06-01-human-item-pairing.md`

- [ ] **Step 1: 创建 README**

在 `D:/Project/training_pair/README.md` 中写入：

```markdown
# human_item_pair

Batch pairing pipeline for human-item multi-image editing data.

## Features

- Matches `gen` and `ref` images only when their metadata dimensions are identical.
- Randomizes gen traversal with a reproducible seed.
- Balances ref usage within each batch.
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

## Outputs

- `output/human-item_<batch_id>.json`
- `output/human-item_<batch_id>.audit.jsonl`
```

- [ ] **Step 2: 运行完整测试**

Run:

```bash
pytest D:/Project/training_pair/tests/test_human_item_pairing.py -v
```

Expected:

```text
22 passed
```

- [ ] **Step 3: 检查 git 状态**

Run:

```bash
git status --short
```

Expected:

```text
显示 README、计划、脚本、测试等待提交文件；没有意外的大型图片或缓存文件。
```

- [ ] **Step 4: 提交文档**

```bash
git add D:/Project/training_pair/README.md D:/Project/training_pair/docs/superpowers/specs/2026-06-01-human-item-pairing-design.md D:/Project/training_pair/docs/superpowers/plans/2026-06-01-human-item-pairing.md
git commit -m "docs: add human item pairing plan"
```

- [ ] **Step 5: 推送到 GitHub**

确认远程：

```bash
git remote -v
```

Expected:

```text
origin  https://github.com/IcelandBee/human_item_pair.git (fetch)
origin  https://github.com/IcelandBee/human_item_pair.git (push)
```

推送：

```bash
git branch -M main
git push -u origin main
```

Expected:

```text
branch 'main' set up to track 'origin/main'
```

如果远程已经有提交，先拉取并 rebase：

```bash
git pull --rebase origin main
git push -u origin main
```

## 自检结果

- Spec 覆盖：尺寸强约束、metadata 预筛、随机 gen、同批次 ref 均衡、VLM JSON 判定、输出命名、audit、seed、batch_id、错误记录、效率参数均有对应任务。
- 占位扫描：计划中没有未完成标记或空泛的延后实现步骤。
- 类型一致性：`ImageItem`、`PairingConfig`、`resolve_size_key`、`build_valid_items`、`build_size_buckets`、`choose_balanced_ref`、`run_pairing`、`parse_vlm_decision`、`should_accept_decision` 在任务间保持一致。
