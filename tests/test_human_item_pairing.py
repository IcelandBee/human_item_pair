import json
import random
from pathlib import Path

from human_item_pairing import (
    ImageItem,
    PairingConfig,
    build_output_paths,
    build_size_buckets,
    build_valid_items,
    choose_balanced_ref,
    compact_gen_metadata,
    compact_ref_metadata,
    is_valid_gen_metadata,
    is_valid_ref_metadata,
    make_mock_accept_decision,
    parse_vlm_decision,
    resolve_size_key,
    run_pairing,
    should_accept_decision,
    shuffled_gen_pass,
    write_outputs,
)


def make_item(stem: str, size_key: str = "100x100") -> ImageItem:
    return ImageItem(
        stem=stem,
        file_name=f"{stem}.jpg",
        image_path=Path(f"{stem}.jpg"),
        metadata_path=Path(f"{stem}.json"),
        size_key=size_key,
        metadata={},
        dimension_source="resized_width_height",
    )


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
        make_item("g1", "100x100"),
        make_item("g2", "200x200"),
    ]
    ref = [
        make_item("r1", "100x100"),
    ]

    buckets = build_size_buckets(gen, ref)

    assert sorted(buckets.keys()) == ["100x100"]
    assert buckets["100x100"]["gen"] == [gen[0]]
    assert buckets["100x100"]["ref"] == [ref[0]]


def test_choose_balanced_ref_prefers_least_used_ref():
    refs = [make_item("r1"), make_item("r2"), make_item("r3")]
    usage = {"r1": 3, "r2": 1, "r3": 1}
    rng = random.Random(5)

    selected = choose_balanced_ref(
        refs=refs,
        ref_usage_count=usage,
        attempted_ref_stems=set(),
        rng=rng,
    )

    assert selected is not None
    assert selected.stem in {"r2", "r3"}


def test_choose_balanced_ref_excludes_attempted_refs():
    refs = [make_item("r1"), make_item("r2")]
    usage = {"r1": 0, "r2": 0}
    rng = random.Random(7)

    selected = choose_balanced_ref(
        refs=refs,
        ref_usage_count=usage,
        attempted_ref_stems={"r1"},
        rng=rng,
    )

    assert selected is not None
    assert selected.stem == "r2"


def test_shuffled_gen_pass_is_reproducible_with_seed():
    gen_items = [make_item("g1"), make_item("g2"), make_item("g3")]

    first = shuffled_gen_pass(gen_items, random.Random(123))
    second = shuffled_gen_pass(gen_items, random.Random(123))

    assert [item.stem for item in first] == [item.stem for item in second]
    assert sorted(item.stem for item in first) == ["g1", "g2", "g3"]


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


def test_run_pairing_balances_refs_and_stops_at_target_count():
    gen_items = [make_item("g1"), make_item("g2"), make_item("g3")]
    ref_items = [make_item("r1"), make_item("r2")]
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
            "prompt": "Let the person in image 1 hold a mock object shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged.",
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
    assert sorted(ref_counts.values()) == [1, 2]


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


def test_make_mock_accept_decision_returns_valid_prompt():
    gen_item = make_item("g")
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
