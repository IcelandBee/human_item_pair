import json
import random
from pathlib import Path

from hairstyle_pairing import (
    FIXED_HAIRSTYLE_PROMPT,
    HAIRSTYLE_SYSTEM_PROMPT,
    ImageItem,
    PairingConfig,
    build_output_paths,
    build_size_buckets,
    build_valid_items,
    choose_balanced_ref,
    compact_person_metadata,
    format_summary,
    is_valid_gen_metadata,
    is_valid_person_metadata,
    make_mock_accept_decision,
    parse_vlm_decision,
    render_progress_bar,
    resolve_size_key,
    run_pairing,
    should_accept_decision,
    shuffled_gen_pass,
    write_outputs,
)


def make_item(stem: str, size_key: str = "100x100", metadata=None) -> ImageItem:
    return ImageItem(
        stem=stem,
        file_name=f"{stem}.jpg",
        image_path=Path(f"{stem}.jpg"),
        metadata_path=Path(f"{stem}.json"),
        size_key=size_key,
        metadata=metadata or {},
        dimension_source="resized_width_height",
    )


def test_resolve_size_key_prefers_resized_dimensions():
    size_key, source = resolve_size_key({
        "resized_width": 1248,
        "resized_height": 832,
        "preferred_resolution": [512, 512],
    })

    assert size_key == "1248x832"
    assert source == "resized_width_height"


def test_valid_person_metadata_requires_visible_hair_and_clear_face():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "hair_visible": "yes",
                "facial_features_clear": "yes",
                "person_size_in_frame": "large",
            }
        },
    }

    assert is_valid_person_metadata(metadata) is True


def test_valid_gen_metadata_requires_close_up_or_half_body_shot_type():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "hair_visible": "yes",
                "facial_features_clear": "yes",
                "shot_type": "close_up",
            }
        },
    }

    assert is_valid_gen_metadata(metadata) is True

    metadata["original_annotation"]["annotation"]["shot_type"] = "half_body"
    assert is_valid_gen_metadata(metadata) is True

    metadata["original_annotation"]["annotation"]["shot_type"] = "full_body"
    assert is_valid_gen_metadata(metadata) is False


def test_ref_metadata_does_not_apply_gen_shot_type_filter():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "hair_visible": "yes",
                "facial_features_clear": "yes",
                "shot_type": "full_body",
            }
        },
    }

    assert is_valid_person_metadata(metadata) is True


def test_invalid_person_metadata_rejects_hidden_hair():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "hair_visible": "no",
                "facial_features_clear": "yes",
            }
        },
    }

    assert is_valid_person_metadata(metadata) is False


def test_person_metadata_does_not_reject_gender_or_face_direction():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "gender": "female",
                "head_visible": "yes",
                "hair_visible": "yes",
                "facial_features_clear": "yes",
                "face_direction": "profile",
            }
        },
    }

    assert is_valid_person_metadata(metadata) is True


def test_build_valid_items_loads_hairstyle_metadata(tmp_path):
    image_dir = tmp_path / "gen"
    metadata_dir = tmp_path / "gen_metadata"
    image_dir.mkdir()
    metadata_dir.mkdir()
    (image_dir / "000001.jpg").write_bytes(b"fake")
    (metadata_dir / "000001.json").write_text(
        json.dumps({
            "file_name": "000001.jpg",
            "resized_width": 1248,
            "resized_height": 832,
            "original_annotation": {
                "annotation": {
                    "person_count": "1",
                    "head_visible": "yes",
                    "hair_visible": "yes",
                    "facial_features_clear": "yes",
                    "shot_type": "half_body",
                }
            },
        }),
        encoding="utf-8",
    )

    items, audit = build_valid_items(image_dir, metadata_dir, "gen")

    assert len(items) == 1
    assert items[0].size_key == "1248x832"
    assert audit == []


def test_build_size_buckets_keeps_matching_dimensions():
    gen_items = [make_item("g1", "1248x832"), make_item("g2", "832x1248")]
    ref_items = [make_item("r1", "1248x832")]

    buckets = build_size_buckets(gen_items, ref_items)

    assert sorted(buckets) == ["1248x832"]


def test_choose_balanced_ref_prefers_least_used_ref():
    refs = [make_item("r1"), make_item("r2"), make_item("r3")]
    selected = choose_balanced_ref(
        refs=refs,
        ref_usage_count={"r1": 4, "r2": 1, "r3": 1},
        attempted_ref_stems=set(),
        rng=random.Random(3),
    )

    assert selected is not None
    assert selected.stem in {"r2", "r3"}


def test_shuffled_gen_pass_is_reproducible():
    items = [make_item("g1"), make_item("g2"), make_item("g3")]

    first = shuffled_gen_pass(items, random.Random(123))
    second = shuffled_gen_pass(items, random.Random(123))

    assert [item.stem for item in first] == [item.stem for item in second]


def test_compact_person_metadata_keeps_hair_fields():
    compact = compact_person_metadata({
        "file_name": "000001.jpg",
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "gender": "male",
                "hair_visible": "yes",
                "hair_color": "brown",
                "head_visible": "yes",
                "facial_features_clear": "yes",
                "face_direction": "frontal",
            }
        },
        "unused": "x" * 1000,
    })

    assert compact["hair_visible"] == "yes"
    assert compact["hair_color"] == "brown"
    assert compact["face_direction"] == "frontal"
    assert "unused" not in compact


def test_hairstyle_prompt_contains_confirmed_judgement_scope():
    prompt = HAIRSTYLE_SYSTEM_PROMPT.lower()

    assert "both images show clear, visible hair" in prompt
    assert "hairstyle difference" in prompt
    assert "too similar" in prompt
    assert "hats, headwear, headscarves" in prompt
    assert "hair accessories" in prompt
    assert "blur, severe crop, strong occlusion, or complex background" in prompt
    assert "do not reject only because of gender" in prompt
    assert "do not reject only because the transferred hairstyle may cover part of the face" in prompt


def test_parse_and_accept_fixed_hairstyle_prompt():
    raw = f"""```json
{{
  "suitable": true,
  "score": 0.86,
  "reason": "Both hairstyles are clear and no hats are present.",
  "hairstyle_difference": "different",
  "prompt": "{FIXED_HAIRSTYLE_PROMPT}"
}}
```"""

    decision = parse_vlm_decision(raw)

    assert should_accept_decision(decision, 0.75) is True


def test_rejects_too_similar_hairstyle_difference():
    decision = {
        "suitable": True,
        "score": 0.95,
        "hairstyle_difference": "too_similar",
        "prompt": FIXED_HAIRSTYLE_PROMPT,
    }

    assert should_accept_decision(decision, 0.75) is False


def test_rejects_non_hairstyle_prompt_format():
    decision = {
        "suitable": True,
        "score": 0.95,
        "prompt": "Replace the shirt worn by the person in image 1.",
    }

    assert should_accept_decision(decision, 0.75) is False


def test_run_pairing_outputs_fixed_prompt():
    gen_items = [make_item("g1"), make_item("g2")]
    ref_items = [make_item("r1")]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=False,
    )

    def fake_judge(gen_item, ref_item):
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "hairstyle_difference": "different",
            "prompt": FIXED_HAIRSTYLE_PROMPT,
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert results[0]["prompt"] == FIXED_HAIRSTYLE_PROMPT
    assert len([row for row in audit if row["event"] == "pair_accepted"]) == 2


def test_output_paths_use_hairstyle_prefix():
    output_json, audit_jsonl = build_output_paths(Path("output"), "exp_v1")

    assert output_json == Path("output") / "hairstyle_exp_v1.json"
    assert audit_jsonl == Path("output") / "hairstyle_exp_v1.audit.jsonl"


def test_mock_accept_decision_returns_fixed_prompt():
    decision = make_mock_accept_decision(make_item("g"), make_item("r"))

    assert decision["prompt"] == FIXED_HAIRSTYLE_PROMPT
    assert decision["hairstyle_difference"] == "different"
    assert should_accept_decision(decision, 0.75) is True


def test_progress_and_summary_helpers():
    line = render_progress_bar("pairing", 2, 10, 4, 7, width=10)
    summary = format_summary(
        batch_id="exp",
        seed=1,
        target_count=10,
        accepted_count=2,
        output_json_path=Path("output/hairstyle_exp.json"),
        audit_jsonl_path=Path("output/hairstyle_exp.audit.jsonl"),
    )

    assert " 20.0%" in line
    assert "accepted=2/10" in line
    assert "batch_id=exp" in summary


def test_write_outputs_writes_json_and_audit(tmp_path):
    output_json = tmp_path / "hairstyle_unit.json"
    audit_jsonl = tmp_path / "hairstyle_unit.audit.jsonl"

    write_outputs(
        output_json_path=output_json,
        audit_jsonl_path=audit_jsonl,
        results=[{"cond_1": "g.jpg", "cond_2": "r.jpg", "prompt": FIXED_HAIRSTYLE_PROMPT}],
        audit=[{"event": "pair_accepted", "score": 0.9}],
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["prompt"] == FIXED_HAIRSTYLE_PROMPT
    assert json.loads(audit_jsonl.read_text(encoding="utf-8").strip())["event"] == "pair_accepted"
