import json
import random
from pathlib import Path

from virtual_clothing_pairing import (
    ImageItem,
    PairingConfig,
    VIRTUAL_CLOTHING_SYSTEM_PROMPT,
    build_output_paths,
    build_size_buckets,
    build_valid_items,
    choose_balanced_ref,
    compact_gen_metadata,
    compact_ref_metadata,
    format_summary,
    is_valid_gen_metadata,
    is_valid_ref_metadata,
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
        "resized_width": 880,
        "resized_height": 1184,
        "preferred_resolution": [1024, 1024],
    })

    assert size_key == "880x1184"
    assert source == "resized_width_height"


def test_valid_gen_metadata_requires_visible_single_person_with_clothes():
    metadata = {
        "resized_width": 880,
        "resized_height": 1184,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "clothes_visible": "yes",
                "shot_type": "half_body",
                "person_size_in_frame": "large",
            }
        },
    }

    assert is_valid_gen_metadata(metadata) is True


def test_invalid_gen_metadata_rejects_missing_visible_clothes():
    metadata = {
        "resized_width": 880,
        "resized_height": 1184,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "clothes_visible": "no",
                "shot_type": "half_body",
            }
        },
    }

    assert is_valid_gen_metadata(metadata) is False


def test_valid_ref_metadata_requires_single_complete_high_quality_garment():
    metadata = {
        "resized_width": 880,
        "resized_height": 1184,
        "original_annotation": {
            "annotation": {
                "is_valid_garment_image": "yes",
                "garment_count": "single",
                "display_mode": "flat",
                "background_type": "white",
                "garment_completeness": "complete",
                "image_quality": "high",
                "garment_position_type": "top",
                "garment_category": "shirt",
            }
        },
    }

    assert is_valid_ref_metadata(metadata) is True


def test_invalid_ref_metadata_rejects_multi_garment():
    metadata = {
        "resized_width": 880,
        "resized_height": 1184,
        "original_annotation": {
            "annotation": {
                "is_valid_garment_image": "yes",
                "garment_count": "multiple",
                "display_mode": "flat",
                "background_type": "white",
                "garment_completeness": "complete",
                "image_quality": "high",
            }
        },
    }

    assert is_valid_ref_metadata(metadata) is False


def test_build_valid_items_loads_virtual_clothing_metadata(tmp_path):
    image_dir = tmp_path / "gen"
    metadata_dir = tmp_path / "gen_metadata"
    image_dir.mkdir()
    metadata_dir.mkdir()
    (image_dir / "000001.jpg").write_bytes(b"fake")
    (metadata_dir / "000001.json").write_text(
        json.dumps({
            "file_name": "000001.jpg",
            "resized_width": 880,
            "resized_height": 1184,
            "original_annotation": {
                "annotation": {
                    "person_count": "1",
                    "head_visible": "yes",
                    "clothes_visible": "yes",
                    "shot_type": "half_body",
                }
            },
        }),
        encoding="utf-8",
    )

    items, audit = build_valid_items(image_dir, metadata_dir, "gen")

    assert len(items) == 1
    assert items[0].size_key == "880x1184"
    assert audit == []


def test_build_size_buckets_keeps_only_matching_dimensions():
    gen_items = [make_item("g1", "880x1184"), make_item("g2", "1024x1024")]
    ref_items = [make_item("r1", "880x1184")]

    buckets = build_size_buckets(gen_items, ref_items)

    assert sorted(buckets) == ["880x1184"]
    assert buckets["880x1184"]["gen"] == [gen_items[0]]
    assert buckets["880x1184"]["ref"] == [ref_items[0]]


def test_choose_balanced_ref_prefers_least_used_ref():
    refs = [make_item("r1"), make_item("r2"), make_item("r3")]
    selected = choose_balanced_ref(
        refs=refs,
        ref_usage_count={"r1": 3, "r2": 1, "r3": 1},
        attempted_ref_stems=set(),
        rng=random.Random(5),
    )

    assert selected is not None
    assert selected.stem in {"r2", "r3"}


def test_shuffled_gen_pass_is_reproducible():
    items = [make_item("g1"), make_item("g2"), make_item("g3")]

    first = shuffled_gen_pass(items, random.Random(123))
    second = shuffled_gen_pass(items, random.Random(123))

    assert [item.stem for item in first] == [item.stem for item in second]


def test_compact_metadata_keeps_virtual_clothing_fields():
    gen_compact = compact_gen_metadata({
        "file_name": "000001.jpg",
        "resized_width": 880,
        "resized_height": 1184,
        "original_annotation": {
            "annotation": {
                "gender": "female",
                "shot_type": "half_body",
                "clothes_visible": "yes",
                "person_size_in_frame": "large",
            }
        },
        "unused": "x" * 1000,
    })
    ref_compact = compact_ref_metadata({
        "file_name": "000002.jpg",
        "original_annotation": {
            "annotation": {
                "target_user_group": "female",
                "pattern_type": "denim",
                "garment_position_type": "bottom",
                "garment_category": "pants",
            }
        },
    })

    assert gen_compact["shot_type"] == "half_body"
    assert gen_compact["clothes_visible"] == "yes"
    assert "unused" not in gen_compact
    assert ref_compact["garment_position_type"] == "bottom"
    assert ref_compact["garment_category"] == "pants"


def test_virtual_clothing_prompt_contains_required_judgement_points():
    prompt = VIRTUAL_CLOTHING_SYSTEM_PROMPT.lower()

    assert "corresponding body region" in prompt
    assert "replace visible clothing" in prompt
    assert "top, bottom, or one-piece" in prompt
    assert "soft compatibility signal" in prompt
    assert "major body pose changes" in prompt


def test_parse_and_accept_virtual_clothing_decision():
    raw = """```json
{
  "suitable": true,
  "score": 0.88,
  "reason": "The upper body is visible and the shirt can replace the current top.",
  "source_clothes": "black t-shirt",
  "reference_clothes": "white shirt",
  "prompt": "Replace the black t-shirt worn by the person in image 1 with the white shirt in image 2, while making minimal changes and preserving the original pose of the person."
}
```"""

    decision = parse_vlm_decision(raw)

    assert should_accept_decision(decision, 0.75) is True


def test_rejects_non_virtual_clothing_prompt_format():
    decision = {
        "suitable": True,
        "score": 0.95,
        "prompt": "Let the person in image 1 hold a shirt shown in image 2.",
    }

    assert should_accept_decision(decision, 0.75) is False


def test_run_pairing_outputs_virtual_clothing_prompt():
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
            "source_clothes": "black t-shirt",
            "reference_clothes": "white shirt",
            "prompt": "Replace the black t-shirt worn by the person in image 1 with the white shirt in image 2, while making minimal changes and preserving the original pose of the person.",
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert results[0]["prompt"].startswith("Replace the")
    assert len([row for row in audit if row["event"] == "pair_accepted"]) == 2


def test_output_paths_use_virtual_clothing_prefix():
    output_json, audit_jsonl = build_output_paths(Path("output"), "exp_v1")

    assert output_json == Path("output") / "virtual-clothing_exp_v1.json"
    assert audit_jsonl == Path("output") / "virtual-clothing_exp_v1.audit.jsonl"


def test_mock_accept_decision_uses_garment_category():
    decision = make_mock_accept_decision(
        make_item("g"),
        make_item("r", metadata={
            "original_annotation": {
                "annotation": {
                    "garment_category": "shirt",
                    "pattern_type": "white",
                }
            }
        }),
    )

    assert should_accept_decision(decision, 0.75) is True
    assert "white shirt" in decision["prompt"]


def test_progress_and_summary_helpers():
    line = render_progress_bar("配对判断", 3, 10, 8, 12, width=10)
    summary = format_summary(
        batch_id="exp",
        seed=1,
        target_count=10,
        accepted_count=3,
        output_json_path=Path("output/virtual-clothing_exp.json"),
        audit_jsonl_path=Path("output/virtual-clothing_exp.audit.jsonl"),
    )

    assert " 30.0%" in line
    assert "accepted=3/10" in line
    assert "batch_id=exp" in summary
    assert "accepted=3" in summary
