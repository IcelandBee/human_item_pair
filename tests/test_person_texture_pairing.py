import json
import random
from pathlib import Path

from person_texture_pairing import (
    PERSON_TEXTURE_SYSTEM_PROMPT,
    ImageItem,
    PairingConfig,
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


def make_gen_metadata(width: int = 832, height: int = 1248):
    return {
        "file_name": "000001.jpg",
        "resized_width": width,
        "resized_height": height,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "shot_type": "half_body",
                "clothes_visible": "yes",
                "person_size_in_frame": "large",
                "person_prominence": "close",
            }
        },
    }


def make_ref_metadata(width: int = 832, height: int = 1248):
    return {
        "file_name": "000002.jpg",
        "resized_width": width,
        "resized_height": height,
        "source_record_id": 1,
        "source_record_filename": "texture.jpg",
        "original_record": {"id": 1, "filename": "texture.jpg"},
    }


def make_item(stem: str, size_key: str = "832x1248", metadata=None) -> ImageItem:
    return ImageItem(
        stem=stem,
        file_name=f"{stem}.jpg",
        image_path=Path(f"{stem}.jpg"),
        metadata_path=Path(f"{stem}.json"),
        size_key=size_key,
        metadata=metadata or make_gen_metadata(),
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
    assert is_valid_gen_metadata(make_gen_metadata()) is True


def test_invalid_gen_metadata_rejects_missing_visible_clothes():
    metadata = make_gen_metadata()
    metadata["original_annotation"]["annotation"]["clothes_visible"] = "no"

    assert is_valid_gen_metadata(metadata) is False


def test_ref_metadata_only_requires_valid_size():
    assert is_valid_ref_metadata(make_ref_metadata()) is True


def test_invalid_ref_metadata_rejects_missing_size():
    metadata = make_ref_metadata()
    metadata.pop("resized_width")
    metadata.pop("resized_height")
    metadata.pop("original_width", None)
    metadata.pop("original_height", None)
    metadata.pop("preferred_resolution", None)

    assert is_valid_ref_metadata(metadata) is False


def test_build_valid_items_loads_person_texture_metadata(tmp_path):
    image_dir = tmp_path / "gen"
    metadata_dir = tmp_path / "gen_metadata"
    image_dir.mkdir()
    metadata_dir.mkdir()
    (image_dir / "000001.jpg").write_bytes(b"fake")
    (metadata_dir / "000001.json").write_text(json.dumps(make_gen_metadata()), encoding="utf-8")

    items, audit = build_valid_items(image_dir, metadata_dir, "gen")

    assert len(items) == 1
    assert items[0].size_key == "832x1248"
    assert audit == []


def test_build_size_buckets_keeps_matching_dimensions():
    gen_items = [make_item("g1", "832x1248"), make_item("g2", "1248x832")]
    ref_items = [make_item("r1", "832x1248", make_ref_metadata())]

    buckets = build_size_buckets(gen_items, ref_items)

    assert sorted(buckets) == ["832x1248"]


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


def test_compact_metadata_keeps_person_texture_fields():
    gen_compact = compact_gen_metadata(make_gen_metadata())
    ref_compact = compact_ref_metadata(make_ref_metadata())

    assert gen_compact["shot_type"] == "half_body"
    assert gen_compact["clothes_visible"] == "yes"
    assert ref_compact["source_record_id"] == 1
    assert ref_compact["source_record_filename"] == "texture.jpg"


def test_person_texture_prompt_contains_confirmed_judgement_scope():
    prompt = PERSON_TEXTURE_SYSTEM_PROMPT.lower()

    assert "target garment" in prompt
    assert "clear, editable clothing region" in prompt
    assert "slight blur" in prompt
    assert "reference texture" in prompt
    assert "too similar" in prompt
    assert "complexity" in prompt
    assert "do not reject metallic" in prompt
    assert "preserve the garment's shape and fit" in prompt


def test_parse_and_accept_person_texture_decision():
    raw = """```json
{
  "suitable": true,
  "score": 0.88,
  "reason": "The target sweater is clear and the reference texture is visibly different.",
  "target_garment": "black turtleneck",
  "texture_difference": "different",
  "complexity_match": "compatible",
  "prompt": "Transfer the textures from image 2 to the black turtleneck of the person in image 1. Preserve the garment's shape and fit. Edit only the black turtleneck region, keeping all other areas unchanged."
}
```"""

    decision = parse_vlm_decision(raw)

    assert should_accept_decision(decision, 0.75) is True


def test_rejects_too_similar_texture_difference():
    decision = {
        "suitable": True,
        "score": 0.92,
        "target_garment": "brown skirt",
        "texture_difference": "too_similar",
        "complexity_match": "compatible",
        "prompt": "Transfer the textures from image 2 to the brown skirt of the person in image 1. Preserve the garment's shape and fit. Edit only the brown skirt region, keeping all other areas unchanged.",
    }

    assert should_accept_decision(decision, 0.75) is False


def test_rejects_incompatible_texture_complexity():
    decision = {
        "suitable": True,
        "score": 0.9,
        "target_garment": "thin strap top",
        "texture_difference": "different",
        "complexity_match": "incompatible",
        "prompt": "Transfer the textures from image 2 to the thin strap top of the person in image 1. Preserve the garment's shape and fit. Edit only the thin strap top region, keeping all other areas unchanged.",
    }

    assert should_accept_decision(decision, 0.75) is False


def test_rejects_non_template_prompt():
    decision = {
        "suitable": True,
        "score": 0.95,
        "target_garment": "shirt",
        "texture_difference": "different",
        "complexity_match": "compatible",
        "prompt": "Replace the shirt with image 2.",
    }

    assert should_accept_decision(decision, 0.75) is False


def test_get_output_dimensions_uses_resized_dimensions():
    from person_texture_pairing import get_output_dimensions

    assert get_output_dimensions(make_item("g", metadata=make_gen_metadata(880, 1184))) == {
        "width": 880,
        "height": 1184,
    }


def test_run_pairing_outputs_dynamic_prompt_and_dimensions():
    gen_item = make_item("g1", "880x1184", make_gen_metadata(880, 1184))
    ref_item = make_item("r1", "880x1184", make_ref_metadata(880, 1184))

    def fake_judge(_gen, _ref):
        return {
            "suitable": True,
            "score": 0.91,
            "reason": "Good target garment and distinct texture.",
            "target_garment": "black turtleneck",
            "texture_difference": "different",
            "complexity_match": "compatible",
            "prompt": "Transfer the textures from image 2 to the black turtleneck of the person in image 1. Preserve the garment's shape and fit. Edit only the black turtleneck region, keeping all other areas unchanged.",
        }

    results, audit = run_pairing(
        size_buckets={"880x1184": {"gen": [gen_item], "ref": [ref_item]}},
        config=PairingConfig(
            target_count=1,
            batch_id="test",
            seed=7,
            max_ref_attempts_per_gen=1,
            score_threshold=0.75,
            workers=1,
            allow_gen_reuse=False,
        ),
        judge_pair=fake_judge,
    )

    assert len(results) == 1
    assert results[0]["prompt"].startswith("Transfer the textures from image 2 to the black turtleneck")
    assert results[0]["width"] == 880
    assert results[0]["height"] == 1184
    assert audit[-1]["target_garment"] == "black turtleneck"


def test_output_paths_use_person_texture_prefix(tmp_path):
    output_json, audit_jsonl = build_output_paths(tmp_path, "exp_v1")

    assert output_json.name == "person-texture_exp_v1.json"
    assert audit_jsonl.name == "person-texture_exp_v1.audit.jsonl"


def test_mock_accept_decision_returns_template_prompt():
    decision = make_mock_accept_decision(make_item("g"), make_item("r"))

    assert decision["target_garment"]
    assert decision["texture_difference"] == "different"
    assert should_accept_decision(decision, 0.75) is True


def test_progress_and_summary_helpers(tmp_path):
    progress = render_progress_bar({
        "accepted": 5,
        "target": 10,
        "processed_gen": 7,
        "attempts": 12,
    })
    summary = format_summary(
        batch_id="b1",
        seed=123,
        target=10,
        accepted=5,
        output_path=tmp_path / "out.json",
        audit_path=tmp_path / "audit.jsonl",
    )

    assert "50.0%" in progress
    assert "[pairing]" in progress
    assert "accepted=5" in summary


def test_write_outputs_writes_json_and_audit(tmp_path):
    output_json = tmp_path / "out.json"
    audit_jsonl = tmp_path / "audit.jsonl"

    write_outputs(
        output_json,
        audit_jsonl,
        [{"cond_1": "a.jpg", "cond_2": "b.jpg", "prompt": "p"}],
        [{"event": "pair_accepted", "target_garment": "shirt"}],
    )

    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["prompt"] == "p"
    assert json.loads(audit_jsonl.read_text(encoding="utf-8").strip())["target_garment"] == "shirt"
