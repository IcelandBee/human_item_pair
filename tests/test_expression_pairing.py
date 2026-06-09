import json
import random
from pathlib import Path

import expression_pairing as expression_pairing_module
from PIL import Image
from expression_pairing import (
    EXPRESSION_SYSTEM_PROMPT,
    FIXED_EXPRESSION_PROMPT,
    ImageItem,
    PairingConfig,
    build_output_paths,
    build_size_buckets,
    build_valid_items,
    choose_balanced_ref,
    choose_next_ref,
    compact_person_metadata,
    format_summary,
    get_output_dimensions,
    get_expression,
    is_valid_gen_metadata,
    is_valid_ref_metadata,
    make_mock_accept_decision,
    materialize_pair_outputs,
    parse_vlm_decision,
    render_progress_bar,
    resolve_size_key,
    run_pairing,
    should_accept_decision,
    shuffled_gen_pass,
    write_outputs,
)


def make_metadata(expression: str = "neutral", width: int = 1248, height: int = 832):
    return {
        "resized_width": width,
        "resized_height": height,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "expression": expression,
                "facial_features_clear": "yes",
                "face_direction": "frontal",
            }
        },
    }


def make_item(stem: str, size_key: str = "1248x832", metadata=None) -> ImageItem:
    return ImageItem(
        stem=stem,
        file_name=f"{stem}.jpg",
        image_path=Path(f"{stem}.jpg"),
        metadata_path=Path(f"{stem}.json"),
        size_key=size_key,
        metadata=metadata or make_metadata(),
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


def test_valid_gen_metadata_requires_neutral_clear_face():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "expression": "neutral",
                "facial_features_clear": "yes",
                "face_direction": "frontal",
                "person_size_in_frame": "large",
            }
        },
    }

    assert is_valid_gen_metadata(metadata) is True


def test_invalid_gen_metadata_rejects_non_neutral_expression():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "expression": "smile",
                "facial_features_clear": "yes",
                "face_direction": "frontal",
            }
        },
    }

    assert is_valid_gen_metadata(metadata) is False


def test_valid_ref_metadata_accepts_clear_non_neutral_expression():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "expression": "worried",
                "facial_features_clear": "yes",
                "face_direction": "near_frontal",
            }
        },
    }

    assert is_valid_ref_metadata(metadata) is True


def test_invalid_ref_metadata_rejects_neutral_expression():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "expression": "neutral",
                "facial_features_clear": "yes",
                "face_direction": "frontal",
            }
        },
    }

    assert is_valid_ref_metadata(metadata) is False


def test_expression_metadata_does_not_reject_gender_age_makeup_or_hair():
    metadata = {
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "gender": "female",
                "head_visible": "yes",
                "expression": "angry",
                "facial_features_clear": "yes",
                "face_direction": "near_frontal",
                "obvious_makeup": "yes",
                "hair_visible": "partial",
            }
        },
    }

    assert is_valid_ref_metadata(metadata) is True


def test_build_valid_items_loads_expression_metadata(tmp_path):
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
                    "expression": "neutral",
                    "facial_features_clear": "yes",
                    "face_direction": "frontal",
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


def test_compact_person_metadata_keeps_expression_fields():
    compact = compact_person_metadata({
        "file_name": "000001.jpg",
        "resized_width": 1248,
        "resized_height": 832,
        "original_annotation": {
            "annotation": {
                "gender": "male",
                "expression": "worried",
                "head_visible": "yes",
                "facial_features_clear": "yes",
                "face_direction": "frontal",
                "obvious_makeup": "no",
            }
        },
        "unused": "x" * 1000,
    })

    assert compact["expression"] == "worried"
    assert compact["facial_features_clear"] == "yes"
    assert compact["face_direction"] == "frontal"
    assert "unused" not in compact


def test_expression_prompt_contains_confirmed_judgement_scope():
    prompt = EXPRESSION_SYSTEM_PROMPT.lower()

    assert "eyes, eyebrows, and mouth" in prompt
    assert "clear and visible" in prompt
    assert "do not reject because the expression category is ambiguous" in prompt
    assert "gender, age, hairstyle, or makeup" in prompt
    assert "face direction and visible face region are compatible" in prompt
    assert "pose, head angle, and background unchanged" in prompt
    assert "expression difference" in prompt
    assert "too similar" in prompt


def test_parse_and_accept_fixed_expression_prompt():
    raw = f"""```json
{{
  "suitable": true,
  "score": 0.86,
  "reason": "Both faces are clear and expression regions are visible.",
  "prompt": "{FIXED_EXPRESSION_PROMPT}"
}}
```"""

    decision = parse_vlm_decision(raw)

    assert should_accept_decision(decision, 0.75) is True


def test_rejects_non_expression_prompt_format():
    decision = {
        "suitable": True,
        "score": 0.95,
        "prompt": "Transfer the hairstyle from image 2 to image 1.",
    }

    assert should_accept_decision(decision, 0.75) is False


def test_get_output_dimensions_halves_resized_dimensions():
    item = make_item("g", metadata={"resized_width": 1248, "resized_height": 832})

    assert get_output_dimensions(item) == {"width": 624, "height": 416}


def test_get_expression_reads_annotation():
    assert get_expression(make_item("r", metadata=make_metadata("sad"))) == "sad"


def test_choose_next_ref_respects_smile_and_big_laugh_soft_caps():
    refs_by_expression = {
        "smile": [make_item("smile1", metadata=make_metadata("smile"))],
        "big_laugh": [make_item("laugh1", metadata=make_metadata("big_laugh"))],
        "angry": [make_item("angry1", metadata=make_metadata("angry"))],
        "sad": [make_item("sad1", metadata=make_metadata("sad"))],
    }
    accepted_by_expression = {"smile": 2, "big_laugh": 2, "angry": 3, "sad": 3}
    ref_usage = {"smile1": 0, "laugh1": 0, "angry1": 0, "sad1": 0}

    selected = choose_next_ref(
        refs_by_expression=refs_by_expression,
        accepted_by_expression=accepted_by_expression,
        ref_usage_count=ref_usage,
        blocked_ref_stems=set(),
        rng=random.Random(1),
        max_smile_ratio=0.2,
        max_big_laugh_ratio=0.2,
    )

    assert selected is not None
    assert get_expression(selected) in {"angry", "sad"}


def test_choose_next_ref_falls_back_to_capped_expression_when_needed():
    refs_by_expression = {
        "smile": [make_item("smile1", metadata=make_metadata("smile"))],
        "big_laugh": [make_item("laugh1", metadata=make_metadata("big_laugh"))],
    }
    accepted_by_expression = {"smile": 2, "big_laugh": 2}
    ref_usage = {"smile1": 0, "laugh1": 0}

    selected = choose_next_ref(
        refs_by_expression=refs_by_expression,
        accepted_by_expression=accepted_by_expression,
        ref_usage_count=ref_usage,
        blocked_ref_stems={"laugh1"},
        rng=random.Random(1),
        max_smile_ratio=0.2,
        max_big_laugh_ratio=0.2,
    )

    assert selected is not None
    assert get_expression(selected) == "smile"


def test_run_pairing_is_ref_driven_and_limits_smile_big_laugh_distribution():
    gen_items = [
        make_item("g1", metadata=make_metadata("neutral")),
        make_item("g2", metadata=make_metadata("neutral")),
        make_item("g3", metadata=make_metadata("neutral")),
        make_item("g4", metadata=make_metadata("neutral")),
        make_item("g5", metadata=make_metadata("neutral")),
    ]
    ref_items = [
        make_item("smile1", metadata=make_metadata("smile")),
        make_item("laugh1", metadata=make_metadata("big_laugh")),
        make_item("angry1", metadata=make_metadata("angry")),
        make_item("sad1", metadata=make_metadata("sad")),
        make_item("worried1", metadata=make_metadata("worried")),
    ]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=5,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=3,
        max_ref_attempts_per_pass=30,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=False,
        max_smile_ratio=0.2,
        max_big_laugh_ratio=0.2,
    )

    def fake_judge(gen_item, ref_item):
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "prompt": FIXED_EXPRESSION_PROMPT,
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 5
    assert results[0]["prompt"] == FIXED_EXPRESSION_PROMPT
    assert results[0]["width"] == 624
    assert results[0]["height"] == 416
    accepted_rows = [row for row in audit if row["event"] == "pair_accepted"]
    accepted_expressions = [row["ref_expression"] for row in accepted_rows]
    assert len(accepted_rows) == 5
    assert accepted_expressions.count("smile") <= 1
    assert accepted_expressions.count("big_laugh") <= 1
    assert sum(expr not in {"smile", "big_laugh"} for expr in accepted_expressions) >= 3


def test_run_pairing_limits_bad_ref_attempts_per_pass_and_resets_next_pass(monkeypatch):
    gen_items = [make_item(f"g{i}") for i in range(10)]
    ref_items = [
        make_item("r_bad", metadata=make_metadata("angry")),
        make_item("r_good", metadata=make_metadata("sad")),
    ]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        expression_pairing_module,
        "choose_next_ref",
        lambda refs_by_expression, accepted_by_expression, ref_usage_count, blocked_ref_stems, rng, max_smile_ratio=1.0, max_big_laugh_ratio=1.0: (
            next(
                (
                    ref
                    for ref in sorted(
                        [item for refs in refs_by_expression.values() for item in refs],
                        key=lambda item: item.stem,
                    )
                    if ref.stem not in blocked_ref_stems
                ),
                None,
            )
        ),
    )
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=0,
        max_ref_attempts_per_gen=2,
        max_ref_attempts_per_pass=3,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=True,
    )
    calls_by_ref = {"r_bad": 0, "r_good": 0}

    def fake_judge(_gen_item, ref_item):
        calls_by_ref[ref_item.stem] += 1
        accepted = ref_item.stem == "r_good"
        return {
            "suitable": accepted,
            "score": 0.9 if accepted else 0.1,
            "reason": "mock accepted" if accepted else "mock rejected",
            "prompt": FIXED_EXPRESSION_PROMPT if accepted else "",
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert calls_by_ref == {"r_bad": 6, "r_good": 2}
    limit_events = [row for row in audit if row["event"] == "ref_pass_attempt_limit_reached"]
    assert len(limit_events) == 2
    assert {row["attempted_count"] for row in limit_events} == {3}
    assert {row["max_ref_attempts_per_pass"] for row in limit_events} == {3}


def test_run_pairing_does_not_cap_successful_ref_reuse_across_passes():
    gen_items = [make_item("g1")]
    ref_items = [make_item("r_good", metadata=make_metadata("sad"))]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=4,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        max_ref_attempts_per_pass=1,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=True,
    )
    calls = 0

    def fake_judge(_gen_item, _ref_item):
        nonlocal calls
        calls += 1
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "prompt": FIXED_EXPRESSION_PROMPT,
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 4
    assert calls == 4
    assert not [row for row in audit if row["event"] == "ref_pass_attempt_limit_reached"]


def test_run_pairing_allows_only_one_pair_per_ref_per_pass(monkeypatch):
    gen_items = [make_item("g1"), make_item("g2")]
    ref_items = [make_item("r_good", metadata=make_metadata("sad"))]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        expression_pairing_module,
        "choose_next_ref",
        lambda refs_by_expression, accepted_by_expression, ref_usage_count, blocked_ref_stems, rng, max_smile_ratio=1.0, max_big_laugh_ratio=1.0: (
            None if "r_good" in blocked_ref_stems else refs_by_expression["sad"][0]
        ),
    )
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=2,
        max_ref_attempts_per_pass=30,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=True,
    )
    calls = []

    def fake_judge(gen_item, ref_item):
        calls.append((gen_item.stem, ref_item.stem))
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "prompt": FIXED_EXPRESSION_PROMPT,
        }

    results, _audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1] == "r_good"


def test_output_paths_use_expression_prefix():
    output_json, audit_jsonl = build_output_paths(Path("output"), "exp_v1")

    assert output_json == Path("output") / "expression_exp_v1.json"
    assert audit_jsonl == Path("output") / "expression_exp_v1.audit.jsonl"


def test_mock_accept_decision_returns_fixed_prompt():
    decision = make_mock_accept_decision(make_item("g"), make_item("r"))

    assert decision["prompt"] == FIXED_EXPRESSION_PROMPT
    assert should_accept_decision(decision, 0.75) is True


def test_progress_and_summary_helpers():
    line = render_progress_bar("pairing", 2, 10, 4, 7, width=10)
    summary = format_summary(
        batch_id="exp",
        seed=1,
        target_count=10,
        accepted_count=2,
        output_json_path=Path("output/expression_exp.json"),
        audit_jsonl_path=Path("output/expression_exp.audit.jsonl"),
        expression_counts={"smile": 1, "sad": 1},
    )

    assert " 20.0%" in line
    assert "accepted=2/10" in line
    assert "batch_id=exp" in summary
    assert "expression_distribution=sad:1, smile:1" in summary


def test_write_outputs_writes_json_and_audit(tmp_path):
    output_json = tmp_path / "expression_unit.json"
    audit_jsonl = tmp_path / "expression_unit.audit.jsonl"

    write_outputs(
        output_json_path=output_json,
        audit_jsonl_path=audit_jsonl,
        results=[{
            "cond_1": "g.jpg",
            "cond_2": "r.jpg",
            "prompt": FIXED_EXPRESSION_PROMPT,
            "width": 624,
            "height": 416,
        }],
        audit=[{"event": "pair_accepted", "score": 0.9}],
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))[0]
    assert result["prompt"] == FIXED_EXPRESSION_PROMPT
    assert result["width"] == 624
    assert json.loads(audit_jsonl.read_text(encoding="utf-8").strip())["event"] == "pair_accepted"


def test_materialize_pair_outputs_copies_pairs_as_ordered_pngs_and_writes_training_json(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    gen_path_0 = source_dir / "person0.jpg"
    ref_path_0 = source_dir / "expr0.webp"
    gen_path_1 = source_dir / "person1.png"
    ref_path_1 = source_dir / "expr1.jpg"
    Image.new("RGB", (17, 19), (255, 0, 0)).save(gen_path_0)
    Image.new("RGB", (11, 13), (0, 255, 0)).save(ref_path_0)
    Image.new("RGB", (23, 29), (0, 0, 255)).save(gen_path_1)
    Image.new("RGB", (7, 9), (255, 255, 0)).save(ref_path_1)
    materialized_dir = tmp_path / "materialized"

    output_json = materialize_pair_outputs(
        results=[
            {"cond_1": str(gen_path_0), "cond_2": str(ref_path_0), "prompt": "expression prompt 0"},
            {"cond_1": str(gen_path_1), "cond_2": str(ref_path_1), "prompt": "expression prompt 1"},
        ],
        output_root=materialized_dir,
        batch_id="unit",
    )

    copied_gen_0 = materialized_dir / "gen" / "00000.png"
    copied_ref_0 = materialized_dir / "ref" / "00000.png"
    copied_gen_1 = materialized_dir / "gen" / "00001.png"
    copied_ref_1 = materialized_dir / "ref" / "00001.png"
    assert output_json == materialized_dir / "expression_unit.json"
    assert copied_gen_0.exists()
    assert copied_ref_0.exists()
    assert copied_gen_1.exists()
    assert copied_ref_1.exists()
    assert Image.open(copied_gen_0).size == (17, 19)
    assert Image.open(copied_gen_1).size == (23, 29)
    rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert rows == [
        {
            "file_name": str(materialized_dir / "tgt" / "00000.png"),
            "cond_1": str(copied_gen_0),
            "cond_2": str(copied_ref_0),
            "prompt": "expression prompt 0",
            "width": 17,
            "height": 19,
        },
        {
            "file_name": str(materialized_dir / "tgt" / "00001.png"),
            "cond_1": str(copied_gen_1),
            "cond_2": str(copied_ref_1),
            "prompt": "expression prompt 1",
            "width": 23,
            "height": 29,
        },
    ]
