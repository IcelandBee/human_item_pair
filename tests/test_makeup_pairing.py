import json
import random
from pathlib import Path

from PIL import Image

from makeup_pairing import (
    MAKEUP_SYSTEM_PROMPT,
    MAKEUP_PROMPT_WITH_CONTACT_LENSES,
    MAKEUP_PROMPT_WITHOUT_CONTACT_LENSES,
    ImageItem,
    PairingConfig,
    build_output_paths,
    build_size_buckets,
    build_valid_items,
    choose_balanced_ref,
    choose_next_gen,
    compact_person_metadata,
    format_summary,
    get_gender,
    get_output_dimensions,
    is_valid_person_metadata,
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

EXPECTED_MAKEUP_PROMPT = (
    "Transfer the facial makeup: including the lips color, eyeliner, eyeshadow and facial foundation "
    "from the person in image 2 to the person in image 1, keeping the rest unchanged. Ensure the makeup "
    "is natural and matches the person's facial features. Keep the eye color of the person in image 1 "
    "unchanged."
)
EXPECTED_MAKEUP_CONTACT_LENSES_PROMPT = (
    "Transfer the facial makeup: including the lips color, eyeliner, eyeshadow, colored eye contact "
    "lenses color and facial foundation from the person in image 2 to the person in image 1, keeping "
    "the rest unchanged. Ensure the makeup is natural and matches the person's facial features."
)


def make_metadata(gender: str = "female", width: int = 1248, height: int = 832):
    return {
        "resized_width": width,
        "resized_height": height,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "gender": gender,
                "head_visible": "yes",
                "facial_features_clear": "yes",
                "face_direction": "frontal",
                "obvious_makeup": "no",
                "expression": "neutral",
            }
        },
    }


def make_item(stem: str, gender: str = "female", size_key: str = "1248x832") -> ImageItem:
    return ImageItem(
        stem=stem,
        file_name=f"{stem}.jpg",
        image_path=Path(f"{stem}.jpg"),
        metadata_path=Path(f"{stem}.json"),
        size_key=size_key,
        metadata=make_metadata(gender=gender),
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


def test_valid_person_metadata_requires_clear_face():
    assert is_valid_person_metadata(make_metadata()) is True


def test_invalid_person_metadata_rejects_unclear_face():
    metadata = make_metadata()
    metadata["original_annotation"]["annotation"]["facial_features_clear"] = "no"

    assert is_valid_person_metadata(metadata) is False


def test_person_metadata_does_not_reject_gender_or_makeup_state():
    metadata = make_metadata(gender="male")
    metadata["original_annotation"]["annotation"]["obvious_makeup"] = "yes"

    assert is_valid_person_metadata(metadata) is True


def test_build_valid_items_loads_makeup_metadata(tmp_path):
    image_dir = tmp_path / "gen"
    metadata_dir = tmp_path / "gen_metadata"
    image_dir.mkdir()
    metadata_dir.mkdir()
    (image_dir / "000001.jpg").write_bytes(b"fake")
    (metadata_dir / "000001.json").write_text(
        json.dumps({
            "file_name": "000001.jpg",
            **make_metadata(),
        }),
        encoding="utf-8",
    )

    items, audit = build_valid_items(image_dir, metadata_dir, "gen")

    assert len(items) == 1
    assert items[0].size_key == "1248x832"
    assert audit == []


def test_build_size_buckets_keeps_matching_dimensions():
    gen_items = [make_item("g1", "female", "1248x832"), make_item("g2", "male", "832x1248")]
    ref_items = [make_item("r1", "female", "1248x832")]

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


def test_get_gender_reads_annotation():
    assert get_gender(make_item("m", "male")) == "male"
    assert get_gender(make_item("f", "female")) == "female"


def test_choose_next_gen_prefers_gender_needed_for_three_to_seven_ratio():
    male = [make_item("m1", "male")]
    female = [make_item("f1", "female"), make_item("f2", "female")]
    queues = {"male": male.copy(), "female": female.copy(), "unknown": []}
    accepted_gender_counts = {"male": 3, "female": 2, "unknown": 0}

    selected = choose_next_gen(
        gen_queues=queues,
        accepted_gender_counts=accepted_gender_counts,
        rng=random.Random(1),
    )

    assert selected is not None
    assert get_gender(selected) == "female"


def test_choose_next_gen_approaches_three_to_seven_ratio_over_batch():
    queues = {
        "male": [make_item(f"m{i}", "male") for i in range(3)],
        "female": [make_item(f"f{i}", "female") for i in range(7)],
        "unknown": [],
    }
    counts = {"male": 0, "female": 0, "unknown": 0}
    selected_genders = []

    for _ in range(10):
        item = choose_next_gen(queues, counts, random.Random(1))
        assert item is not None
        gender = get_gender(item)
        counts[gender] += 1
        selected_genders.append(gender)

    assert selected_genders.count("male") == 3
    assert selected_genders.count("female") == 7


def test_compact_person_metadata_keeps_makeup_fields():
    compact = compact_person_metadata(make_metadata(gender="female"))

    assert compact["gender"] == "female"
    assert compact["facial_features_clear"] == "yes"
    assert compact["face_direction"] == "frontal"
    assert compact["obvious_makeup"] == "no"


def test_makeup_prompt_contains_confirmed_judgement_scope():
    assert MAKEUP_PROMPT_WITHOUT_CONTACT_LENSES == EXPECTED_MAKEUP_PROMPT
    assert MAKEUP_PROMPT_WITH_CONTACT_LENSES == EXPECTED_MAKEUP_CONTACT_LENSES_PROMPT
    prompt = MAKEUP_SYSTEM_PROMPT.lower()

    assert "key makeup regions" in prompt
    assert "reference makeup" in prompt
    assert "face makeup regions" in prompt
    assert "masks, sunglasses, hands, hair, props" in prompt
    assert "strong shadow, blur, or overexposure" in prompt
    assert "eye color of the person in image 1 unchanged" in prompt
    assert "colored eye contact lenses color" in prompt
    assert "iris_color_difference" in prompt
    assert "at least one fully open eye" in prompt
    assert "iris color can be reliably judged" in prompt
    assert "squinting or half-closed" in prompt
    assert "set that eye clarity field to false" in prompt
    assert "extra strict rule for gen_eyes_clear" in prompt
    assert "be stricter for image 1 than image 2" in prompt
    assert "squinting, has narrow eyes" in prompt
    assert "when uncertain, set gen_eyes_clear to false" in prompt


def test_parse_and_accept_makeup_prompt_without_contact_lenses():
    raw = f"""```json
{{
  "suitable": true,
  "score": 0.86,
  "reason": "Both faces are clear and makeup regions are visible.",
  "gen_eyes_clear": false,
  "ref_eyes_clear": true,
  "iris_color_difference": "unclear",
  "prompt": "{MAKEUP_PROMPT_WITHOUT_CONTACT_LENSES}"
}}
```"""

    decision = parse_vlm_decision(raw)

    assert should_accept_decision(decision, 0.75) is True


def test_accepts_makeup_prompt_with_contact_lenses_when_iris_colors_differ():
    decision = {
        "suitable": True,
        "score": 0.92,
        "reason": "Both eyes are clear and iris colors differ.",
        "gen_eyes_clear": True,
        "ref_eyes_clear": True,
        "iris_color_difference": "different",
        "prompt": MAKEUP_PROMPT_WITH_CONTACT_LENSES,
    }

    assert should_accept_decision(decision, 0.75) is True


def test_rejects_contact_lenses_prompt_when_eye_decision_is_inconsistent():
    decision = {
        "suitable": True,
        "score": 0.92,
        "reason": "Prompt conflicts with eye clarity judgement.",
        "gen_eyes_clear": False,
        "ref_eyes_clear": True,
        "iris_color_difference": "different",
        "prompt": MAKEUP_PROMPT_WITH_CONTACT_LENSES,
    }

    assert should_accept_decision(decision, 0.75) is False


def test_rejects_non_makeup_prompt_format():
    decision = {
        "suitable": True,
        "score": 0.95,
        "prompt": "Transfer the facial expression from image 2 to image 1.",
    }

    assert should_accept_decision(decision, 0.75) is False


def test_get_output_dimensions_halves_resized_dimensions():
    assert get_output_dimensions(make_item("g")) == {"width": 624, "height": 416}


def test_run_pairing_outputs_fixed_prompt_dimensions_and_gender_ratio():
    gen_items = [make_item(f"m{i}", "male") for i in range(3)] + [make_item(f"f{i}", "female") for i in range(7)]
    ref_items = [make_item("r1", "female")]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=10,
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
            "gen_eyes_clear": True,
            "ref_eyes_clear": True,
            "iris_color_difference": "different",
            "prompt": MAKEUP_PROMPT_WITH_CONTACT_LENSES,
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 10
    assert results[0]["prompt"] == MAKEUP_PROMPT_WITH_CONTACT_LENSES
    assert results[0]["width"] == 624
    assert results[0]["height"] == 416
    assert len([row for row in audit if row["event"] == "pair_accepted" and row["gen_gender"] == "male"]) == 3
    assert len([row for row in audit if row["event"] == "pair_accepted" and row["gen_gender"] == "female"]) == 7


def test_run_pairing_limits_bad_gen_attempts_per_pass_and_resets_next_pass(monkeypatch):
    gen_items = [make_item("g_bad"), make_item("g_good")]
    ref_items = [make_item(f"r{i}") for i in range(10)]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        "makeup_pairing._gender_balanced_gen_pass",
        lambda gen_pool, _counts, _rng: sorted(gen_pool, key=lambda item: item.stem),
    )
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=0,
        max_ref_attempts_per_gen=2,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=True,
        max_gen_attempts_per_pass=3,
    )
    calls_by_gen = {"g_bad": 0, "g_good": 0}

    def fake_judge(gen_item, _ref_item):
        calls_by_gen[gen_item.stem] += 1
        accepted = gen_item.stem == "g_good"
        return {
            "suitable": accepted,
            "score": 0.9 if accepted else 0.1,
            "reason": "mock accepted" if accepted else "mock rejected",
            "gen_eyes_clear": True,
            "ref_eyes_clear": True,
            "iris_color_difference": "different",
            "prompt": MAKEUP_PROMPT_WITH_CONTACT_LENSES if accepted else "",
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert calls_by_gen == {"g_bad": 5, "g_good": 2}
    limit_events = [row for row in audit if row["event"] == "gen_pass_attempt_limit_reached"]
    assert len(limit_events) == 1
    assert limit_events[0]["attempted_count"] == 3
    assert limit_events[0]["max_gen_attempts_per_pass"] == 3


def test_run_pairing_does_not_cap_successful_gen_reuse_across_passes():
    gen_items = [make_item("g_good")]
    ref_items = [make_item("r1")]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=4,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=True,
        max_gen_attempts_per_pass=1,
    )
    calls = 0

    def fake_judge(_gen_item, _ref_item):
        nonlocal calls
        calls += 1
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "gen_eyes_clear": True,
            "ref_eyes_clear": True,
            "iris_color_difference": "different",
            "prompt": MAKEUP_PROMPT_WITH_CONTACT_LENSES,
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 4
    assert calls == 4
    assert not [row for row in audit if row["event"] == "gen_pass_attempt_limit_reached"]


def test_run_pairing_allows_only_one_pair_per_gen_per_pass(monkeypatch):
    gen_items = [make_item("g1")]
    ref_items = [make_item("r1"), make_item("r2")]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        "makeup_pairing._gender_balanced_gen_pass",
        lambda gen_pool, _counts, _rng: list(gen_pool),
    )
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=2,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=True,
        max_gen_attempts_per_pass=2,
    )
    pass_ref_sequence = []

    def fake_judge(_gen_item, ref_item):
        pass_ref_sequence.append(ref_item.stem)
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "gen_eyes_clear": True,
            "ref_eyes_clear": True,
            "iris_color_difference": "different",
            "prompt": MAKEUP_PROMPT_WITH_CONTACT_LENSES,
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert len(pass_ref_sequence) == 2
    assert len([row for row in audit if row["event"] == "pair_accepted"]) == 2


def test_output_paths_use_makeup_prefix():
    output_json, audit_jsonl = build_output_paths(Path("output"), "exp_v1")

    assert output_json == Path("output") / "makeup_exp_v1.json"
    assert audit_jsonl == Path("output") / "makeup_exp_v1.audit.jsonl"


def test_mock_accept_decision_returns_fixed_prompt():
    decision = make_mock_accept_decision(make_item("g"), make_item("r"))

    assert decision["prompt"] == MAKEUP_PROMPT_WITHOUT_CONTACT_LENSES
    assert should_accept_decision(decision, 0.75) is True


def test_progress_and_summary_helpers():
    line = render_progress_bar("pairing", 2, 10, 4, 7, width=10)
    summary = format_summary(
        batch_id="exp",
        seed=1,
        target_count=10,
        accepted_count=2,
        output_json_path=Path("output/makeup_exp.json"),
        audit_jsonl_path=Path("output/makeup_exp.audit.jsonl"),
    )

    assert " 20.0%" in line
    assert "accepted=2/10" in line
    assert "batch_id=exp" in summary


def test_write_outputs_writes_json_and_audit(tmp_path):
    output_json = tmp_path / "makeup_unit.json"
    audit_jsonl = tmp_path / "makeup_unit.audit.jsonl"

    write_outputs(
        output_json_path=output_json,
        audit_jsonl_path=audit_jsonl,
        results=[{
            "cond_1": "g.jpg",
            "cond_2": "r.jpg",
            "prompt": MAKEUP_PROMPT_WITHOUT_CONTACT_LENSES,
            "width": 624,
            "height": 416,
        }],
        audit=[{"event": "pair_accepted", "score": 0.9}],
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))[0]
    assert result["prompt"] == MAKEUP_PROMPT_WITHOUT_CONTACT_LENSES
    assert result["width"] == 624
    assert json.loads(audit_jsonl.read_text(encoding="utf-8").strip())["event"] == "pair_accepted"


def test_materialize_pair_outputs_copies_pairs_as_ordered_pngs_and_writes_training_json(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    gen_0 = source_dir / "gen0.jpg"
    ref_0 = source_dir / "ref0.webp"
    gen_1 = source_dir / "gen1.png"
    ref_1 = source_dir / "ref1.jpg"
    Image.new("RGB", (17, 19), (255, 0, 0)).save(gen_0)
    Image.new("RGB", (11, 13), (0, 255, 0)).save(ref_0)
    Image.new("RGB", (23, 29), (0, 0, 255)).save(gen_1)
    Image.new("RGB", (31, 37), (255, 255, 0)).save(ref_1)
    materialized_dir = tmp_path / "materialized"

    output_json = materialize_pair_outputs(
        results=[
            {"cond_1": str(gen_0), "cond_2": str(ref_0), "prompt": "makeup 0"},
            {"cond_1": str(gen_1), "cond_2": str(ref_1), "prompt": "makeup 1"},
        ],
        output_root=materialized_dir,
        batch_id="unit",
    )

    copied_gen_0 = materialized_dir / "gen" / "00000.png"
    copied_ref_0 = materialized_dir / "ref" / "00000.png"
    copied_gen_1 = materialized_dir / "gen" / "00001.png"
    copied_ref_1 = materialized_dir / "ref" / "00001.png"
    assert output_json == materialized_dir / "makeup_unit.json"
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
            "prompt": "makeup 0",
            "width": 17,
            "height": 19,
        },
        {
            "file_name": str(materialized_dir / "tgt" / "00001.png"),
            "cond_1": str(copied_gen_1),
            "cond_2": str(copied_ref_1),
            "prompt": "makeup 1",
            "width": 23,
            "height": 29,
        },
    ]
