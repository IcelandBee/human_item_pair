import json
import threading
from pathlib import Path

import makeup_pairing_concurrent as makeup_pairing_concurrent_module
from PIL import Image
from makeup_pairing_concurrent import (
    MAKEUP_PROMPT_WITH_CONTACT_LENSES,
    ImageItem,
    PairingConfig,
    build_size_buckets,
    materialize_pair_outputs,
    run_pairing,
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


def test_run_pairing_keeps_workers_busy_and_stays_near_makeup_gender_ratio():
    gen_items = [make_item(f"m{i}", "male") for i in range(3)] + [
        make_item(f"f{i}", "female") for i in range(7)
    ]
    ref_items = [make_item(f"r{i}", "female") for i in range(4)]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=10,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        score_threshold=0.75,
        workers=4,
        allow_gen_reuse=False,
    )
    lock = threading.Lock()
    both_running = threading.Event()
    active_calls = 0
    max_active_calls = 0

    def fake_judge(_gen_item, _ref_item):
        nonlocal active_calls, max_active_calls
        with lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            if max_active_calls >= config.workers:
                both_running.set()

        both_running.wait(timeout=0.5)

        with lock:
            active_calls -= 1

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
    assert max_active_calls >= config.workers


def test_concurrent_makeup_script_is_standalone():
    source = Path("makeup_pairing_concurrent.py").read_text(encoding="utf-8")

    assert "from makeup_pairing import" not in source
    assert "run_pairing_serial" not in source


def test_concurrent_run_pairing_limits_bad_gen_attempts_per_pass(monkeypatch):
    gen_items = [make_item("g_bad"), make_item("g_good")]
    ref_items = [make_item(f"r{i}") for i in range(10)]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        makeup_pairing_concurrent_module,
        "_gender_balanced_gen_pass",
        lambda gen_pool, _counts, _rng: sorted(gen_pool, key=lambda item: item.stem),
    )
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=0,
        max_ref_attempts_per_gen=2,
        score_threshold=0.75,
        workers=2,
        allow_gen_reuse=True,
        max_gen_attempts_per_pass=3,
    )
    lock = threading.Lock()
    calls_by_gen = {"g_bad": 0, "g_good": 0}

    def fake_judge(gen_item, _ref_item):
        with lock:
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
    assert 3 < calls_by_gen["g_bad"] <= 6
    assert calls_by_gen["g_good"] == 2
    limit_events = [row for row in audit if row["event"] == "gen_pass_attempt_limit_reached"]
    assert len(limit_events) == 1
    assert {row["attempted_count"] for row in limit_events} == {3}


def test_materialize_pair_outputs_is_available_in_standalone_concurrent_script(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    gen_path = source_dir / "person.jpg"
    ref_path = source_dir / "makeup.png"
    Image.new("RGB", (17, 19), (255, 0, 0)).save(gen_path)
    Image.new("RGB", (11, 13), (0, 255, 0)).save(ref_path)
    materialized_dir = tmp_path / "materialized"

    output_json = materialize_pair_outputs(
        results=[{"cond_1": str(gen_path), "cond_2": str(ref_path), "prompt": "transfer makeup"}],
        output_root=materialized_dir,
        batch_id="unit",
    )

    copied_gen = materialized_dir / "gen" / "00000.png"
    copied_ref = materialized_dir / "ref" / "00000.png"
    rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert output_json == materialized_dir / "makeup_unit.json"
    assert copied_gen.exists()
    assert copied_ref.exists()
    assert rows[0]["file_name"] == str(materialized_dir / "tgt" / "00000.png")
    assert rows[0]["cond_1"] == str(copied_gen)
    assert rows[0]["cond_2"] == str(copied_ref)
    assert rows[0]["prompt"] == "transfer makeup"
    assert rows[0]["width"] == 17
    assert rows[0]["height"] == 19
