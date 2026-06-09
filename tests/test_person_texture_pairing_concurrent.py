import json
import threading
from pathlib import Path

import person_texture_pairing_concurrent as person_texture_pairing_concurrent_module
from PIL import Image
from person_texture_pairing_concurrent import (
    ImageItem,
    PairingConfig,
    build_size_buckets,
    build_texture_prompt,
    materialize_pair_outputs,
    run_pairing,
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


def test_run_pairing_uses_workers_and_keeps_dynamic_person_texture_output():
    gen_items = [
        make_item("g1", metadata=make_gen_metadata(832, 1248)),
        make_item("g2", metadata=make_gen_metadata(832, 1248)),
    ]
    ref_items = [
        make_item("r1", metadata=make_ref_metadata(832, 1248)),
        make_item("r2", metadata=make_ref_metadata(832, 1248)),
    ]
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        score_threshold=0.75,
        workers=2,
        allow_gen_reuse=False,
        max_gen_attempts_per_pass=30,
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
            if max_active_calls >= 2:
                both_running.set()

        both_running.wait(timeout=0.3)

        with lock:
            active_calls -= 1

        prompt = build_texture_prompt("black turtleneck")
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "target_garment": "black turtleneck",
            "texture_difference": "different",
            "complexity_match": "compatible",
            "prompt": prompt,
        }

    results, audit = run_pairing(
        size_buckets={"832x1248": {"gen": gen_items, "ref": ref_items}},
        config=config,
        judge_pair=fake_judge,
    )

    assert len(results) == 2
    assert results[0]["prompt"] == build_texture_prompt("black turtleneck")
    assert results[0]["width"] == 832
    assert results[0]["height"] == 1248
    assert len([row for row in audit if row["event"] == "pair_accepted"]) == 2
    assert max_active_calls >= 2


def test_concurrent_script_is_standalone():
    source = Path("person_texture_pairing_concurrent.py").read_text(encoding="utf-8")

    assert "from person_texture_pairing import" not in source
    assert "run_pairing_serial" not in source


def test_concurrent_run_pairing_limits_bad_gen_attempts_per_pass(monkeypatch):
    gen_items = [make_item("g_bad"), make_item("g_good")]
    ref_items = [make_item(f"r{i}") for i in range(10)]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        person_texture_pairing_concurrent_module,
        "shuffled_gen_pass",
        lambda gen_pool, _rng: sorted(gen_pool, key=lambda item: item.stem),
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
            "target_garment": "black turtleneck" if accepted else "",
            "texture_difference": "different" if accepted else "",
            "complexity_match": "compatible" if accepted else "",
            "prompt": build_texture_prompt("black turtleneck") if accepted else "",
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert 3 < calls_by_gen["g_bad"] <= 6
    assert calls_by_gen["g_good"] == 2
    limit_events = [row for row in audit if row["event"] == "gen_pass_attempt_limit_reached"]
    assert len(limit_events) == 1
    assert limit_events[0]["attempted_count"] == 3
    assert limit_events[0]["max_gen_attempts_per_pass"] == 3


def test_materialize_pair_outputs_copies_pairs_as_ordered_pngs_and_writes_training_json(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    gen_path = source_dir / "person.jpg"
    ref_path = source_dir / "texture.png"
    second_gen_path = source_dir / "person2.jpg"
    second_ref_path = source_dir / "texture2.jpg"
    Image.new("RGB", (17, 19), (255, 0, 0)).save(gen_path)
    Image.new("RGB", (11, 13), (0, 255, 0)).save(ref_path)
    Image.new("RGB", (23, 29), (0, 0, 255)).save(second_gen_path)
    Image.new("RGB", (31, 37), (255, 255, 0)).save(second_ref_path)
    materialized_dir = tmp_path / "materialized"

    output_json = materialize_pair_outputs(
        results=[
            {"cond_1": str(gen_path), "cond_2": str(ref_path), "prompt": "first texture"},
            {"cond_1": str(second_gen_path), "cond_2": str(second_ref_path), "prompt": "second texture"},
        ],
        output_root=materialized_dir,
        batch_id="unit",
    )

    copied_gen = materialized_dir / "gen" / "00000.png"
    copied_ref = materialized_dir / "ref" / "00000.png"
    copied_gen_2 = materialized_dir / "gen" / "00001.png"
    copied_ref_2 = materialized_dir / "ref" / "00001.png"
    assert output_json == materialized_dir / "person_texture_unit.json"
    assert copied_gen.exists()
    assert copied_ref.exists()
    assert copied_gen_2.exists()
    assert copied_ref_2.exists()
    rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert rows == [
        {
            "file_name": str(materialized_dir / "tgt" / "00000.png"),
            "cond_1": str(copied_gen),
            "cond_2": str(copied_ref),
            "prompt": "first texture",
            "width": 17,
            "height": 19,
        },
        {
            "file_name": str(materialized_dir / "tgt" / "00001.png"),
            "cond_1": str(copied_gen_2),
            "cond_2": str(copied_ref_2),
            "prompt": "second texture",
            "width": 23,
            "height": 29,
        },
    ]
