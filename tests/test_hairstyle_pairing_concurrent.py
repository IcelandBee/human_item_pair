import json
import threading
from pathlib import Path

import hairstyle_pairing_concurrent as hairstyle_pairing_concurrent_module
from PIL import Image
from hairstyle_pairing_concurrent import (
    FIXED_HAIRSTYLE_PROMPT,
    ImageItem,
    PairingConfig,
    build_size_buckets,
    materialize_pair_outputs,
    run_pairing,
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


def test_run_pairing_uses_workers_for_parallel_judgements():
    gen_items = [make_item("g1"), make_item("g2")]
    ref_items = [make_item("r1"), make_item("r2")]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        score_threshold=0.75,
        workers=2,
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
            if max_active_calls >= 2:
                both_running.set()

        both_running.wait(timeout=0.3)

        with lock:
            active_calls -= 1

        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "hairstyle_difference": "different",
            "prompt": FIXED_HAIRSTYLE_PROMPT,
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert len([row for row in audit if row["event"] == "pair_accepted"]) == 2
    assert max_active_calls >= 2


def test_concurrent_hairstyle_script_is_standalone():
    source = Path("hairstyle_pairing_concurrent.py").read_text(encoding="utf-8")

    assert "from hairstyle_pairing import" not in source
    assert "run_pairing_serial" not in source


def test_concurrent_run_pairing_limits_bad_gen_attempts_per_pass(monkeypatch):
    gen_items = [make_item("g_bad"), make_item("g_good")]
    ref_items = [make_item(f"r{i}") for i in range(10)]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        hairstyle_pairing_concurrent_module,
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
            "hairstyle_difference": "different" if accepted else "too_similar",
            "prompt": FIXED_HAIRSTYLE_PROMPT if accepted else "",
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
    ref_path = source_dir / "hairstyle.png"
    Image.new("RGB", (17, 19), (255, 0, 0)).save(gen_path)
    Image.new("RGB", (11, 13), (0, 255, 0)).save(ref_path)
    materialized_dir = tmp_path / "materialized"

    output_json = materialize_pair_outputs(
        results=[{"cond_1": str(gen_path), "cond_2": str(ref_path), "prompt": "transfer hairstyle"}],
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
    assert rows[0]["prompt"] == "transfer hairstyle"
    assert rows[0]["width"] == 17
    assert rows[0]["height"] == 19
