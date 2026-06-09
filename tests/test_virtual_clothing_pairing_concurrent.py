import json
import threading
from pathlib import Path

import virtual_clothing_pairing_concurrent as virtual_clothing_pairing_concurrent_module
from PIL import Image
from virtual_clothing_pairing_concurrent import (
    ImageItem,
    PairingConfig,
    build_size_buckets,
    materialize_pair_outputs,
    run_pairing,
)


PROMPT = (
    "Replace the black t-shirt worn by the person in image 1 with the white shirt "
    "in image 2, while making minimal changes and preserving the original pose of "
    "the person."
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


def make_decision(accepted: bool):
    return {
        "suitable": accepted,
        "score": 0.9 if accepted else 0.1,
        "reason": "mock accepted" if accepted else "mock rejected",
        "source_clothes": "black t-shirt" if accepted else "",
        "reference_clothes": "white shirt" if accepted else "",
        "prompt": PROMPT if accepted else "",
    }


def test_concurrent_script_is_standalone():
    source = Path("virtual_clothing_pairing_concurrent.py").read_text(encoding="utf-8")

    assert "from virtual_clothing_pairing import" not in source
    assert "run_pairing_serial" not in source


def test_run_pairing_uses_workers_for_parallel_virtual_clothing_judgements():
    gen_items = [make_item("g1"), make_item("g2")]
    ref_items = [make_item("r1"), make_item("r2")]
    buckets = build_size_buckets(gen_items, ref_items)
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        max_gen_attempts_per_pass=30,
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

        return make_decision(True)

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert results[0]["prompt"] == PROMPT
    assert len([row for row in audit if row["event"] == "pair_accepted"]) == 2
    assert max_active_calls >= 2


def test_run_pairing_limits_bad_gen_attempts_per_pass_and_resets_next_pass(monkeypatch):
    gen_items = [make_item("g_bad"), make_item("g_good")]
    ref_items = [make_item(f"r{i}") for i in range(10)]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        virtual_clothing_pairing_concurrent_module,
        "shuffled_gen_pass",
        lambda gen_pool, _rng: sorted(gen_pool, key=lambda item: item.stem),
    )
    config = PairingConfig(
        target_count=2,
        batch_id="unit",
        seed=0,
        max_ref_attempts_per_gen=2,
        max_gen_attempts_per_pass=3,
        score_threshold=0.75,
        workers=2,
        allow_gen_reuse=True,
    )
    lock = threading.Lock()
    calls_by_gen = {"g_bad": 0, "g_good": 0}

    def fake_judge(gen_item, _ref_item):
        with lock:
            calls_by_gen[gen_item.stem] += 1
        return make_decision(gen_item.stem == "g_good")

    results, audit = run_pairing(buckets, config, fake_judge)

    assert len(results) == 2
    assert 3 < calls_by_gen["g_bad"] <= 6
    assert calls_by_gen["g_good"] == 2
    limit_events = [row for row in audit if row["event"] == "gen_pass_attempt_limit_reached"]
    assert len(limit_events) == 1
    assert {row["attempted_count"] for row in limit_events} == {3}


def test_materialize_pair_outputs_copies_pairs_as_ordered_pngs_and_writes_training_json(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    gen_path_0 = source_dir / "person0.jpg"
    ref_path_0 = source_dir / "shirt0.png"
    gen_path_1 = source_dir / "person1.webp"
    ref_path_1 = source_dir / "shirt1.jpg"
    Image.new("RGB", (17, 19), (255, 0, 0)).save(gen_path_0)
    Image.new("RGB", (11, 13), (0, 255, 0)).save(ref_path_0)
    Image.new("RGB", (23, 29), (0, 0, 255)).save(gen_path_1)
    Image.new("RGB", (31, 37), (255, 255, 0)).save(ref_path_1)
    materialized_dir = tmp_path / "materialized"

    output_json = materialize_pair_outputs(
        results=[
            {"cond_1": str(gen_path_0), "cond_2": str(ref_path_0), "prompt": "prompt 0"},
            {"cond_1": str(gen_path_1), "cond_2": str(ref_path_1), "prompt": "prompt 1"},
        ],
        output_root=materialized_dir,
        batch_id="unit",
    )

    copied_gen_0 = materialized_dir / "gen" / "00000.png"
    copied_ref_0 = materialized_dir / "ref" / "00000.png"
    copied_gen_1 = materialized_dir / "gen" / "00001.png"
    copied_ref_1 = materialized_dir / "ref" / "00001.png"
    assert output_json == materialized_dir / "virtual-clothing_unit.json"
    assert copied_gen_0.exists()
    assert copied_ref_0.exists()
    assert copied_gen_1.exists()
    assert copied_ref_1.exists()
    rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert rows == [
        {
            "file_name": str(materialized_dir / "tgt" / "00000.png"),
            "cond_1": str(copied_gen_0),
            "cond_2": str(copied_ref_0),
            "prompt": "prompt 0",
            "width": 17,
            "height": 19,
        },
        {
            "file_name": str(materialized_dir / "tgt" / "00001.png"),
            "cond_1": str(copied_gen_1),
            "cond_2": str(copied_ref_1),
            "prompt": "prompt 1",
            "width": 23,
            "height": 29,
        },
    ]
