import threading
from pathlib import Path

from person_texture_pairing import ImageItem, PairingConfig, build_texture_prompt
from person_texture_pairing_concurrent import run_pairing


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
