import threading
from pathlib import Path

from makeup_pairing import MAKEUP_PROMPT_WITH_CONTACT_LENSES, ImageItem, PairingConfig, build_size_buckets
from makeup_pairing_concurrent import run_pairing


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
