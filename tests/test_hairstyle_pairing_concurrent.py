import threading
from pathlib import Path

from hairstyle_pairing import FIXED_HAIRSTYLE_PROMPT, ImageItem, PairingConfig, build_size_buckets
from hairstyle_pairing_concurrent import run_pairing


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
