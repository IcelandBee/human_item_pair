import threading
from pathlib import Path

from expression_pairing import FIXED_EXPRESSION_PROMPT, ImageItem, PairingConfig, build_size_buckets
from expression_pairing_concurrent import run_pairing


def make_metadata(expression: str = "neutral", width: int = 1248, height: int = 832):
    return {
        "resized_width": width,
        "resized_height": height,
        "original_annotation": {
            "annotation": {
                "person_count": "1",
                "head_visible": "yes",
                "facial_features_clear": "yes",
                "face_direction": "frontal",
                "expression": expression,
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


def test_run_pairing_keeps_workers_busy_and_respects_expression_soft_caps():
    gen_items = [make_item(f"g{i}", metadata=make_metadata("neutral")) for i in range(8)]
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
        score_threshold=0.75,
        workers=4,
        allow_gen_reuse=False,
        max_smile_ratio=0.2,
        max_big_laugh_ratio=0.2,
    )
    lock = threading.Lock()
    workers_running = threading.Event()
    active_calls = 0
    max_active_calls = 0

    def fake_judge(_gen_item, _ref_item):
        nonlocal active_calls, max_active_calls
        with lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            if max_active_calls >= config.workers:
                workers_running.set()

        workers_running.wait(timeout=0.5)

        with lock:
            active_calls -= 1

        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "prompt": FIXED_EXPRESSION_PROMPT,
        }

    results, audit = run_pairing(buckets, config, fake_judge)

    accepted_rows = [row for row in audit if row["event"] == "pair_accepted"]
    accepted_expressions = [row["ref_expression"] for row in accepted_rows]

    assert len(results) == 5
    assert results[0]["prompt"] == FIXED_EXPRESSION_PROMPT
    assert results[0]["width"] == 624
    assert results[0]["height"] == 416
    assert len(accepted_rows) == 5
    assert accepted_expressions.count("smile") <= 1
    assert accepted_expressions.count("big_laugh") <= 1
    assert sum(expr not in {"smile", "big_laugh"} for expr in accepted_expressions) >= 3
    assert max_active_calls >= config.workers
