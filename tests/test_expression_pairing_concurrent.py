import json
import threading
from pathlib import Path

import expression_pairing_concurrent as expression_pairing_concurrent_module
from PIL import Image
from expression_pairing_concurrent import (
    FIXED_EXPRESSION_PROMPT,
    ImageItem,
    PairingConfig,
    build_size_buckets,
    materialize_pair_outputs,
    run_pairing,
)


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
        max_ref_attempts_per_pass=30,
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


def test_concurrent_script_is_standalone():
    source = Path("expression_pairing_concurrent.py").read_text(encoding="utf-8")

    assert "from expression_pairing import" not in source
    assert "run_pairing_serial" not in source


def test_concurrent_run_pairing_limits_bad_ref_attempts_per_pass(monkeypatch):
    gen_items = [make_item(f"g{i}") for i in range(10)]
    ref_items = [
        make_item("r_bad", metadata=make_metadata("angry")),
        make_item("r_good", metadata=make_metadata("sad")),
    ]
    buckets = build_size_buckets(gen_items, ref_items)
    monkeypatch.setattr(
        expression_pairing_concurrent_module,
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
        workers=2,
        allow_gen_reuse=True,
    )
    lock = threading.Lock()
    calls_by_ref = {"r_bad": 0, "r_good": 0}

    def fake_judge(_gen_item, ref_item):
        with lock:
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
    assert 3 < calls_by_ref["r_bad"] <= 6
    assert calls_by_ref["r_good"] == 2
    limit_events = [row for row in audit if row["event"] == "ref_pass_attempt_limit_reached"]
    assert len(limit_events) == 2
    assert {row["attempted_count"] for row in limit_events} == {3}


def test_materialize_pair_outputs_is_available_in_standalone_concurrent_script(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    gen_path_0 = source_dir / "person0.jpg"
    ref_path_0 = source_dir / "expr0.png"
    gen_path_1 = source_dir / "person1.webp"
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
    rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert output_json == materialized_dir / "expression_unit.json"
    assert copied_gen_0.exists()
    assert copied_ref_0.exists()
    assert copied_gen_1.exists()
    assert copied_ref_1.exists()
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
