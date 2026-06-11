import json
import threading
import time
from pathlib import Path

from PIL import Image

from human_item_pairing_nometa import (
    ImageItem,
    PairingConfig,
    build_interaction_prompt,
    build_output_paths,
    center_crop_to_size,
    make_mock_accept_decision,
    make_numbered_png_name,
    run_pairing,
    scan_image_items,
    should_accept_decision,
    write_outputs,
)


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_item(path: Path, size: tuple[int, int] = (80, 60)) -> ImageItem:
    return ImageItem(
        stem=path.stem,
        file_name=path.name,
        image_path=path,
        width=size[0],
        height=size[1],
    )


def test_scan_image_items_reads_sorted_images_and_real_sizes(tmp_path):
    image_dir = tmp_path / "gen"
    write_image(image_dir / "b.png", (20, 10), (255, 0, 0))
    write_image(image_dir / "a.jpg", (11, 13), (0, 255, 0))
    (image_dir / "note.txt").write_text("ignore", encoding="utf-8")

    items, audit = scan_image_items(image_dir, kind="gen")

    assert [item.file_name for item in items] == ["a.jpg", "b.png"]
    assert [(item.width, item.height) for item in items] == [(11, 13), (20, 10)]
    assert audit == []


def test_center_crop_to_size_crops_larger_ref_from_center():
    ref = Image.new("RGB", (6, 4), (0, 0, 0))
    ref.putpixel((2, 1), (255, 0, 0))

    cropped = center_crop_to_size(ref, (2, 2))

    assert cropped.size == (2, 2)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)


def test_center_crop_to_size_resizes_smaller_ref_to_cover_target():
    ref = Image.new("RGB", (2, 2), (10, 20, 30))

    cropped = center_crop_to_size(ref, (6, 4))

    assert cropped.size == (6, 4)
    assert cropped.getpixel((3, 2)) == (10, 20, 30)


def test_build_output_paths_uses_explicit_paths_first(tmp_path):
    paths = build_output_paths(
        output_root=tmp_path / "root",
        batch_id="unit",
        output_gen_dir=tmp_path / "g",
        output_ref_dir=tmp_path / "r",
        output_json=tmp_path / "pairs.json",
        audit_jsonl=tmp_path / "audit.jsonl",
    )

    assert paths.gen_dir == tmp_path / "g"
    assert paths.ref_dir == tmp_path / "r"
    assert paths.output_json == tmp_path / "pairs.json"
    assert paths.audit_jsonl == tmp_path / "audit.jsonl"


def test_build_output_paths_derives_from_output_root(tmp_path):
    paths = build_output_paths(
        output_root=tmp_path / "root",
        batch_id="unit",
        output_gen_dir=None,
        output_ref_dir=None,
        output_json=None,
        audit_jsonl=None,
    )

    assert paths.gen_dir == tmp_path / "root" / "gen"
    assert paths.ref_dir == tmp_path / "root" / "ref"
    assert paths.output_json == tmp_path / "root" / "human-item_unit.json"
    assert paths.audit_jsonl == tmp_path / "root" / "human-item_unit.audit.jsonl"


def test_make_numbered_png_name_starts_at_zero_with_five_digits():
    assert make_numbered_png_name(0) == "00000.png"
    assert make_numbered_png_name(42) == "00042.png"


def test_should_accept_decision_keeps_previous_prompt_format():
    decision = {
        "suitable": True,
        "score": 0.8,
        "reason": "ok",
        "action": "hold",
        "object_description": "a clear glass jar",
        "prompt": "Let the person in image 1 hold a clear glass jar shown in image 2 in a realistic and physically coherent way, preserving object integrity and overall image consistency, while making only the minimal necessary changes and keeping everything else in image 1 unchanged.",
    }

    assert should_accept_decision(decision, score_threshold=0.75) is True
    assert should_accept_decision(decision, score_threshold=0.9) is False


def test_run_pairing_builds_prompt_from_action_and_object_description(tmp_path):
    gen_path = tmp_path / "input_gen" / "person_alpha.jpg"
    ref_path = tmp_path / "input_ref" / "object_beta.jpg"
    write_image(gen_path, (8, 6), (200, 100, 50))
    write_image(ref_path, (12, 10), (20, 30, 40))
    gen_item = make_item(gen_path, (8, 6))
    ref_item = make_item(ref_path, (12, 10))
    paths = build_output_paths(
        output_root=tmp_path / "out",
        batch_id="unit",
        output_gen_dir=None,
        output_ref_dir=None,
        output_json=None,
        audit_jsonl=None,
    )
    config = PairingConfig(
        target_count=1,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=False,
    )

    def fake_judge(_gen_item, _ref_item, _cropped_ref):
        return {
            "suitable": True,
            "score": 0.9,
            "reason": "mock accepted",
            "action": "carry over the shoulder",
            "object_description": "a red leather handbag",
            "prompt": "This prompt should not be copied directly.",
        }

    results, _audit = run_pairing(
        gen_items=[gen_item],
        ref_items=[ref_item],
        output_paths=paths,
        config=config,
        judge_pair=fake_judge,
    )

    assert results == [{
        "cond_1": str(paths.gen_dir / "00000.png"),
        "cond_2": str(paths.ref_dir / "00000.png"),
        "prompt": build_interaction_prompt(
            "carry over the shoulder",
            "a red leather handbag",
        ),
    }]


def test_build_interaction_prompt_places_object_inside_action_when_needed():
    assert build_interaction_prompt(
        "hold by the handle",
        "a white ceramic mug",
    ).startswith("Let the person in image 1 hold a white ceramic mug by the handle shown in image 2 ")

    assert build_interaction_prompt(
        "carry over the shoulder",
        "a red leather handbag",
    ).startswith("Let the person in image 1 carry a red leather handbag over the shoulder shown in image 2 ")


def test_run_pairing_writes_renamed_png_pairs_and_json(tmp_path):
    gen_path = tmp_path / "input_gen" / "person_alpha.jpg"
    ref_path = tmp_path / "input_ref" / "object_beta.jpg"
    write_image(gen_path, (8, 6), (200, 100, 50))
    write_image(ref_path, (12, 10), (20, 30, 40))
    gen_item = make_item(gen_path, (8, 6))
    ref_item = make_item(ref_path, (12, 10))
    paths = build_output_paths(
        output_root=tmp_path / "out",
        batch_id="unit",
        output_gen_dir=None,
        output_ref_dir=None,
        output_json=None,
        audit_jsonl=None,
    )
    config = PairingConfig(
        target_count=1,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        score_threshold=0.75,
        workers=1,
        allow_gen_reuse=False,
    )

    results, audit = run_pairing(
        gen_items=[gen_item],
        ref_items=[ref_item],
        output_paths=paths,
        config=config,
        judge_pair=make_mock_accept_decision,
    )
    write_outputs(paths.output_json, paths.audit_jsonl, results, audit)

    output_gen = paths.gen_dir / "00000.png"
    output_ref = paths.ref_dir / "00000.png"
    assert output_gen.exists()
    assert output_ref.exists()
    assert Image.open(output_gen).size == (8, 6)
    assert Image.open(output_ref).size == (8, 6)
    rows = json.loads(paths.output_json.read_text(encoding="utf-8"))
    assert rows == [{
        "cond_1": str(output_gen),
        "cond_2": str(output_ref),
        "prompt": results[0]["prompt"],
    }]
    accepted = [row for row in audit if row["event"] == "pair_accepted"]
    assert accepted[0]["source_gen_path"] == str(gen_path)
    assert accepted[0]["source_ref_path"] == str(ref_path)


def test_run_pairing_uses_workers_for_concurrent_judging(tmp_path):
    gen_items = []
    ref_items = []
    for index in range(4):
        gen_path = tmp_path / "input_gen" / f"person_{index}.jpg"
        ref_path = tmp_path / "input_ref" / f"object_{index}.jpg"
        write_image(gen_path, (8, 6), (200, index, 50))
        write_image(ref_path, (12, 10), (20, index, 40))
        gen_items.append(make_item(gen_path, (8, 6)))
        ref_items.append(make_item(ref_path, (12, 10)))

    paths = build_output_paths(
        output_root=tmp_path / "out",
        batch_id="unit",
        output_gen_dir=None,
        output_ref_dir=None,
        output_json=None,
        audit_jsonl=None,
    )
    config = PairingConfig(
        target_count=4,
        batch_id="unit",
        seed=123,
        max_ref_attempts_per_gen=1,
        score_threshold=0.75,
        workers=2,
        allow_gen_reuse=False,
    )
    lock = threading.Lock()
    active = 0
    max_active = 0

    def slow_judge(gen_item, ref_item, cropped_ref):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return make_mock_accept_decision(gen_item, ref_item, cropped_ref)

    results, _ = run_pairing(
        gen_items=gen_items,
        ref_items=ref_items,
        output_paths=paths,
        config=config,
        judge_pair=slow_judge,
    )

    assert len(results) == 4
    assert max_active > 1
