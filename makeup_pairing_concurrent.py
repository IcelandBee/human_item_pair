from __future__ import annotations

import logging
import random
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from makeup_pairing import (
    FIXED_MAKEUP_PROMPT,
    ConsoleProgress,
    ImageItem,
    JudgePair,
    PairingConfig,
    ProgressCallback,
    _all_gen_items_from_buckets,
    _build_gen_queues,
    _configure_no_proxy,
    build_output_paths,
    build_size_buckets,
    build_valid_items,
    choose_balanced_ref,
    choose_next_gen,
    format_summary,
    get_gender,
    get_output_dimensions,
    infer_pair_decision,
    make_batch_id,
    make_mock_accept_decision,
    make_seed,
    parse_args,
    run_pairing as run_pairing_serial,
    should_accept_decision,
    write_outputs,
)


@dataclass
class _PairAttempt:
    gen_item: ImageItem
    ref_item: ImageItem
    attempt_index_for_gen: int
    attempted_ref_stems: set[str]


def run_pairing(
    buckets: dict[str, dict[str, list[ImageItem]]],
    config: PairingConfig,
    judge_pair: JudgePair,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if config.workers <= 1:
        return run_pairing_serial(
            buckets=buckets,
            config=config,
            judge_pair=judge_pair,
            progress_callback=progress_callback,
        )

    rng = random.Random(config.seed)
    ref_usage_by_size: dict[str, dict[str, int]] = {
        size_key: {ref.stem: 0 for ref in bucket["ref"]}
        for size_key, bucket in buckets.items()
    }
    results: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    gen_pool = _all_gen_items_from_buckets(buckets)
    gen_queues = _build_gen_queues(gen_pool, rng)
    accepted_gender_counts = {"male": 0, "female": 0, "unknown": 0}
    in_flight_gender_counts = {"male": 0, "female": 0, "unknown": 0}
    processed_gen = 0
    attempts = 0

    def planned_gender_counts() -> dict[str, int]:
        return {
            gender: accepted_gender_counts.get(gender, 0) + in_flight_gender_counts.get(gender, 0)
            for gender in ("male", "female", "unknown")
        }

    def report_progress() -> None:
        if progress_callback is None:
            return
        progress_callback({
            "accepted": len(results),
            "target": config.target_count,
            "processed_gen": processed_gen,
            "attempts": attempts,
        })

    def finish_gen(gen_item: ImageItem, attempted_ref_stems: set[str], accepted_for_gen: bool) -> None:
        nonlocal processed_gen, attempts
        gen_gender = get_gender(gen_item)
        in_flight_gender_counts[gen_gender] -= 1
        if not accepted_for_gen:
            audit.append({
                "event": "gen_skipped_after_attempts",
                "batch_id": config.batch_id,
                "seed": config.seed,
                "size_key": gen_item.size_key,
                "gen_path": str(gen_item.image_path),
                "gen_gender": gen_gender,
                "attempted_count": len(attempted_ref_stems),
            })
        processed_gen += 1
        attempts += len(attempted_ref_stems)
        report_progress()

    def submit_attempt(
        executor: ThreadPoolExecutor,
        active: dict[Future[dict[str, Any]], _PairAttempt],
        gen_item: ImageItem,
        attempted_ref_stems: set[str],
        attempt_index_for_gen: int,
    ) -> bool:
        refs = buckets[gen_item.size_key]["ref"]
        ref_item = choose_balanced_ref(
            refs=refs,
            ref_usage_count=ref_usage_by_size[gen_item.size_key],
            attempted_ref_stems=attempted_ref_stems,
            rng=rng,
        )
        if ref_item is None:
            audit.append({
                "event": "no_ref_candidates_left",
                "batch_id": config.batch_id,
                "seed": config.seed,
                "size_key": gen_item.size_key,
                "gen_path": str(gen_item.image_path),
            })
            finish_gen(gen_item, attempted_ref_stems, accepted_for_gen=False)
            return False

        attempted_ref_stems.add(ref_item.stem)
        future = executor.submit(judge_pair, gen_item, ref_item)
        active[future] = _PairAttempt(
            gen_item=gen_item,
            ref_item=ref_item,
            attempt_index_for_gen=attempt_index_for_gen,
            attempted_ref_stems=attempted_ref_stems,
        )
        return True

    def start_more_gens(
        executor: ThreadPoolExecutor,
        active: dict[Future[dict[str, Any]], _PairAttempt],
    ) -> None:
        while len(active) < config.workers and len(results) + len(active) < config.target_count:
            gen_item = choose_next_gen(gen_queues, planned_gender_counts(), rng)
            if gen_item is None:
                return
            in_flight_gender_counts[get_gender(gen_item)] += 1
            submit_attempt(
                executor=executor,
                active=active,
                gen_item=gen_item,
                attempted_ref_stems=set(),
                attempt_index_for_gen=1,
            )

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        active: dict[Future[dict[str, Any]], _PairAttempt] = {}
        start_more_gens(executor, active)

        while active and len(results) < config.target_count:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                attempt = active.pop(future)
                gen_item = attempt.gen_item
                ref_item = attempt.ref_item
                gen_gender = get_gender(gen_item)
                accepted = False

                try:
                    decision = future.result()
                    accepted = should_accept_decision(decision, config.score_threshold)
                except Exception as exc:
                    audit.append({
                        "event": "pair_error",
                        "batch_id": config.batch_id,
                        "seed": config.seed,
                        "size_key": gen_item.size_key,
                        "gen_path": str(gen_item.image_path),
                        "ref_path": str(ref_item.image_path),
                        "attempt_index_for_gen": attempt.attempt_index_for_gen,
                        "gen_gender": gen_gender,
                        "error": repr(exc),
                    })
                    if attempt.attempt_index_for_gen < config.max_ref_attempts_per_gen:
                        submit_attempt(
                            executor=executor,
                            active=active,
                            gen_item=gen_item,
                            attempted_ref_stems=attempt.attempted_ref_stems,
                            attempt_index_for_gen=attempt.attempt_index_for_gen + 1,
                        )
                    else:
                        finish_gen(gen_item, attempt.attempted_ref_stems, accepted_for_gen=False)
                    continue

                audit.append({
                    "event": "pair_accepted" if accepted else "pair_rejected",
                    "batch_id": config.batch_id,
                    "seed": config.seed,
                    "size_key": gen_item.size_key,
                    "gen_path": str(gen_item.image_path),
                    "ref_path": str(ref_item.image_path),
                    "attempt_index_for_gen": attempt.attempt_index_for_gen,
                    "gen_gender": gen_gender,
                    "suitable": decision.get("suitable"),
                    "score": decision.get("score"),
                    "reason": decision.get("reason"),
                    "prompt": decision.get("prompt", ""),
                })

                if accepted:
                    accepted_gender_counts[gen_gender] += 1
                    ref_usage_by_size[gen_item.size_key][ref_item.stem] += 1
                    result = {
                        "cond_1": str(gen_item.image_path),
                        "cond_2": str(ref_item.image_path),
                        "prompt": FIXED_MAKEUP_PROMPT,
                    }
                    result.update(get_output_dimensions(gen_item))
                    results.append(result)
                    finish_gen(gen_item, attempt.attempted_ref_stems, accepted_for_gen=True)
                elif attempt.attempt_index_for_gen < config.max_ref_attempts_per_gen:
                    submit_attempt(
                        executor=executor,
                        active=active,
                        gen_item=gen_item,
                        attempted_ref_stems=attempt.attempted_ref_stems,
                        attempt_index_for_gen=attempt.attempt_index_for_gen + 1,
                    )
                else:
                    finish_gen(gen_item, attempt.attempted_ref_stems, accepted_for_gen=False)

            start_more_gens(executor, active)

        for future in active:
            future.cancel()

    return results[:config.target_count], audit


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    batch_id = make_batch_id(args.batch_id)
    seed = make_seed(args.seed)
    _configure_no_proxy(args.base_url)
    progress = ConsoleProgress()

    config = PairingConfig(
        target_count=args.target_count,
        batch_id=batch_id,
        seed=seed,
        max_ref_attempts_per_gen=args.max_ref_attempts_per_gen,
        score_threshold=args.score_threshold,
        workers=args.workers,
        allow_gen_reuse=args.allow_gen_reuse,
    )

    progress.stage("prepare-data")
    gen_items, gen_audit = build_valid_items(args.gen_dir, args.gen_metadata_dir, "gen")
    ref_items, ref_audit = build_valid_items(args.ref_dir, args.ref_metadata_dir, "ref")

    progress.stage("build-size-buckets")
    buckets = build_size_buckets(gen_items, ref_items)

    progress.stage("pairing")
    if args.dry_run_accept_all:
        judge_pair = make_mock_accept_decision
    else:
        if not args.base_url or not args.api_key or not args.model_name:
            raise ValueError("--base-url, --api-key and --model-name are required outside dry-run mode")
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)

        def judge_pair(gen_item: ImageItem, ref_item: ImageItem) -> dict[str, Any]:
            return infer_pair_decision(
                client=client,
                model_name=args.model_name or "",
                gen_item=gen_item,
                ref_item=ref_item,
                max_retries=args.max_retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

    results, pairing_audit = run_pairing(
        buckets=buckets,
        config=config,
        judge_pair=judge_pair,
        progress_callback=progress.pairing,
    )
    progress.finish_pairing_line()
    audit = gen_audit + ref_audit + pairing_audit

    progress.stage("write-output")
    output_json_path, audit_jsonl_path = build_output_paths(args.output_dir, batch_id)
    write_outputs(output_json_path, audit_jsonl_path, results, audit)

    progress.stage("done")
    progress.stream.write(format_summary(
        batch_id=batch_id,
        seed=seed,
        target_count=args.target_count,
        accepted_count=len(results),
        output_json_path=output_json_path,
        audit_jsonl_path=audit_jsonl_path,
    ) + "\n")
    progress.stream.flush()


if __name__ == "__main__":
    main()
