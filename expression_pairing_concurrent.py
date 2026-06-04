from __future__ import annotations

import logging
import random
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from expression_pairing import (
    FIXED_EXPRESSION_PROMPT,
    ConsoleProgress,
    ImageItem,
    JudgePair,
    PairingConfig,
    ProgressCallback,
    _all_gen_items_from_buckets,
    _all_ref_items_from_buckets,
    _configure_no_proxy,
    _validate_ratio,
    build_output_paths,
    build_size_buckets,
    build_valid_items,
    choose_next_ref,
    count_accepted_expressions,
    format_summary,
    get_expression,
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
class _RefAttempt:
    ref_item: ImageItem
    gen_item: ImageItem
    ref_expression: str
    attempt_index_for_ref: int
    attempted_gen_stems: set[str]
    remaining_gen_candidates: list[ImageItem]


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
    all_refs = _all_ref_items_from_buckets(buckets)
    refs_by_expression: dict[str, list[ImageItem]] = defaultdict(list)
    for ref in all_refs:
        refs_by_expression[get_expression(ref)].append(ref)
    ref_usage_count: dict[str, int] = {ref.stem: 0 for ref in all_refs}
    accepted_by_expression: dict[str, int] = {
        expression: 0 for expression in refs_by_expression
    }
    in_flight_by_expression: dict[str, int] = {
        expression: 0 for expression in refs_by_expression
    }
    results: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    processed_unique_gen: set[str] = set()
    reserved_gen_stems: set[str] = set()
    processed_gen = 0
    attempts = 0

    def planned_by_expression() -> dict[str, int]:
        expressions = set(accepted_by_expression) | set(in_flight_by_expression)
        return {
            expression: accepted_by_expression.get(expression, 0) + in_flight_by_expression.get(expression, 0)
            for expression in expressions
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

    def finish_ref(
        ref_item: ImageItem,
        ref_expression: str,
        attempted_gen_stems: set[str],
        accepted_for_ref: bool,
        accepted_counter: list[int],
        active_ref_stems: set[str],
        skipped_ref_stems: set[str],
    ) -> None:
        nonlocal processed_gen, attempts
        active_ref_stems.discard(ref_item.stem)
        in_flight_by_expression[ref_expression] = max(0, in_flight_by_expression.get(ref_expression, 0) - 1)
        if not accepted_for_ref:
            audit.append({
                "event": "ref_skipped_after_attempts",
                "batch_id": config.batch_id,
                "seed": config.seed,
                "size_key": ref_item.size_key,
                "ref_path": str(ref_item.image_path),
                "ref_expression": ref_expression,
                "attempted_count": len(attempted_gen_stems),
            })
            skipped_ref_stems.add(ref_item.stem)
        else:
            accepted_counter[0] += 1
        attempts += len(attempted_gen_stems)
        processed_gen += len(attempted_gen_stems)
        report_progress()

    def submit_next_gen_for_ref(
        executor: ThreadPoolExecutor,
        active: dict[Future[dict[str, Any]], _RefAttempt],
        ref_item: ImageItem,
        ref_expression: str,
        attempted_gen_stems: set[str],
        remaining_gen_candidates: list[ImageItem],
    ) -> bool:
        while remaining_gen_candidates and len(attempted_gen_stems) < config.max_ref_attempts_per_gen:
            gen_item = remaining_gen_candidates.pop(0)
            if not config.allow_gen_reuse and (
                gen_item.stem in processed_unique_gen or gen_item.stem in reserved_gen_stems
            ):
                continue
            attempted_gen_stems.add(gen_item.stem)
            if not config.allow_gen_reuse:
                reserved_gen_stems.add(gen_item.stem)
            future = executor.submit(judge_pair, gen_item, ref_item)
            active[future] = _RefAttempt(
                ref_item=ref_item,
                gen_item=gen_item,
                ref_expression=ref_expression,
                attempt_index_for_ref=len(attempted_gen_stems),
                attempted_gen_stems=attempted_gen_stems,
                remaining_gen_candidates=remaining_gen_candidates,
            )
            return True
        return False

    def start_more_refs(
        executor: ThreadPoolExecutor,
        active: dict[Future[dict[str, Any]], _RefAttempt],
        active_ref_stems: set[str],
        skipped_ref_stems: set[str],
    ) -> None:
        while len(active) < config.workers and len(results) + len(active) < config.target_count:
            blocked_ref_stems = skipped_ref_stems | active_ref_stems
            ref_item = choose_next_ref(
                refs_by_expression=refs_by_expression,
                accepted_by_expression=planned_by_expression(),
                ref_usage_count=ref_usage_count,
                blocked_ref_stems=blocked_ref_stems,
                rng=rng,
                max_smile_ratio=config.max_smile_ratio,
                max_big_laugh_ratio=config.max_big_laugh_ratio,
            )
            if ref_item is None:
                return

            ref_expression = get_expression(ref_item)
            gen_candidates = [
                gen for gen in buckets[ref_item.size_key]["gen"]
                if config.allow_gen_reuse or (
                    gen.stem not in processed_unique_gen and gen.stem not in reserved_gen_stems
                )
            ]
            if not gen_candidates:
                audit.append({
                    "event": "no_gen_candidates_left",
                    "batch_id": config.batch_id,
                    "seed": config.seed,
                    "size_key": ref_item.size_key,
                    "ref_path": str(ref_item.image_path),
                    "ref_expression": ref_expression,
                })
                skipped_ref_stems.add(ref_item.stem)
                continue

            rng.shuffle(gen_candidates)
            active_ref_stems.add(ref_item.stem)
            in_flight_by_expression[ref_expression] = in_flight_by_expression.get(ref_expression, 0) + 1
            submitted = submit_next_gen_for_ref(
                executor=executor,
                active=active,
                ref_item=ref_item,
                ref_expression=ref_expression,
                attempted_gen_stems=set(),
                remaining_gen_candidates=gen_candidates,
            )
            if not submitted:
                active_ref_stems.discard(ref_item.stem)
                in_flight_by_expression[ref_expression] = max(
                    0,
                    in_flight_by_expression.get(ref_expression, 0) - 1,
                )
                skipped_ref_stems.add(ref_item.stem)

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        while len(results) < config.target_count:
            active: dict[Future[dict[str, Any]], _RefAttempt] = {}
            active_ref_stems: set[str] = set()
            skipped_ref_stems: set[str] = set()
            accepted_in_pass = [0]
            start_more_refs(executor, active, active_ref_stems, skipped_ref_stems)

            while active and len(results) < config.target_count:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    attempt = active.pop(future)
                    gen_item = attempt.gen_item
                    ref_item = attempt.ref_item
                    ref_expression = attempt.ref_expression
                    if not config.allow_gen_reuse:
                        reserved_gen_stems.discard(gen_item.stem)
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
                            "ref_expression": ref_expression,
                            "attempt_index_for_ref": attempt.attempt_index_for_ref,
                            "error": repr(exc),
                        })
                        submitted = submit_next_gen_for_ref(
                            executor=executor,
                            active=active,
                            ref_item=ref_item,
                            ref_expression=ref_expression,
                            attempted_gen_stems=attempt.attempted_gen_stems,
                            remaining_gen_candidates=attempt.remaining_gen_candidates,
                        )
                        if not submitted:
                            finish_ref(
                                ref_item=ref_item,
                                ref_expression=ref_expression,
                                attempted_gen_stems=attempt.attempted_gen_stems,
                                accepted_for_ref=False,
                                accepted_counter=accepted_in_pass,
                                active_ref_stems=active_ref_stems,
                                skipped_ref_stems=skipped_ref_stems,
                            )
                        continue

                    audit.append({
                        "event": "pair_accepted" if accepted else "pair_rejected",
                        "batch_id": config.batch_id,
                        "seed": config.seed,
                        "size_key": gen_item.size_key,
                        "gen_path": str(gen_item.image_path),
                        "ref_path": str(ref_item.image_path),
                        "ref_expression": ref_expression,
                        "attempt_index_for_ref": attempt.attempt_index_for_ref,
                        "suitable": decision.get("suitable"),
                        "score": decision.get("score"),
                        "reason": decision.get("reason"),
                        "prompt": decision.get("prompt", ""),
                    })

                    if accepted:
                        ref_usage_count[ref_item.stem] += 1
                        accepted_by_expression[ref_expression] = accepted_by_expression.get(ref_expression, 0) + 1
                        processed_unique_gen.add(gen_item.stem)
                        result = {
                            "cond_1": str(gen_item.image_path),
                            "cond_2": str(ref_item.image_path),
                            "prompt": FIXED_EXPRESSION_PROMPT,
                        }
                        result.update(get_output_dimensions(gen_item))
                        results.append(result)
                        finish_ref(
                            ref_item=ref_item,
                            ref_expression=ref_expression,
                            attempted_gen_stems=attempt.attempted_gen_stems,
                            accepted_for_ref=True,
                            accepted_counter=accepted_in_pass,
                            active_ref_stems=active_ref_stems,
                            skipped_ref_stems=skipped_ref_stems,
                        )
                    else:
                        submitted = submit_next_gen_for_ref(
                            executor=executor,
                            active=active,
                            ref_item=ref_item,
                            ref_expression=ref_expression,
                            attempted_gen_stems=attempt.attempted_gen_stems,
                            remaining_gen_candidates=attempt.remaining_gen_candidates,
                        )
                        if not submitted:
                            finish_ref(
                                ref_item=ref_item,
                                ref_expression=ref_expression,
                                attempted_gen_stems=attempt.attempted_gen_stems,
                                accepted_for_ref=False,
                                accepted_counter=accepted_in_pass,
                                active_ref_stems=active_ref_stems,
                                skipped_ref_stems=skipped_ref_stems,
                            )

                start_more_refs(executor, active, active_ref_stems, skipped_ref_stems)

            for future in active:
                future.cancel()

            if len(results) >= config.target_count:
                break
            if accepted_in_pass[0] == 0:
                break

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
        max_smile_ratio=_validate_ratio("--max-smile-ratio", args.max_smile_ratio),
        max_big_laugh_ratio=_validate_ratio("--max-big-laugh-ratio", args.max_big_laugh_ratio),
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
        expression_counts=count_accepted_expressions(pairing_audit),
    ) + "\n")
    progress.stream.flush()


if __name__ == "__main__":
    main()
