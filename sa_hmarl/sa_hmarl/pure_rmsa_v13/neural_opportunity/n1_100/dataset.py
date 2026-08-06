"""Dense exact-Teacher dataset for Neural Opportunity v1."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, RMSAAction
from sa_hmarl.pure_rmsa_v13.core import execute
from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.run_phaseA import (
    _prewarm_routes,
)
from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.selectors_phaseA import (
    OpportunityOnlyDirectSelector,
)
from sa_hmarl.pure_rmsa_v13.train_proposer import make_env

from .model import TinyOpportunityWeights
from .protocol import (
    FEATURE_DIM,
    K_PATHS,
    MAX_FEASIBLE_PATHS,
    NUM_SLOTS,
    PROTOCOL_ID,
)
from .selector import NeuralOpportunitySelector


@dataclass(frozen=True)
class CollectionResult:
    output_path: Path
    recorded: int
    skipped_blocked: int
    teacher_blocked: int


def _teacher_label_and_action(teacher, env, request, expected_batch):
    bitmap, versions, changed = teacher._detect_changed_arcs(env)
    teacher.sketch.sync_state(env, bitmap, changed)
    teacher._mark_versions_seen(versions, changed)
    candidates = teacher._scan_candidates(
        env, request, teacher.sketch.pricer._arc_words_int
    )
    teacher.sketch.ensure_paths([candidate[1] for candidate in candidates])
    if len(candidates) != len(expected_batch.candidates):
        raise AssertionError("Teacher/student feasible path count mismatch")
    targets = np.zeros((MAX_FEASIBLE_PATHS, NUM_SLOTS), dtype=np.float32)
    valid = np.zeros((MAX_FEASIBLE_PATHS, NUM_SLOTS), dtype=np.bool_)
    best_key = None
    best_action = None
    best_flat = -1
    for row, (teacher_candidate, student_candidate) in enumerate(
        zip(candidates, expected_batch.candidates)
    ):
        path_rank, path, modulation, width, starts = teacher_candidate
        if (
            path_rank != student_candidate[0]
            or path != student_candidate[1]
            or width != student_candidate[3]
        ):
            raise AssertionError("Teacher/student candidate identity mismatch")
        opportunity = teacher.sketch.score_path_starts(
            path, starts, width, normalize=True, cache_ready=True
        )
        weighted = teacher.opportunity_weight * opportunity
        targets[row, starts] = weighted.astype(np.float32)
        valid[row, starts] = True
        total = float(path_rank) + weighted
        local = int(np.argmin(total))
        start = int(starts[local])
        key = (float(total[local]), path_rank, start)
        if best_key is None or key < best_key:
            best_key = key
            best_flat = row * NUM_SLOTS + start
            best_action = RMSAAction(
                action_id=(path_rank, modulation, start, width),
                path_rank=path_rank,
                path=path,
                modulation=modulation,
                start_slot=start,
                required_fs=width,
            )
    if not np.array_equal(valid[: len(candidates)], expected_batch.valid_starts):
        raise AssertionError("Teacher/student legal-start mask mismatch")
    if best_action is None:
        best_action = BlockedAction(
            reason=teacher._blocked_reasons[(request.src_node, request.dst_node)]
        )
    return targets, valid, best_flat, best_action


def collect_teacher_dataset(
    *,
    topology: str,
    config: dict,
    seed: int,
    warmup: int,
    record_requests: int,
    output_path: Path,
    holding_truncation: float | None = None,
    path_sort_strategy: str = "hops",
) -> CollectionResult:
    env = make_env(
        seed,
        warmup + record_requests,
        topology=topology,
        num_slots=NUM_SLOTS,
        load_erlang=config["load_erlang"],
        k_paths=K_PATHS,
        holding_truncation=holding_truncation,
        path_sort_strategy=path_sort_strategy,
    )
    _prewarm_routes(env)
    teacher = OpportunityOnlyDirectSelector(
        env,
        budget=config["budget"],
        opportunity_weight=config["opportunity_weight"],
    )
    feature_selector = NeuralOpportunitySelector(
        env, TinyOpportunityWeights.zeros()
    )
    feature_rows = []
    valid_rows = []
    target_rows = []
    best_rows = []
    request_indices = []
    skipped_blocked = 0
    teacher_blocked = 0
    for request_index, request in enumerate(env.trace.requests):
        env.advance_external(request)
        batch = feature_selector.prepare_batch(env, request)
        targets, valid, best_flat, action = _teacher_label_and_action(
            teacher, env, request, batch
        )
        if request_index >= warmup:
            if batch.features.shape[0] == 0:
                skipped_blocked += 1
            else:
                padded = np.zeros(
                    (MAX_FEASIBLE_PATHS, FEATURE_DIM), dtype=np.float32
                )
                padded[: batch.features.shape[0]] = batch.features
                feature_rows.append(padded)
                valid_rows.append(valid)
                target_rows.append(targets)
                best_rows.append(best_flat)
                request_indices.append(request_index)
        result = execute(env, request, action)
        teacher_blocked += int(not result["success"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.asarray(feature_rows, dtype=np.float32),
        valid=np.asarray(valid_rows, dtype=np.bool_),
        targets=np.asarray(target_rows, dtype=np.float32),
        teacher_best=np.asarray(best_rows, dtype=np.int16),
        request_indices=np.asarray(request_indices, dtype=np.int32),
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "topology": topology,
        "seed": seed,
        "warmup": warmup,
        "record_requests": record_requests,
        "recorded": len(feature_rows),
        "skipped_blocked": skipped_blocked,
        "teacher_blocked_total": teacher_blocked,
        "feature_dim": FEATURE_DIM,
        "max_feasible_paths": MAX_FEASIBLE_PATHS,
        "num_slots": NUM_SLOTS,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return CollectionResult(
        output_path=output_path,
        recorded=len(feature_rows),
        skipped_blocked=skipped_blocked,
        teacher_blocked=teacher_blocked,
    )
