"""Phase 1 PDS-MLP vs PreD-DQN trainer for COST239 load 300."""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.pds_rmsa.baselines.ksp_ff import ksp_ff_action
from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, PDSRMSAEnv, RMSAAction
from sa_hmarl.pds_rmsa.env.topology import load_topology
from sa_hmarl.pds_rmsa.env.traffic_trace import generate_trace
from sa_hmarl.pds_rmsa.features.afterstate import (
    build_probe_suite,
    build_schema,
    probe_suite_sha256,
)
from sa_hmarl.pds_rmsa.networks.mlp import count_parameters
from sa_hmarl.pds_rmsa.protocol import K_PATHS, MEAN_HOLDING_TIME
from sa_hmarl.pds_rmsa.training.agents import PDSAgent, PreDAgent, linear_epsilon
from sa_hmarl.pds_rmsa.training.replay import ReplayBuffer


TOPOLOGY = "xlron_cost239_ptrnet_real"
NUM_NODES = 11
NUM_SLOTS = 50
LOAD_ERLANG = 300.0
GAMMA = 0.99
PATH_SORT_STRATEGY = "hops"
PHASE1_PROTOCOL_ID = (
    "pds_rmsa_phase1_cost239_load300_k5_hops_b64_u4_continuing_v3"
)
CHECKPOINT_BUDGETS = (0, 25000, 50000, 100000, 200000, 300000)


def _make_feature_config(num_slots: int) -> dict:
    """Build the feature configuration used by both PDS and PreD agents."""
    probe_suite = build_probe_suite(
        num_nodes=NUM_NODES,
        topology_name=TOPOLOGY,
        num_od_pairs=20,
        bandwidths=(25, 50, 75, 100),
        k_paths=K_PATHS,
        path_sort_strategy=PATH_SORT_STRATEGY,
    )
    feature_config = {
        "num_slots": num_slots,
        "free_block_bins": [1, 2, 4, 8, 16, 32, np.inf],
        "residual_bucket_edges": [0, 2, 5, 10, 20, float("inf")],
        "probe_suite": probe_suite,
        "topology_name": TOPOLOGY,
        "num_nodes": NUM_NODES,
        "k_paths": K_PATHS,
        "path_sort_strategy": PATH_SORT_STRATEGY,
        "max_hops": NUM_NODES - 1,
        "max_path_km": 10000.0,
        "max_required_fs": 10.0,
        "context_dim": 40,
        "mean_holding_time": MEAN_HOLDING_TIME,
    }
    return feature_config


def _compute_input_dim(feature_config: dict) -> int:
    schema = build_schema(
        TOPOLOGY,
        feature_config,
        probe_count=len(feature_config["probe_suite"].probes),
    )
    return len(schema.feature_names) + int(feature_config["context_dim"])


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` using a temp file + atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp")
    try:
        tmp.write(data)
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise


def _feature_schema_sha256(feature_config: dict) -> str:
    schema = build_schema(
        TOPOLOGY,
        feature_config,
        probe_count=len(feature_config["probe_suite"].probes),
    )
    payload = json.dumps(schema.to_dict(), sort_keys=True, default=str)
    import hashlib
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_protocol(
    agent,
    config: Dict[str, Any],
    protocol_id: str = PHASE1_PROTOCOL_ID,
) -> Dict[str, Any]:
    return {
        "protocol_id": protocol_id,
        "topology": TOPOLOGY,
        "load_erlang": LOAD_ERLANG,
        "num_slots": NUM_SLOTS,
        "k_paths": int(agent.k_paths),
        "path_sort_strategy": str(agent.path_sort_strategy),
        "batch_size": int(config.get("batch_size", 64)),
        "update_every": int(config.get("update_every", 4)),
        "input_dim": int(agent.input_dim),
        "feature_schema_sha256": _feature_schema_sha256(agent.feature_config),
        "probe_suite_sha256": probe_suite_sha256(agent.feature_config["probe_suite"]),
    }


def _validate_checkpoint_protocol(
    payload: Dict[str, Any],
    agent,
    config: Dict[str, Any],
    protocol_id: str = PHASE1_PROTOCOL_ID,
) -> None:
    actual = payload.get("protocol")
    expected = _checkpoint_protocol(agent, config, protocol_id=protocol_id)
    if actual is None:
        raise ValueError("Checkpoint has no protocol metadata; refusing legacy/incompatible checkpoint")
    mismatches = {
        key: {"checkpoint": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint protocol mismatch: {mismatches}")


def _save_agent_checkpoint(
    path: Path,
    agent,
    step: int,
    config: Dict[str, Any],
    protocol_id: str = PHASE1_PROTOCOL_ID,
) -> None:
    payload = {
        "step": step,
        "online_decisions": step,
        "epsilon": agent.epsilon,
        "protocol": _checkpoint_protocol(agent, config, protocol_id=protocol_id),
        "online_net": agent.online_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp")
    try:
        torch.save(payload, tmp.name)
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise


def _load_agent_checkpoint(
    path: Path,
    agent,
    config: Dict[str, Any],
    protocol_id: str = PHASE1_PROTOCOL_ID,
) -> None:
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    _validate_checkpoint_protocol(payload, agent, config, protocol_id=protocol_id)
    agent.online_net.load_state_dict(payload["online_net"])
    agent.target_net.load_state_dict(payload["target_net"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    agent.epsilon = float(payload["epsilon"])


def pickle_bytes(obj: Any) -> bytes:
    import pickle
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _make_env(seed: int, trace_length: int) -> PDSRMSAEnv:
    trace = generate_trace(
        TOPOLOGY,
        NUM_NODES,
        trace_length,
        LOAD_ERLANG,
        seed,
        num_slots=NUM_SLOTS,
    )
    env = PDSRMSAEnv(
        TOPOLOGY,
        NUM_SLOTS,
        LOAD_ERLANG,
        seed,
        trace=trace,
        k_paths=K_PATHS,
        path_sort_strategy=PATH_SORT_STRATEGY,
    )
    env.reset()
    return env


def _collect_prefill_transitions(
    env: PDSRMSAEnv,
    replay: ReplayBuffer,
    num_transitions: int,
) -> int:
    """Run KSP-FF K=5 hops and add continuing-task transitions."""
    requests = env.trace.requests
    added = 0
    for i in range(min(num_transitions, max(0, len(requests) - 1))):
        request = requests[i]
        env.advance_external(request)
        pre_state = env.snapshot_true_state()
        action = ksp_ff_action(env, request)
        result = env.step(action, request=request)
        reward = 1.0 if result["success"] else 0.0
        done = False
        next_request = requests[i + 1]
        env.advance_external(next_request)
        next_pre_state = env.snapshot_true_state()
        replay.add(pre_state, action, reward, next_pre_state, done, request, next_request)
        added += 1
    return added


def _run_online_decision(
    env: PDSRMSAEnv,
    agent: PDSAgent,
    request,
    next_request,
    done: bool,
    replay: ReplayBuffer,
) -> Tuple[Union[RMSAAction, BlockedAction], float]:
    """Advance, select an action, step, and store the transition."""
    env.advance_external(request)
    pre_state = env.snapshot_true_state()
    action = agent.select_action(env, request)
    result = env.step(action, request=request)
    reward = 1.0 if result["success"] else 0.0
    env.advance_external(next_request)
    next_pre_state = env.snapshot_true_state()
    replay.add(pre_state, action, reward, next_pre_state, done, request, next_request)
    return action, reward


def _validate_agent(
    agent: PDSAgent,
    trace_seed: int,
    num_requests: int,
    warmup_requests: int = 0,
) -> Dict[str, Any]:
    """Run an agent greedily on a validation trace and return blocking metrics.

    The first ``warmup_requests`` are executed but not counted in the reported
    blocking rate, matching the protocol's warmup/evaluation separation.
    """
    env = _make_env(trace_seed, num_requests + warmup_requests)
    requests = env.trace.requests
    admitted = 0
    blocked = 0
    agent.set_epsilon(0.0)
    for i, request in enumerate(requests):
        env.advance_external(request)
        action = agent.select_action(env, request)
        result = env.step(action, request=request)
        if i < warmup_requests:
            continue
        if result["success"]:
            admitted += 1
        else:
            blocked += 1
    total = admitted + blocked
    return {
        "admitted": admitted,
        "blocked": blocked,
        "blocking_rate": blocked / total if total > 0 else 0.0,
        "total_requests": total,
        "warmup_requests": warmup_requests,
    }


def _validate_checkpoint_agents(
    pds_agent: PDSAgent,
    pred_agent: PreDAgent,
    validation_seeds: List[int],
    validation_requests: int,
    validation_warmup: int,
) -> Dict[str, Any]:
    """Evaluate both agents on every paired validation trace."""
    per_seed = []
    for val_seed in validation_seeds:
        per_seed.append({
            "seed": int(val_seed),
            "pds": _validate_agent(
                pds_agent,
                val_seed,
                validation_requests,
                warmup_requests=validation_warmup,
            ),
            "pred": _validate_agent(
                pred_agent,
                val_seed,
                validation_requests,
                warmup_requests=validation_warmup,
            ),
        })
    return {
        "per_seed": per_seed,
        "pds_mean_blocking": float(np.mean(
            [item["pds"]["blocking_rate"] for item in per_seed]
        )) if per_seed else float("inf"),
        "pred_mean_blocking": float(np.mean(
            [item["pred"]["blocking_rate"] for item in per_seed]
        )) if per_seed else float("inf"),
    }


def _update_agent(
    agent,
    replay: ReplayBuffer,
    batch_size: int,
    grad_updates: int,
    target_sync_every: int,
    current_total_updates: int,
    progress_label: Optional[str] = None,
    progress_every: int = 0,
) -> Tuple[float, int]:
    """Perform ``grad_updates`` gradient updates for a single agent and sync targets."""
    loss = 0.0
    started = time.perf_counter()
    for local_update in range(grad_updates):
        if len(replay) < batch_size:
            continue
        batch = replay.sample_to_states(batch_size)
        loss = agent.update(batch)
        current_total_updates += 1
        if current_total_updates % target_sync_every == 0:
            agent.sync_target()
        completed = local_update + 1
        if (
            progress_label
            and progress_every > 0
            and (completed % progress_every == 0 or completed == grad_updates)
        ):
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            print(
                f"[{progress_label}] {completed}/{grad_updates} updates "
                f"({rate:.2f} updates/s, loss={loss:.6g})",
                flush=True,
            )
    return loss, current_total_updates


def _ksp_ff_blocking(env: PDSRMSAEnv, warmup_requests: int = 0) -> float:
    """Compute blocking rate of KSP-FF on the provided environment/trace.

    The first ``warmup_requests`` are executed but not counted, matching the
    validation/test protocol.
    """
    blocked = 0
    admitted = 0
    for i, request in enumerate(env.trace.requests):
        env.advance_external(request)
        action = ksp_ff_action(env, request)
        result = env.step(action, request=request)
        if i < warmup_requests:
            continue
        if result["success"]:
            admitted += 1
        else:
            blocked += 1
    total = admitted + blocked
    return blocked / total if total > 0 else 0.0


def _write_training_progress(
    path: Path,
    config: Dict[str, Any],
    online_decision_count: int,
    pds_total_updates: int,
    pred_total_updates: int,
    pds_loss: float,
    pred_loss: float,
    epsilon: float,
    phase_started: float,
    device: torch.device,
) -> None:
    """Write a live progress snapshot; atomic rename to avoid readers seeing partial writes."""
    elapsed = time.perf_counter() - phase_started
    progress = {
        "seed": int(config["seed"]),
        "online_decisions": int(online_decision_count),
        "online_decisions_target": int(config["online_decisions"]),
        "update_every": int(config.get("update_every", 4)),
        "batch_size": int(config.get("batch_size", 64)),
        "pds_total_updates": int(pds_total_updates),
        "pred_total_updates": int(pred_total_updates),
        "epsilon": float(epsilon),
        "pds_loss": float(pds_loss),
        "pred_loss": float(pred_loss),
        "elapsed_seconds": float(elapsed),
        "decisions_per_second": float(online_decision_count / elapsed) if elapsed > 0 else 0.0,
        "cuda_allocated_mb": (
            float(torch.cuda.memory_allocated(device) / 1024**2)
            if device.type == "cuda" else 0.0
        ),
        "cuda_reserved_mb": (
            float(torch.cuda.memory_reserved(device) / 1024**2)
            if device.type == "cuda" else 0.0
        ),
    }
    _atomic_write(path, json.dumps(progress, indent=2, default=str).encode("utf-8"))


def _default_config() -> Dict[str, Any]:
    return {
        "seed": 6101,
        "output_dir": "sa_hmarl/pds_rmsa/artifacts/phase1_cost239_load300_k5hops_b64_u4",
        "prefill_size": 50000,
        "pretrain_updates": 10000,
        "online_decisions": 300000,
        "target_sync_every": 2000,
        "batch_size": 64,
        "update_every": 4,
        "lr": 1e-4,
        "gamma": GAMMA,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 200000,
        "grad_clip_norm": 10.0,
        "validation_seeds": [8101, 8102, 8103, 8104, 8105],
        "validation_requests": 10000,
        "validation_warmup": 1000,
        "device": "cpu",
        "smoke": False,
    }


def train_cost239_phase1(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run Phase 1 training for a single seed and return results."""
    config = {**_default_config(), **config}
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False; "
            "refusing an implicit CPU training run"
        )
    feature_config = _make_feature_config(NUM_SLOTS)
    input_dim = _compute_input_dim(feature_config)
    num_directed_links = 2 * load_topology(TOPOLOGY).num_edges

    pds_agent = PDSAgent(
        input_dim,
        feature_config,
        TOPOLOGY,
        NUM_SLOTS,
        gamma=config["gamma"],
        lr=config["lr"],
        device=device,
        k_paths=K_PATHS,
        path_sort_strategy=PATH_SORT_STRATEGY,
    )
    pred_agent = PreDAgent(
        input_dim,
        feature_config,
        TOPOLOGY,
        NUM_SLOTS,
        gamma=config["gamma"],
        lr=config["lr"],
        device=device,
        k_paths=K_PATHS,
        path_sort_strategy=PATH_SORT_STRATEGY,
    )
    pds_device = next(pds_agent.online_net.parameters()).device
    pred_device = next(pred_agent.online_net.parameters()).device
    def _same_device(a: torch.device, b: torch.device) -> bool:
        if a.type != b.type:
            return False
        if a.type == "cuda":
            return a.index == b.index or a.index is None or b.index is None
        return True
    if not _same_device(pds_device, device) or not _same_device(pred_device, device):
        raise RuntimeError(
            f"Model device mismatch: requested={device}, "
            f"pds={pds_device}, pred={pred_device}"
        )
    device_audit = {
        "requested_device": str(device),
        "pds_parameter_device": str(pds_device),
        "pred_parameter_device": str(pred_device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }
    print(f"[seed {config['seed']}] device audit: {device_audit}", flush=True)
    _atomic_write(
        output_dir / f"seed_{config['seed']}_device_audit.json",
        json.dumps(device_audit, indent=2).encode("utf-8"),
    )

    assert pds_agent.input_dim == pred_agent.input_dim, "PDS/PreD input dimension mismatch"
    assert pds_agent.count_parameters() == pred_agent.count_parameters(), "PDS/PreD parameter count mismatch"

    pds_replay = ReplayBuffer(
        capacity=100000,
        num_directed_links=num_directed_links,
        num_slots=NUM_SLOTS,
    )
    pred_replay = ReplayBuffer(
        capacity=100000,
        num_directed_links=num_directed_links,
        num_slots=NUM_SLOTS,
    )

    # Prefill: use one env for KSP-FF and add the same transitions to both buffers.
    trace_length = max(
        config["prefill_size"] + 1,
        config["online_decisions"] + 1,
    )
    phase_started = time.perf_counter()
    print(
        f"[seed {config['seed']}] collecting {config['prefill_size']} "
        "KSP-FF K=5 hops prefill transitions",
        flush=True,
    )
    prefill_env = _make_env(config["seed"], trace_length)
    prefill_added = _collect_prefill_transitions(
        prefill_env, pds_replay, config["prefill_size"]
    )
    _collect_prefill_transitions(
        _make_env(config["seed"], trace_length), pred_replay, config["prefill_size"]
    )
    print(
        f"[seed {config['seed']}] prefill complete in "
        f"{time.perf_counter() - phase_started:.1f}s",
        flush=True,
    )

    # Pre-train
    pds_agent.set_epsilon(config["epsilon_start"])
    pred_agent.set_epsilon(config["epsilon_start"])
    pds_total_updates = 0
    pred_total_updates = 0
    pds_loss, pds_total_updates = _update_agent(
        pds_agent,
        pds_replay,
        config["batch_size"],
        config["pretrain_updates"],
        config["target_sync_every"],
        pds_total_updates,
        progress_label=f"seed {config['seed']} pds_pretrain",
        progress_every=max(1, min(100, config["pretrain_updates"])),
    )
    pred_loss, pred_total_updates = _update_agent(
        pred_agent,
        pred_replay,
        config["batch_size"],
        config["pretrain_updates"],
        config["target_sync_every"],
        pred_total_updates,
        progress_label=f"seed {config['seed']} pred_pretrain",
        progress_every=max(1, min(100, config["pretrain_updates"])),
    )

    # Online training with separate envs for each agent
    online_decisions = config["online_decisions"]
    pds_env = _make_env(config["seed"], trace_length)
    pred_env = _make_env(config["seed"], trace_length)
    pds_requests = pds_env.trace.requests
    pred_requests = pred_env.trace.requests
    online_decision_count = 0
    curve = []
    checkpoints = sorted({
        budget for budget in CHECKPOINT_BUDGETS if budget <= online_decisions
    } | {online_decisions})
    saved_checkpoints = []
    checkpoint_validations = []
    saved_budgets = set()

    def save_and_validate_checkpoint(online_budget: int) -> None:
        ckpt_dir = (
            output_dir / "checkpoints"
            / f"seed_{config['seed']}_online_{online_budget}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _save_agent_checkpoint(ckpt_dir / "pds.pt", pds_agent, online_budget, config)
        _save_agent_checkpoint(ckpt_dir / "pred.pt", pred_agent, online_budget, config)
        validation = _validate_checkpoint_agents(
            pds_agent,
            pred_agent,
            list(config["validation_seeds"]),
            int(config["validation_requests"]),
            int(config.get("validation_warmup", 0)),
        )
        validation.update({
            "online_decisions": int(online_budget),
            "checkpoint_dir": str(ckpt_dir),
            "pds_loss": float(pds_loss),
            "pred_loss": float(pred_loss),
        })
        checkpoint_validations.append(validation)
        saved_checkpoints.append(str(ckpt_dir))
        saved_budgets.add(int(online_budget))

    # Budget zero means prefill and pretraining are complete, with no online decisions.
    save_and_validate_checkpoint(0)

    progress_path = output_dir / "TRAINING_PROGRESS.json"
    last_progress_write = 0

    for step in range(online_decisions):
        if step >= len(pds_requests) - 1 or step >= len(pred_requests) - 1:
            break
        request = pds_requests[step]
        next_request = pds_requests[step + 1]
        done = False

        epsilon = linear_epsilon(
            step,
            config["epsilon_decay_steps"],
            config["epsilon_start"],
            config["epsilon_end"],
        )
        pds_agent.set_epsilon(epsilon)
        pred_agent.set_epsilon(epsilon)

        _, _ = _run_online_decision(pds_env, pds_agent, request, next_request, done, pds_replay)
        _, _ = _run_online_decision(pred_env, pred_agent, request, next_request, done, pred_replay)
        online_decision_count += 1

        if online_decision_count % config["update_every"] == 0:
            pds_loss, pds_total_updates = _update_agent(
                pds_agent,
                pds_replay,
                config["batch_size"],
                1,
                config["target_sync_every"],
                pds_total_updates,
            )
            pred_loss, pred_total_updates = _update_agent(
                pred_agent,
                pred_replay,
                config["batch_size"],
                1,
                config["target_sync_every"],
                pred_total_updates,
            )

        budget = online_decision_count
        curve.append({
            "online_decisions": budget,
            "online_step": step + 1,
            "epsilon": epsilon,
            "pds_loss": pds_loss,
            "pred_loss": pred_loss,
            "pds_total_updates": pds_total_updates,
            "pred_total_updates": pred_total_updates,
        })

        if budget % 1000 == 0 and budget > last_progress_write:
            _write_training_progress(
                progress_path,
                config,
                online_decision_count,
                pds_total_updates,
                pred_total_updates,
                pds_loss,
                pred_loss,
                epsilon,
                phase_started,
                device,
            )
            last_progress_write = budget

        if budget in checkpoints:
            save_and_validate_checkpoint(budget)

    # Final update to ensure all collected online transitions are trained on at least once.
    if online_decision_count > 0 and online_decision_count % config["update_every"] != 0:
        pds_loss, pds_total_updates = _update_agent(
            pds_agent,
            pds_replay,
            config["batch_size"],
            1,
            config["target_sync_every"],
            pds_total_updates,
        )
        pred_loss, pred_total_updates = _update_agent(
            pred_agent,
            pred_replay,
            config["batch_size"],
            1,
            config["target_sync_every"],
            pred_total_updates,
        )

    # Save final checkpoint if not already saved
    final_budget = online_decision_count
    if final_budget not in saved_budgets:
        save_and_validate_checkpoint(final_budget)

    final_validation = next(
        item for item in checkpoint_validations
        if item["online_decisions"] == final_budget
    )

    # KSP-FF baseline on a small trace for reference
    baseline_env = _make_env(config["seed"] + 100000, min(5000, len(pds_env.trace.requests)))
    ksp_ff_blocking_rate = _ksp_ff_blocking(baseline_env)

    results = {
        "seed": config["seed"],
        "input_dim": input_dim,
        "batch_size": int(config.get("batch_size", 64)),
        "update_every": int(config.get("update_every", 4)),
        "pds_param_count": pds_agent.count_parameters(),
        "pred_param_count": pred_agent.count_parameters(),
        "prefill_added": prefill_added,
        "online_decisions": online_decision_count,
        "pds_total_updates": pds_total_updates,
        "pred_total_updates": pred_total_updates,
        "ksp_ff_baseline_blocking": ksp_ff_blocking_rate,
        "validation_results": final_validation["per_seed"],
        "checkpoint_validations": checkpoint_validations,
        "checkpoints": saved_checkpoints,
        "protocol": _checkpoint_protocol(pds_agent, config),
        "device_audit": device_audit,
    }

    _atomic_write(
        output_dir / f"seed_{config['seed']}_results.json",
        json.dumps(results, indent=2, default=str).encode("utf-8"),
    )
    _atomic_write(
        output_dir / f"seed_{config['seed']}_curve.json",
        json.dumps(curve, indent=2, default=str).encode("utf-8"),
    )

    return results
