"""Train and evaluate a fresh paper-parity pure-RMSA PPO-R proposer warm start.

This is R1 of the native retraining protocol. It deliberately does not load the
legacy PPO-R or Strict v1.3 Ranker checkpoints. The actor is supervised with a
multi-positive objective over the canonical KSP-FF and FF-KSP actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sa_hmarl.agents.ppo_agents import PPOAgentR
from sa_hmarl.baselines.rmsa_baselines import (
    ff_ksp_highest_mod_action,
    ksp_ff_highest_mod_action,
)
from sa_hmarl.evaluation.pure_rmsa_paper_env import (
    PureRMSAPaperEnv,
    PureRMSARequest,
    generate_paper_requests,
    paper_modulation_registry,
)


TOPOLOGIES = {
    "cost239": "cost239_deeprmsa",
    "nsfnet": "xlron_nsfnet_deeprmsa",
    "usnet": "xlron_usnet_gcnrmsa",
    "jpn48": "xlron_jpn48",
}
FEATURE_SCALE = np.asarray(
    [10000.0, 50.0, 100.0, 1.0, 1.0, 4.0, 10000.0, 100.0, 100.0, 1.0, 1.0],
    dtype=np.float32,
)
MAX_ACTIONS = 50 * 4 * 10


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _trace_hash(requests: Sequence[PureRMSARequest]) -> str:
    raw = json.dumps([asdict(req) for req in requests], sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _normalize_pad(agent: PPOAgentR, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    features, mask = agent.build_action_features(obs)
    features = np.asarray(features, dtype=np.float32) / FEATURE_SCALE
    mask = np.asarray(mask, dtype=bool)
    if len(mask) > MAX_ACTIONS:
        raise RuntimeError(f"Action count {len(mask)} exceeds protocol maximum {MAX_ACTIONS}")
    padded_f = np.zeros((MAX_ACTIONS, len(FEATURE_SCALE)), dtype=np.float32)
    padded_m = np.zeros(MAX_ACTIONS, dtype=bool)
    padded_f[: len(features)] = features
    padded_m[: len(mask)] = mask
    if not np.all(np.isfinite(padded_f)):
        raise RuntimeError("Non-finite native proposer features")
    return padded_f, padded_m


def _expert_actions(obs: Dict[str, Any]) -> List[int]:
    actions = [ksp_ff_highest_mod_action(obs), ff_ksp_highest_mod_action(obs)]
    return sorted({int(a) for a in actions if a is not None})


def _collect(
    agent: PPOAgentR,
    topology: str,
    load: float,
    seeds: Iterable[int],
    requests_per_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    positives: List[np.ndarray] = []
    hashes: List[str] = []
    for seed in seeds:
        env = PureRMSAPaperEnv(TOPOLOGIES[topology], num_slots=100, k_paths=50, max_blocks=10)
        requests = generate_paper_requests(env.num_nodes, seed, requests_per_seed, load)
        hashes.append(_trace_hash(requests))
        env.reset()
        for index, req in enumerate(requests):
            env.advance_time(req.arrival_time)
            obs = env.build_observation(req)
            expert = _expert_actions(obs)
            if not expert:
                continue
            f, m = _normalize_pad(agent, obs)
            positive = np.zeros(MAX_ACTIONS, dtype=bool)
            positive[expert] = True
            if not np.all(m[expert]):
                raise RuntimeError("Canonical expert returned an action outside the legal mask")
            features.append(f.astype(np.float16))
            masks.append(m)
            positives.append(positive)
            # Alternate the two canonical experts to expose the actor to both state distributions.
            deployed = expert[index % len(expert)]
            result = env.step(deployed, obs, req)
            if not result["success"]:
                raise RuntimeError(f"Expert action failed: {result}")
    return np.stack(features), np.stack(masks), np.stack(positives), hashes


def _multi_positive_loss(
    agent: PPOAgentR,
    features: torch.Tensor,
    masks: torch.Tensor,
    positives: torch.Tensor,
    entropy_coef: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = agent.policy_net(features)
    legal_logits = logits.masked_fill(~masks, -1e9)
    positive_logits = logits.masked_fill(~positives, -1e9)
    loss = (torch.logsumexp(legal_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()
    log_probs = torch.log_softmax(legal_logits, dim=1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return loss - entropy_coef * entropy, entropy


def _recall_metrics(agent: PPOAgentR, f: np.ndarray, m: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    agent.policy_net.eval()
    top1_hits = top30_hits = 0
    with torch.no_grad():
        for start in range(0, len(f), 64):
            bf = torch.as_tensor(f[start:start + 64], dtype=torch.float32, device=agent.device)
            bm = torch.as_tensor(m[start:start + 64], dtype=torch.bool, device=agent.device)
            bp = torch.as_tensor(p[start:start + 64], dtype=torch.bool, device=agent.device)
            logits = agent.policy_net(bf).masked_fill(~bm, -1e9)
            top1 = logits.argmax(dim=1)
            top1_hits += int(bp.gather(1, top1[:, None]).sum().item())
            top30 = torch.topk(logits, k=30, dim=1).indices
            top30_hits += int(bp.gather(1, top30).any(dim=1).sum().item())
    agent.policy_net.train()
    n = max(len(f), 1)
    return {"samples": len(f), "expert_top1_recall": top1_hits / n, "expert_top30_recall": top30_hits / n}


def _phi_after(env: PureRMSAPaperEnv, obs: Dict[str, Any], action: int) -> float:
    mods = env.mod_reg.num_formats
    path_idx = int(action) // (mods * env.max_blocks)
    rem = int(action) % (mods * env.max_blocks)
    mod_idx, block_idx = divmod(rem, env.max_blocks)
    path = obs["candidate_paths"][path_idx]
    start, _ = obs["candidate_blocks_per_path_mod"][path_idx][mod_idx][block_idx]
    required = int(obs["required_fs_per_path_mod"][path_idx][mod_idx])
    if not env.net.allocate(path, int(start), required):
        raise RuntimeError("Exact temporary afterstate allocation failed")
    summary = env.net.spectrum_summary()
    env.net.release(path, int(start), required)
    lfb_ratio = float(summary["largest_free_block"]) / env.num_slots
    return float(math.log1p(float(summary["free_ratio"]) * 100.0) + 0.3 * math.log1p(lfb_ratio * 100.0))


def _select(agent: PPOAgentR, obs: Dict[str, Any], method: str) -> Tuple[Optional[int], Dict[str, Any]]:
    if method == "ksp_ff_k50_hops":
        return ksp_ff_highest_mod_action(obs), {}
    if method == "ff_ksp_k50_hops":
        return ff_ksp_highest_mod_action(obs), {}
    f, m = _normalize_pad(agent, obs)
    legal = np.flatnonzero(m)
    if legal.size == 0:
        return None, {"expert_top30": False}
    with torch.no_grad():
        logits = agent.policy_net(torch.as_tensor(f, dtype=torch.float32, device=agent.device)[None]).squeeze(0).cpu().numpy()
    ordered = legal[np.argsort(-logits[legal], kind="stable")]
    top30 = ordered[: min(30, len(ordered))]
    expert = set(_expert_actions(obs))
    info = {"expert_top30": bool(expert.intersection(map(int, top30)))}
    if method == "native_proposer_top1":
        return int(ordered[0]), info
    if method == "native_top30_phi_oracle":
        raise RuntimeError("Oracle selection requires the environment")
    raise ValueError(method)


def _evaluate_one(
    agent: PPOAgentR,
    topology: str,
    load: float,
    seed: int,
    method: str,
    warmup: int,
    evaluated: int,
) -> Dict[str, Any]:
    env = PureRMSAPaperEnv(TOPOLOGIES[topology], num_slots=100, k_paths=50, max_blocks=10)
    requests = generate_paper_requests(env.num_nodes, seed, warmup + evaluated, load)
    env.reset()
    blocked = total = recall_hits = decisions = 0
    for index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs = env.build_observation(req)
        if method == "native_top30_phi_oracle":
            f, m = _normalize_pad(agent, obs)
            legal = np.flatnonzero(m)
            if legal.size:
                with torch.no_grad():
                    logits = agent.policy_net(torch.as_tensor(f, dtype=torch.float32)[None]).squeeze(0).numpy()
                top30 = legal[np.argsort(-logits[legal], kind="stable")[: min(30, len(legal))]]
                action = int(max(top30, key=lambda a: _phi_after(env, obs, int(a))))
                info = {"expert_top30": bool(set(_expert_actions(obs)).intersection(map(int, top30)))}
            else:
                action, info = None, {"expert_top30": False}
        else:
            action, info = _select(agent, obs, method)
        success = False if action is None else bool(env.step(action, obs, req)["success"])
        if index >= warmup:
            total += 1
            blocked += int(not success)
            if method.startswith("native") and action is not None:
                decisions += 1
                recall_hits += int(info["expert_top30"])
    return {
        "topology": topology, "load_erlang": load, "seed": seed, "method": method,
        "requests": total, "blocked": blocked, "blocking_rate": blocked / max(total, 1),
        "expert_top30_recall": recall_hits / max(decisions, 1), "trace_hash": _trace_hash(requests),
    }


def _evaluate_job(job: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    agent = PPOAgentR(11, paper_modulation_registry(), hidden_dims=(128, 64), device="cpu")
    agent.policy_net.load_state_dict(checkpoint["model_state"])
    agent.policy_net.eval()
    return _evaluate_one(
        agent, job["topology"], job["load"], job["seed"], job["method"],
        job["warmup"], job["evaluated"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="sa_hmarl/experiments/pure_rmsa_native_retrain_r1/run_20260721")
    parser.add_argument("--topology", default="cost239", choices=sorted(TOPOLOGIES))
    parser.add_argument("--load-erlang", type=float, default=600.0)
    parser.add_argument("--train-seeds", default="8101,8102")
    parser.add_argument("--train-requests", type=int, default=1500)
    parser.add_argument("--val-seed", type=int, default=8201)
    parser.add_argument("--val-requests", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=1e-4)
    parser.add_argument("--eval-seeds", default="5001,5002,5003,5004,5005")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--evaluated", type=int, default=6000)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Any] = {"stage": "collect", "started": time.time(), "args": vars(args)}
    _atomic_json(output / "RUN_STATUS.json", status)

    agent = PPOAgentR(11, paper_modulation_registry(), hidden_dims=(128, 64), lr=args.lr, device=args.device)
    train_seeds = [int(x) for x in args.train_seeds.split(",")]
    train_f, train_m, train_p, train_hashes = _collect(
        agent, args.topology, args.load_erlang, train_seeds, args.train_requests
    )
    val_f, val_m, val_p, val_hashes = _collect(
        agent, args.topology, args.load_erlang, [args.val_seed], args.val_requests
    )
    status.update({"stage": "train", "train_samples": len(train_f), "val_samples": len(val_f)})
    _atomic_json(output / "RUN_STATUS.json", status)

    best_key = (-1.0, -1.0, float("-inf"))
    history = []
    checkpoint = output / "native_proposer_best.pt"
    for epoch in range(1, args.epochs + 1):
        permutation = np.random.permutation(len(train_f))
        losses = []
        for start in range(0, len(permutation), args.batch_size):
            idx = permutation[start:start + args.batch_size]
            bf = torch.as_tensor(train_f[idx], dtype=torch.float32, device=args.device)
            bm = torch.as_tensor(train_m[idx], dtype=torch.bool, device=args.device)
            bp = torch.as_tensor(train_p[idx], dtype=torch.bool, device=args.device)
            loss, _ = _multi_positive_loss(agent, bf, bm, bp, args.entropy_coef)
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 0.5)
            agent.optimizer.step()
            losses.append(float(loss.item()))
        metrics = _recall_metrics(agent, val_f, val_m, val_p)
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses))})
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        selection_key = (
            metrics["expert_top30_recall"],
            metrics["expert_top1_recall"],
            -metrics["train_loss"],
        )
        if selection_key > best_key:
            best_key = selection_key
            torch.save({
                "model_state": agent.policy_net.state_dict(), "input_dim": 11,
                "hidden_dims": (128, 64), "agent_r_feature_mode": "default",
                "modulation_profile": "paper_parity", "feature_scale": FEATURE_SCALE,
                "k_paths": 50, "max_blocks": 10, "path_sort": "hops",
                "training_stage": "multi_positive_bc_warm_start", "epoch": epoch,
                "val_metrics": metrics,
            }, checkpoint)

    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    agent.policy_net.load_state_dict(saved["model_state"])
    status["stage"] = "evaluate"
    _atomic_json(output / "RUN_STATUS.json", status)
    runs = []
    methods = ("ksp_ff_k50_hops", "ff_ksp_k50_hops", "native_proposer_top1", "native_top30_phi_oracle")
    jobs = [
        {
            "checkpoint": str(checkpoint), "topology": args.topology,
            "load": args.load_erlang, "seed": seed, "method": method,
            "warmup": args.warmup, "evaluated": args.evaluated,
        }
        for seed in [int(x) for x in args.eval_seeds.split(",")]
        for method in methods
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.eval_workers)) as executor:
        futures = [executor.submit(_evaluate_job, job) for job in jobs]
        for future in as_completed(futures):
            run = future.result()
            runs.append(run)
            print(json.dumps(run), flush=True)
    runs.sort(key=lambda r: (r["seed"], r["method"]))

    aggregate = {}
    for method in methods:
        subset = [r for r in runs if r["method"] == method]
        aggregate[method] = {
            "blocking_mean": float(np.mean([r["blocking_rate"] for r in subset])),
            "blocking_per_seed": [r["blocking_rate"] for r in subset],
            "expert_top30_recall_mean": float(np.mean([r["expert_top30_recall"] for r in subset])),
        }
    better = min(aggregate["ksp_ff_k50_hops"]["blocking_mean"], aggregate["ff_ksp_k50_hops"]["blocking_mean"])
    native = aggregate["native_proposer_top1"]["blocking_mean"]
    recall = aggregate["native_proposer_top1"]["expert_top30_recall_mean"]
    gate = bool(recall >= 0.95 and native - better <= 0.01)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "protocol": "pure_rmsa_native_retrain_r1", "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "train_trace_hashes": train_hashes, "val_trace_hashes": val_hashes,
        "history": history, "runs": runs, "aggregate": aggregate,
        "r1_gate_pass": gate,
        "r1_gate_rule": "expert Top-30 recall >=95% and Top-1 blocking gap <=1.0 pp",
    }
    _atomic_json(output / "R1_RESULTS.json", result)
    lines = ["# Pure-RMSA Native Proposer R1", "", "| Method | Blocking | Expert Top-30 recall |", "|---|---:|---:|"]
    for method in methods:
        row = aggregate[method]
        lines.append(f"| {method} | {100*row['blocking_mean']:.4f}% | {100*row['expert_top30_recall_mean']:.2f}% |")
    lines.extend(["", f"**R1 gate: {'PASS' if gate else 'FAIL'}**", "", result["r1_gate_rule"]])
    (output / "R1_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    phi_oracle = aggregate["native_top30_phi_oracle"]["blocking_mean"]
    if gate and phi_oracle < native:
        decision = (
            "GO: run common-future Top-30 blocking sensitivity, then generate native Ranker data. "
            "Immediate PhiSpec is only a diagnostic and is not automatically accepted as the label."
        )
    elif gate:
        decision = (
            "CONDITIONAL GO: proposer R1 passed, but the immediate PhiSpec Oracle did not improve "
            "blocking. Do not train a PhiSpec Ranker. First run common-future Top-30 blocking/identity "
            "sensitivity and continue only if that causal ceiling is non-zero."
        )
    else:
        decision = "STOP: do not train the Ranker; improve proposer recall or closed-loop behavior first."
    (output / "NEXT_STEP_DECISION.md").write_text(f"# Next Step\n\n{decision}\n", encoding="utf-8")
    status.update({"stage": "complete", "completed": time.time(), "r1_gate_pass": gate})
    _atomic_json(output / "RUN_STATUS.json", status)


if __name__ == "__main__":
    main()
