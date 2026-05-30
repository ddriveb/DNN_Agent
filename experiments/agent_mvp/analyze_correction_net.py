"""CorrectionNet interpretability analysis.

Answers:
1. What is the distribution of correction_score for successful vs blocked requests?
2. Does correction_score correlate with future blocking risk?
3. Does correction_score correlate with fragmentation increase?
4. Does CorrectionNet suppress high-P_success actions that hurt long-term resources?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json
from collections import defaultdict

from eval_imitation import create_env
from fixed_trace import load_trace, generate_and_save_trace
from topk_selector_agent import TopKSelectorAgent
from correction_agent import TopK2CorrectionAgent
from baselines import YinLikeAgent


def load_predictor(path: str, max_servers: int = 128, device: str = "cpu"):
    from predictor import Predictor
    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)
    return predictor


def run_with_logging(agent, requests, topology, num_slots, num_servers, preload, seed):
    """Run agent and log per-request decisions."""
    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
    env.reset()
    if hasattr(agent, 'mec'):
        agent.mec = env.mec
    if hasattr(agent, 'encoder'):
        agent.encoder = env.encoder

    rng = np.random.RandomState(seed)
    from dnn_models import get_split_bandwidth_map
    bw_map = get_split_bandwidth_map()
    for _ in range(preload):
        src = rng.randint(0, env.net.NUM_NODES)
        dst = rng.randint(0, env.net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, env.net.NUM_NODES)
        bw = bw_map[rng.randint(0, 3)]
        success, path, start_slot, _ = env.mapper.map(src, dst, bw)
        if success:
            env.active_connections.append((path, start_slot, bw, rng.exponential(10.0), -1, 0.0))

    logs = []
    for req in requests:
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)

        log_entry = {
            "request_idx": len(logs),
            "split_id": split_id,
            "server_id": server_id,
            "success": result.success,
            "reward": float(result.reward),
            "delay_ms": float(result.real_delay_ms),
            "bw_slots": result.bandwidth_slots,
            "info_type": info.get("type", "unknown"),
        }

        # Extract evaluated candidates if available
        if "evaluated" in info:
            log_entry["evaluated"] = []
            for cand in info["evaluated"]:
                log_entry["evaluated"].append({
                    "action_id": cand["action_id"],
                    "split_id": cand.get("split_id", -1),
                    "server_id": cand.get("server_id", -1),
                    "pred_score": float(cand.get("pred_score", cand.get("score", 0.0))),
                    "correction": float(cand.get("correction", 0.0)),
                    "final_score": float(cand.get("final_score", cand.get("score", 0.0))),
                })
            # Identify which candidate was selected
            selected = None
            for cand in log_entry["evaluated"]:
                if cand.get("split_id") == split_id and cand.get("server_id") == server_id:
                    selected = cand
                    break
            log_entry["selected"] = selected

        logs.append(log_entry)

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean([l["reward"] for l in logs]))
    return logs, metrics


def compute_frag_index(env):
    """Compute global fragmentation index."""
    frags = []
    for link, slots in env.net.link_states.items():
        free = ~slots
        total_free = int(np.sum(free))
        if total_free == 0 or total_free == env.net.num_slots:
            continue
        max_free = 0
        curr = 0
        for v in free:
            if v:
                curr += 1
                max_free = max(max_free, curr)
            else:
                curr = 0
        frags.append(1.0 - max_free / total_free)
    return float(np.mean(frags)) if frags else 0.0


def analyze():
    device = "cpu"
    topology, num_slots, num_servers = "nsfnet", 32, 5
    num_requests, arr, ht, preload, seed = 2000, 5.0, 10.0, 300, 42

    base = Path(__file__).parent.parent / "predictor_mvp"
    predictor_path = str(base / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)
    correction_ckpt = str(Path(__file__).parent / "checkpoints" / "correction_net.pt")

    trace_fname = f"trace_n14_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
    trace_dir = Path(__file__).parent / "traces"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        trace_path = generate_and_save_trace(14, num_requests, arr, ht, seed)
    requests = load_trace(trace_path)

    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

    # Run with CorrectionNet λ=0.2
    agent = TopK2CorrectionAgent(
        correction_ckpt, predictor, encoder, mec,
        num_servers=num_servers, top_k=2, lambda_corr=0.2,
        alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
        use_enhanced_state=True, device=device,
    )
    logs, metrics = run_with_logging(agent, requests, topology, num_slots, num_servers, preload, seed)
    print(f"CorrectionNet λ=0.2: blocking={metrics['blocking_rate']*100:.2f}%")

    # Run baseline TopK2 for comparison
    agent_base = TopKSelectorAgent(
        "checkpoints/imitation_agent_enhanced.pt",
        predictor, encoder, mec,
        num_servers=num_servers, top_k=2,
        alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
        use_enhanced_state=True, device=device,
    )
    logs_base, metrics_base = run_with_logging(agent_base, requests, topology, num_slots, num_servers, preload, seed)
    print(f"TopK2-30D baseline: blocking={metrics_base['blocking_rate']*100:.2f}%")

    # ===== Analysis 1: correction_score distribution =====
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Correction Score Distribution")
    print("=" * 60)

    sel_corrs_success = []
    sel_corrs_fail = []
    for log in logs:
        if log.get("selected"):
            c = log["selected"]["correction"]
            if log["success"]:
                sel_corrs_success.append(c)
            else:
                sel_corrs_fail.append(c)

    print(f"Selected action correction — Success: mean={np.mean(sel_corrs_success):.3f}, std={np.std(sel_corrs_success):.3f}, n={len(sel_corrs_success)}")
    print(f"Selected action correction — Fail:    mean={np.mean(sel_corrs_fail):.3f}, std={np.std(sel_corrs_fail):.3f}, n={len(sel_corrs_fail)}")

    # ===== Analysis 2: All evaluated candidates =====
    print("\n" + "=" * 60)
    print("ANALYSIS 2: All Evaluated Candidates")
    print("=" * 60)

    all_cands = []
    for log in logs:
        for cand in log.get("evaluated", []):
            all_cands.append({
                "request_idx": log["request_idx"],
                "success": log["success"],
                "pred_score": cand["pred_score"],
                "correction": cand["correction"],
                "final_score": cand["final_score"],
                "selected": (log.get("selected") and
                             cand["action_id"] == log["selected"]["action_id"]),
            })

    selected_cands = [c for c in all_cands if c["selected"]]
    not_selected_cands = [c for c in all_cands if not c["selected"]]

    print(f"Selected candidates — pred_score: {np.mean([c['pred_score'] for c in selected_cands]):.3f}, correction: {np.mean([c['correction'] for c in selected_cands]):.3f}")
    if not_selected_cands:
        print(f"Not-selected candidates — pred_score: {np.mean([c['pred_score'] for c in not_selected_cands]):.3f}, correction: {np.mean([c['correction'] for c in not_selected_cands]):.3f}")

    # ===== Analysis 3: Cases where CorrectionNet changed the decision =====
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Decision Changes (TopK2 vs CorrectionNet)")
    print("=" * 60)

    changed = 0
    changed_to_success = 0
    changed_to_fail = 0
    same = 0
    suppress_high_psuccess = 0

    for i, (log, log_base) in enumerate(zip(logs, logs_base)):
        if log["split_id"] != log_base["split_id"] or log["server_id"] != log_base["server_id"]:
            changed += 1
            if log["success"] and not log_base["success"]:
                changed_to_success += 1
            elif not log["success"] and log_base["success"]:
                changed_to_fail += 1

            # Check if CorrectionNet suppressed a high-predictor-score candidate
            base_selected = None
            corr_selected = None
            for cand in log.get("evaluated", []):
                if cand["split_id"] == log["split_id"] and cand["server_id"] == log["server_id"]:
                    corr_selected = cand
            for cand in log_base.get("evaluated", []):
                if cand["split_id"] == log_base["split_id"] and cand["server_id"] == log_base["server_id"]:
                    base_selected = cand

            if base_selected and corr_selected:
                # Find the base-selected candidate in CorrectionNet's evaluated list
                base_cand_in_corr = None
                for cand in log.get("evaluated", []):
                    if cand["action_id"] == base_selected["action_id"]:
                        base_cand_in_corr = cand
                        break
                if base_cand_in_corr and base_cand_in_corr["pred_score"] > corr_selected["pred_score"]:
                    suppress_high_psuccess += 1
        else:
            same += 1

    print(f"Total decisions: {len(logs)}")
    print(f"Same decision: {same} ({same/len(logs)*100:.1f}%)")
    print(f"Changed decision: {changed} ({changed/len(logs)*100:.1f}%)")
    print(f"  Changed and improved (fail→success): {changed_to_success}")
    print(f"  Changed and worsened (success→fail): {changed_to_fail}")
    print(f"  Suppressed higher-predictor-score candidate: {suppress_high_psuccess}")

    # ===== Analysis 4: Correction score vs future blocking =====
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Correction Score vs Future Blocking (window=50)")
    print("=" * 60)

    window = 50
    corr_vs_future = []
    for i, log in enumerate(logs):
        if log.get("selected"):
            future_success = all(logs[j]["success"] for j in range(i+1, min(i+window+1, len(logs))))
            future_block_rate = 1.0 - np.mean([logs[j]["success"] for j in range(i+1, min(i+window+1, len(logs)))])
            corr_vs_future.append({
                "correction": log["selected"]["correction"],
                "future_block_rate": future_block_rate,
                "success": log["success"],
            })

    # Bin by correction score
    bins = [(-10, -0.5), (-0.5, 0), (0, 0.5), (0.5, 10)]
    for lo, hi in bins:
        items = [x for x in corr_vs_future if lo <= x["correction"] < hi]
        if items:
            avg_future_block = np.mean([x["future_block_rate"] for x in items])
            success_rate = np.mean([x["success"] for x in items])
            print(f"  correction in [{lo:>5.1f}, {hi:>4.1f}): n={len(items):>4}, future_block={avg_future_block:.3f}, immediate_success={success_rate:.3f}")

    # ===== Save detailed results =====
    out = {
        "analysis_1": {
            "success_correction_mean": float(np.mean(sel_corrs_success)) if sel_corrs_success else None,
            "success_correction_std": float(np.std(sel_corrs_success)) if sel_corrs_success else None,
            "fail_correction_mean": float(np.mean(sel_corrs_fail)) if sel_corrs_fail else None,
            "fail_correction_std": float(np.std(sel_corrs_fail)) if sel_corrs_fail else None,
        },
        "analysis_3": {
            "total": len(logs),
            "same": same,
            "changed": changed,
            "changed_to_success": changed_to_success,
            "changed_to_fail": changed_to_fail,
            "suppress_high_psuccess": suppress_high_psuccess,
        },
        "analysis_4": [
            {"correction": x["correction"], "future_block_rate": x["future_block_rate"], "success": x["success"]}
            for x in corr_vs_future
        ],
        "per_request": [
            {
                "idx": log["request_idx"],
                "success": log["success"],
                "selected": log.get("selected"),
                "evaluated_count": len(log.get("evaluated", [])),
            }
            for log in logs
        ],
    }

    out_path = Path(__file__).parent / "results" / "correction_net_interpretability.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    analyze()
