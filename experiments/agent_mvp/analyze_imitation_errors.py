"""Phase C: Deep error analysis of ImitationAgent vs AdaptiveRuleAgent (teacher).

Outputs:
  - C1: Split/Server/Full action accuracy, Top-K hit rate
  - C2: Regret distribution, catastrophic mismatch rate
  - C3: Error bucketing by load, deadline, model, strategy, split_id

Run: python analyze_imitation_errors.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import torch.nn.functional as F
import pickle
import json
from collections import defaultdict

from train_imitation import ImitationAgent


def load_data_and_model():
    """Load dataset and trained model."""
    data_path = Path(__file__).parent / "data" / "imitation_dataset.pkl"
    ckpt_path = Path(__file__).parent / "checkpoints" / "imitation_agent.pt"

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = ImitationAgent(
        ckpt["state_dim"],
        ckpt["num_actions"],
        hidden_dims=ckpt.get("hidden_dims", (128, 128)),
        dropout=0.0,  # no dropout for inference
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return data, model, ckpt


def deterministic_split(states, actions, metadata, val_frac=0.2, seed=42):
    """Deterministic train/val split for reproducible analysis."""
    rng = np.random.RandomState(seed)
    n = len(states)
    perm = rng.permutation(n)
    split = int((1 - val_frac) * n)
    train_idx = perm[:split]
    val_idx = perm[split:]
    return (
        states[train_idx], actions[train_idx], [metadata[i] for i in train_idx],
        states[val_idx], actions[val_idx], [metadata[i] for i in val_idx],
    )


def compute_metrics(model, states, actions, metadata, num_servers):
    """Compute full suite of error analysis metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Batch forward pass
    batch_size = 512
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch = torch.from_numpy(states[i:i+batch_size]).float().to(device)
            logits = model(batch)
            all_logits.append(logits.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)  # (N, num_actions)

    num_actions = all_logits.shape[1]
    num_splits = num_actions // num_servers

    # Top-K predictions
    top_k = {}
    for k in [1, 3, 5]:
        top_k[k] = np.argsort(-all_logits, axis=1)[:, :k]

    # Per-sample analysis
    results = []
    n = len(states)

    for i in range(n):
        teacher_action = int(actions[i])
        teacher_split = teacher_action // num_servers
        teacher_server = teacher_action % num_servers
        meta = metadata[i]

        logits_i = all_logits[i]
        pred_action = int(np.argmax(logits_i))
        pred_split = pred_action // num_servers
        pred_server = pred_action % num_servers

        # Top-K hit
        top1_hit = pred_action == teacher_action
        top3_hit = teacher_action in top_k[3][i]
        top5_hit = teacher_action in top_k[5][i]

        # Component accuracy
        split_match = pred_split == teacher_split
        server_match = pred_server == teacher_server

        # Regret: logit gap between best student action and teacher action
        best_logit = float(np.max(logits_i))
        teacher_logit = float(logits_i[teacher_action])
        regret = best_logit - teacher_logit  # >= 0

        # Catastrophic mismatch: teacher not in top-3
        catastrophic = not top3_hit

        # Confidence on teacher action (softmax prob)
        probs = F.softmax(torch.from_numpy(logits_i), dim=0).numpy()
        teacher_prob = float(probs[teacher_action])
        pred_prob = float(probs[pred_action])

        # Rank of teacher action
        sorted_actions = np.argsort(-logits_i)
        teacher_rank = int(np.where(sorted_actions == teacher_action)[0][0]) + 1

        results.append({
            "teacher_action": teacher_action,
            "teacher_split": teacher_split,
            "teacher_server": teacher_server,
            "pred_action": pred_action,
            "pred_split": pred_split,
            "pred_server": pred_server,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
            "split_match": split_match,
            "server_match": server_match,
            "regret": regret,
            "catastrophic": catastrophic,
            "teacher_prob": teacher_prob,
            "pred_prob": pred_prob,
            "teacher_rank": teacher_rank,
            "model_name": meta.get("model_name", "unknown"),
            "deadline_ms": meta.get("deadline_ms", 0.0),
            "strategy": meta.get("strategy", meta.get("chosen_strategy", "unknown")),
            "load_level": meta.get("load_level", "unknown"),
            "source_node": meta.get("source_node", -1),
            "split_id_meta": meta.get("split_id", -1),
            "server_id_meta": meta.get("server_id", -1),
        })

    return results, all_logits


def aggregate_overall(results):
    """C1 + C2: Overall accuracy and regret metrics."""
    n = len(results)

    # C1: Accuracy
    full_match = sum(1 for r in results if r["top1_hit"]) / n
    split_acc = sum(1 for r in results if r["split_match"]) / n
    server_acc = sum(1 for r in results if r["server_match"]) / n
    top3_hit = sum(1 for r in results if r["top3_hit"]) / n
    top5_hit = sum(1 for r in results if r["top5_hit"]) / n

    # C2: Regret
    regrets = [r["regret"] for r in results]
    avg_regret = np.mean(regrets)
    median_regret = np.median(regrets)
    p95_regret = np.percentile(regrets, 95)
    max_regret = np.max(regrets)

    # Catastrophic mismatch
    cat_rate = sum(1 for r in results if r["catastrophic"]) / n

    # Teacher probability
    teacher_probs = [r["teacher_prob"] for r in results]
    avg_teacher_prob = np.mean(teacher_probs)

    # Teacher rank distribution
    ranks = [r["teacher_rank"] for r in results]
    rank_dist = defaultdict(int)
    for rank in ranks:
        rank_dist[min(rank, 10)] += 1  # bucket >=10
    rank_dist = {f"rank_{k}": v / n for k, v in sorted(rank_dist.items())}

    return {
        "n_samples": n,
        "full_action_accuracy": round(full_match, 4),
        "split_accuracy": round(split_acc, 4),
        "server_accuracy": round(server_acc, 4),
        "top3_hit_rate": round(top3_hit, 4),
        "top5_hit_rate": round(top5_hit, 4),
        "avg_regret": round(avg_regret, 4),
        "median_regret": round(median_regret, 4),
        "p95_regret": round(p95_regret, 4),
        "max_regret": round(max_regret, 4),
        "catastrophic_mismatch_rate": round(cat_rate, 4),
        "avg_teacher_prob": round(avg_teacher_prob, 4),
        "teacher_rank_distribution": {k: round(v, 4) for k, v in rank_dist.items()},
    }


def bucket_analysis(results, key_fn, label):
    """C3: Analyze errors by a metadata bucket."""
    buckets = defaultdict(list)
    for r in results:
        bucket = key_fn(r)
        if bucket is not None:
            buckets[bucket].append(r)

    out = {}
    for bucket, items in sorted(buckets.items()):
        n = len(items)
        out[bucket] = {
            "n": n,
            "full_acc": round(sum(1 for r in items if r["top1_hit"]) / n, 4),
            "split_acc": round(sum(1 for r in items if r["split_match"]) / n, 4),
            "server_acc": round(sum(1 for r in items if r["server_match"]) / n, 4),
            "top3_hit": round(sum(1 for r in items if r["top3_hit"]) / n, 4),
            "avg_regret": round(np.mean([r["regret"] for r in items]), 4),
            "catastrophic_rate": round(sum(1 for r in items if r["catastrophic"]) / n, 4),
            "avg_teacher_prob": round(np.mean([r["teacher_prob"] for r in items]), 4),
        }
    return out


def deadline_bucket(deadline_ms):
    """Bucket deadline into urgency levels."""
    if deadline_ms < 50:
        return "very_tight(<50)"
    elif deadline_ms < 100:
        return "tight(50-100)"
    elif deadline_ms < 150:
        return "moderate(100-150)"
    else:
        return "loose(>150)"


def split_bucket(split_id):
    return f"split_{split_id}"


def server_bucket(server_id):
    return f"server_{server_id}"


def cross_analysis(results):
    """C3: Cross-tabulation of split × server accuracy."""
    # Build confusion-like matrix: teacher_split × pred_split
    split_confusion = defaultdict(lambda: defaultdict(int))
    server_confusion = defaultdict(lambda: defaultdict(int))

    for r in results:
        split_confusion[r["teacher_split"]][r["pred_split"]] += 1
        server_confusion[r["teacher_server"]][r["pred_server"]] += 1

    # Normalize rows
    split_matrix = {}
    for ts, preds in sorted(split_confusion.items()):
        total = sum(preds.values())
        split_matrix[ts] = {ps: round(c / total, 4) for ps, c in sorted(preds.items())}

    server_matrix = {}
    for ts, preds in sorted(server_confusion.items()):
        total = sum(preds.values())
        server_matrix[ts] = {ps: round(c / total, 4) for ps, c in sorted(preds.items())}

    return {"split_confusion": split_matrix, "server_confusion": server_matrix}


def analyze_mismatch_patterns(results):
    """Deep dive: what kind of mismatches are most harmful?"""
    mismatches = [r for r in results if not r["top1_hit"]]
    n_mis = len(mismatches)

    if n_mis == 0:
        return {}

    # When split matches but server doesn't
    split_ok_server_bad = [r for r in mismatches if r["split_match"] and not r["server_match"]]
    # When server matches but split doesn't
    server_ok_split_bad = [r for r in mismatches if r["server_match"] and not r["split_match"]]
    # Both wrong
    both_wrong = [r for r in mismatches if not r["split_match"] and not r["server_match"]]

    patterns = {
        "total_mismatches": n_mis,
        "split_correct_server_wrong": {
            "count": len(split_ok_server_bad),
            "fraction_of_mismatches": round(len(split_ok_server_bad) / n_mis, 4),
            "avg_regret": round(np.mean([r["regret"] for r in split_ok_server_bad]), 4) if split_ok_server_bad else 0,
            "catastrophic_rate": round(sum(1 for r in split_ok_server_bad if r["catastrophic"]) / max(len(split_ok_server_bad), 1), 4),
        },
        "server_correct_split_wrong": {
            "count": len(server_ok_split_bad),
            "fraction_of_mismatches": round(len(server_ok_split_bad) / n_mis, 4),
            "avg_regret": round(np.mean([r["regret"] for r in server_ok_split_bad]), 4) if server_ok_split_bad else 0,
            "catastrophic_rate": round(sum(1 for r in server_ok_split_bad if r["catastrophic"]) / max(len(server_ok_split_bad), 1), 4),
        },
        "both_wrong": {
            "count": len(both_wrong),
            "fraction_of_mismatches": round(len(both_wrong) / n_mis, 4),
            "avg_regret": round(np.mean([r["regret"] for r in both_wrong]), 4) if both_wrong else 0,
            "catastrophic_rate": round(sum(1 for r in both_wrong if r["catastrophic"]) / max(len(both_wrong), 1), 4),
        },
    }
    return patterns


def generate_markdown_report(overall, by_strategy, by_load, by_deadline, by_model,
                             by_split, by_server, cross_tab, mismatch_patterns, val_acc):
    """Generate human-readable markdown report."""
    lines = [
        "# Phase C: Imitation Agent Error Analysis Report",
        "",
        f"Generated: {np.datetime64('now')}",
        f"Training Val Accuracy (from checkpoint): {val_acc:.4f}",
        "",
        "---",
        "",
        "## C1: Action-Level Accuracy",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Full Action Accuracy (Top-1) | {overall['full_action_accuracy']*100:.2f}% |",
        f"| Split Accuracy | {overall['split_accuracy']*100:.2f}% |",
        f"| Server Accuracy | {overall['server_accuracy']*100:.2f}% |",
        f"| Top-3 Hit Rate | {overall['top3_hit_rate']*100:.2f}% |",
        f"| Top-5 Hit Rate | {overall['top5_hit_rate']*100:.2f}% |",
        "",
        "### Teacher Action Rank Distribution",
        "",
        "| Rank | Fraction of Samples |",
        "|------|--------------------|",
    ]
    for k, v in overall["teacher_rank_distribution"].items():
        rank = k.replace("rank_", "")
        lines.append(f"| {rank} | {v*100:.2f}% |")
    lines.append("")

    lines.extend([
        "---",
        "",
        "## C2: Regret & Catastrophic Mismatch",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Average Regret | {overall['avg_regret']:.4f} |",
        f"| Median Regret | {overall['median_regret']:.4f} |",
        f"| P95 Regret | {overall['p95_regret']:.4f} |",
        f"| Max Regret | {overall['max_regret']:.4f} |",
        f"| Catastrophic Mismatch Rate | {overall['catastrophic_mismatch_rate']*100:.2f}% |",
        f"| Avg Teacher Prob (student model) | {overall['avg_teacher_prob']:.4f} |",
        "",
    ])

    lines.extend([
        "### Mismatch Pattern Breakdown",
        "",
        "| Pattern | Count | Fraction of Mismatches | Avg Regret | Catastrophic Rate |",
        "|---------|-------|------------------------|------------|-------------------|",
    ])
    for name, data in mismatch_patterns.items():
        if isinstance(data, dict) and "count" in data:
            lines.append(
                f"| {name} | {data['count']} | {data['fraction_of_mismatches']*100:.1f}% | "
                f"{data['avg_regret']:.4f} | {data['catastrophic_rate']*100:.1f}% |"
            )
    lines.append("")

    lines.extend([
        "---",
        "",
        "## C3: Error by Bucket",
        "",
    ])

    def print_bucket_table(lines, title, bucket_data, cols=["n", "full_acc", "split_acc", "server_acc", "top3_hit", "avg_regret", "catastrophic_rate"]):
        lines.append(f"### {title}")
        lines.append("")
        header = "| Bucket | " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["-" * (len(c) + 2) for c in ["Bucket"] + cols]) + "|"
        lines.append(header)
        lines.append(sep)
        for bucket, data in sorted(bucket_data.items()):
            row = f"| {bucket} | " + " | ".join(str(round(data.get(c, 0) * 100, 2) if c.endswith("_rate") or c.endswith("_acc") or c.endswith("_hit") else round(data.get(c, 0), 4) if c != "n" else data.get(c, 0)) for c in cols) + " |"
            lines.append(row)
        lines.append("")

    print_bucket_table(lines, "By Strategy", by_strategy)
    print_bucket_table(lines, "By Load Level", by_load)
    print_bucket_table(lines, "By Deadline", by_deadline)
    print_bucket_table(lines, "By Model", by_model)
    print_bucket_table(lines, "By Teacher Split", by_split)
    print_bucket_table(lines, "By Teacher Server", by_server)

    lines.extend([
        "---",
        "",
        "## Cross-Tabulation: Teacher vs Predicted",
        "",
        "### Split Confusion Matrix (row=teacher, col=predicted)",
        "",
    ])
    split_matrix = cross_tab["split_confusion"]
    split_ids = sorted(set(list(split_matrix.keys()) + [k for v in split_matrix.values() for k in v.keys()]))
    header = "| T\\P | " + " | ".join(f"split_{s}" for s in split_ids) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["-" * 8] * (len(split_ids) + 1)) + "|")
    for ts in split_ids:
        row_data = split_matrix.get(ts, {})
        vals = [f"{row_data.get(ps, 0)*100:.1f}%" for ps in split_ids]
        lines.append(f"| split_{ts} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.extend([
        "### Server Confusion Matrix (row=teacher, col=predicted)",
        "",
    ])
    server_matrix = cross_tab["server_confusion"]
    server_ids = sorted(set(list(server_matrix.keys()) + [k for v in server_matrix.values() for k in v.keys()]))
    header = "| T\\P | " + " | ".join(f"srv_{s}" for s in server_ids) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["-" * 8] * (len(server_ids) + 1)) + "|")
    for ts in server_ids:
        row_data = server_matrix.get(ts, {})
        vals = [f"{row_data.get(ps, 0)*100:.1f}%" for ps in server_ids]
        lines.append(f"| srv_{ts} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.extend([
        "---",
        "",
        "## Summary & Diagnosis",
        "",
        "### Key Findings",
        "",
        "1. **Action Decomposition**:",
    ])

    split_acc = overall["split_accuracy"]
    server_acc = overall["server_accuracy"]
    full_acc = overall["full_action_accuracy"]
    if split_acc > server_acc:
        lines.append(f"   - Split accuracy ({split_acc*100:.1f}%) > Server accuracy ({server_acc*100:.1f}%). "
                     "The model struggles more with server selection than split point selection.")
    else:
        lines.append(f"   - Server accuracy ({server_acc*100:.1f}%) > Split accuracy ({split_acc*100:.1f}%). "
                     "The model struggles more with split point selection than server selection.")

    lines.append(f"   - Full action accuracy is {full_acc*100:.1f}%, meaning {100-full_acc*100:.1f}% of samples have at least one component wrong.")
    lines.append("")

    lines.append("2. **Top-K Analysis**:")
    lines.append(f"   - Top-3 hit rate: {overall['top3_hit_rate']*100:.1f}%. "
                 f"This means {100-overall['top3_hit_rate']*100:.1f}% of teacher actions are not even in student's top-3.")
    lines.append(f"   - Top-5 hit rate: {overall['top5_hit_rate']*100:.1f}%.")
    lines.append("")

    lines.append("3. **Regret Distribution**:")
    lines.append(f"   - Average regret: {overall['avg_regret']:.4f}")
    lines.append(f"   - P95 regret: {overall['p95_regret']:.4f}")
    lines.append(f"   - Catastrophic mismatch rate: {overall['catastrophic_mismatch_rate']*100:.1f}%")
    lines.append("")

    lines.append("4. **Mismatch Patterns**:")
    for name, data in mismatch_patterns.items():
        if isinstance(data, dict) and "count" in data and data["count"] > 0:
            lines.append(f"   - {name}: {data['count']} cases ({data['fraction_of_mismatches']*100:.1f}% of mismatches), "
                        f"avg regret {data['avg_regret']:.4f}")
    lines.append("")

    lines.extend([
        "---",
        "",
        "## Recommendations",
        "",
        "Based on this analysis:",
        "",
        "1. If split accuracy >> server accuracy: state vector needs better server-level features.",
        "2. If server accuracy >> split accuracy: DNN request features (model type, bandwidth) may be insufficient.",
        "3. If catastrophic mismatch rate is high (>10%): student is confidently wrong — consider soft-label training or larger state.",
        "4. If regret is concentrated in specific strategies/load levels: DQN reward shaping should target those scenarios.",
        "",
    ])

    return "\n".join(lines)


def main():
    print("=" * 80)
    print("PHASE C: Imitation Agent Error Analysis")
    print("=" * 80)

    # Load
    data, model, ckpt = load_data_and_model()
    states = data["states"]
    actions = data["actions"]
    metadata = data.get("metadata", [])
    num_servers = 5  # Known from architecture

    print(f"Dataset: {len(states)} samples, state_dim={states.shape[1]}, num_actions={data['num_actions']}")
    print(f"Checkpoint val accuracy: {ckpt.get('best_val_acc', 'N/A')}")

    # Deterministic split
    train_states, train_actions, train_meta, val_states, val_actions, val_meta = \
        deterministic_split(states, actions, metadata, val_frac=0.2, seed=42)

    print(f"Train: {len(train_states)}, Val: {len(val_states)}")

    # Analyze both
    print("\n[1/3] Computing metrics on TRAIN set...")
    train_results, _ = compute_metrics(model, train_states, train_actions, train_meta, num_servers)
    train_overall = aggregate_overall(train_results)

    print("[2/3] Computing metrics on VAL set...")
    val_results, val_logits = compute_metrics(model, val_states, val_actions, val_meta, num_servers)
    val_overall = aggregate_overall(val_results)

    # C3: Bucket analysis on VAL set
    print("[3/3] Bucket analysis...")
    by_strategy = bucket_analysis(val_results, lambda r: r["strategy"], "strategy")
    by_load = bucket_analysis(val_results, lambda r: r["load_level"], "load_level")
    by_deadline = bucket_analysis(val_results, lambda r: deadline_bucket(r["deadline_ms"]), "deadline")
    by_model = bucket_analysis(val_results, lambda r: r["model_name"], "model")
    by_split = bucket_analysis(val_results, lambda r: split_bucket(r["teacher_split"]), "split")
    by_server = bucket_analysis(val_results, lambda r: server_bucket(r["teacher_server"]), "server")

    # Cross-tabulation
    cross_tab = cross_analysis(val_results)

    # Mismatch patterns
    mismatch_patterns = analyze_mismatch_patterns(val_results)

    # Compile full report
    report = {
        "train_overall": train_overall,
        "val_overall": val_overall,
        "by_strategy": by_strategy,
        "by_load_level": by_load,
        "by_deadline": by_deadline,
        "by_model": by_model,
        "by_teacher_split": by_split,
        "by_teacher_server": by_server,
        "cross_tabulation": cross_tab,
        "mismatch_patterns": mismatch_patterns,
    }

    # Save JSON
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "imitation_error_analysis.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Save Markdown
    md_path = out_dir / "imitation_error_analysis.md"
    val_acc = ckpt.get("best_val_acc", 0.0)
    md_content = generate_markdown_report(
        val_overall, by_strategy, by_load, by_deadline, by_model,
        by_split, by_server, cross_tab, mismatch_patterns, val_acc
    )
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown saved: {md_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (VAL SET)")
    print("=" * 80)
    print(f"Full Action Accuracy:  {val_overall['full_action_accuracy']*100:.2f}%")
    print(f"Split Accuracy:        {val_overall['split_accuracy']*100:.2f}%")
    print(f"Server Accuracy:       {val_overall['server_accuracy']*100:.2f}%")
    print(f"Top-3 Hit Rate:        {val_overall['top3_hit_rate']*100:.2f}%")
    print(f"Top-5 Hit Rate:        {val_overall['top5_hit_rate']*100:.2f}%")
    print(f"Avg Regret:            {val_overall['avg_regret']:.4f}")
    print(f"Catastrophic Mismatch: {val_overall['catastrophic_mismatch_rate']*100:.2f}%")
    print(f"Avg Teacher Prob:      {val_overall['avg_teacher_prob']:.4f}")
    print("\n" + "=" * 80)
    print("BY STRATEGY (VAL)")
    print("=" * 80)
    for strat, d in sorted(by_strategy.items()):
        print(f"  {strat:20s}: full={d['full_acc']*100:.1f}% split={d['split_acc']*100:.1f}% srv={d['server_acc']*100:.1f}% "
              f"top3={d['top3_hit']*100:.1f}% regret={d['avg_regret']:.3f} cat={d['catastrophic_rate']*100:.1f}%")

    print("\n" + "=" * 80)
    print("BY LOAD LEVEL (VAL)")
    print("=" * 80)
    for load, d in sorted(by_load.items()):
        print(f"  {load:20s}: full={d['full_acc']*100:.1f}% split={d['split_acc']*100:.1f}% srv={d['server_acc']*100:.1f}% "
              f"top3={d['top3_hit']*100:.1f}% regret={d['avg_regret']:.3f} cat={d['catastrophic_rate']*100:.1f}%")

    print("\n" + "=" * 80)
    print("MISMATCH PATTERNS (VAL)")
    print("=" * 80)
    for name, d in mismatch_patterns.items():
        if isinstance(d, dict) and "count" in d:
            print(f"  {name:30s}: count={d['count']:4d} frac={d['fraction_of_mismatches']*100:5.1f}% "
                  f"regret={d['avg_regret']:.3f} cat={d['catastrophic_rate']*100:5.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
