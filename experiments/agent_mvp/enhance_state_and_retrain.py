"""Phase B Quick Prototype: Enhance state vector with split profiles.

From error analysis: split_2 accuracy is only 50.8%.
Root cause: state vector only has model_id (1 float) which cannot encode
3D split differences (bw, compute, intermediate_size).

This script:
  1. Loads existing imitation dataset
  2. Appends split-profile features to each state
  3. Retrains ImitationAgent with enhanced state
  4. Evaluates closed-loop performance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import pickle
import json

from dnn_models import get_model
from train_imitation import ImitationAgent, ImitationDataset, train
from eval_imitation import ImitationAgentWrapper, build_state_vector_for_imitation
from eval_imitation import create_env, run_agent
from fixed_trace import load_trace, generate_and_save_trace
from baselines import YinLikeAgent


def build_enhanced_state(original_state, model_name):
    """Append split-profile features to a 21D state.

    Original state (21D):
      z[0:10], model_id, deadline_norm, src_norm, frag, max_free,
      server_utils[0:5], candidate_count

    Enhanced features (9D):
      For each split_id in [0,1,2]:
        bandwidth_slots / 8.0
        compute_cost
        intermediate_size_mb / 3.0
    """
    model = get_model(model_name)
    features = []
    for split in model.splits:
        features.append(split.bandwidth_slots / 8.0)
        features.append(split.compute_cost)
        features.append(split.intermediate_size_mb / 3.0)
    enhanced = np.concatenate([original_state, np.array(features, dtype=np.float32)])
    return enhanced


def enhance_dataset(dataset_path="data/imitation_dataset.pkl"):
    """Load dataset and enhance states with split profiles."""
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    states = data["states"]
    actions = data["actions"]
    metadata = data.get("metadata", [])

    print(f"Original: {states.shape}, state_dim={states.shape[1]}")

    enhanced_states = []
    for i, state in enumerate(states):
        model_name = metadata[i].get("model_name", "ResNet18") if i < len(metadata) else "ResNet18"
        enhanced = build_enhanced_state(state, model_name)
        enhanced_states.append(enhanced)

    enhanced_states = np.stack(enhanced_states)
    print(f"Enhanced: {enhanced_states.shape}, state_dim={enhanced_states.shape[1]}")

    enhanced_data = {
        "states": enhanced_states,
        "actions": actions,
        "metadata": metadata,
        "state_dim": enhanced_states.shape[1],
        "num_actions": data["num_actions"],
        "num_samples": len(enhanced_states),
    }

    out_path = Path(dataset_path).parent / "imitation_dataset_enhanced.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(enhanced_data, f)
    print(f"Enhanced dataset saved: {out_path}")

    return enhanced_data, out_path


def train_enhanced_imagent(dataset_path, epochs=50, batch_size=256, lr=1e-3, device="cpu"):
    """Train ImitationAgent on enhanced dataset."""
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    states = data["states"]
    actions = data["actions"]
    state_dim = data["state_dim"]
    num_actions = data["num_actions"]

    # Split
    n = len(states)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = ImitationDataset(states[train_idx], actions[train_idx])
    val_ds = ImitationDataset(states[val_idx], actions[val_idx])

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ImitationAgent(state_dim, num_actions, hidden_dims=(128, 128), dropout=0.1)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            optimizer.zero_grad()
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_states)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch_actions).sum().item()
            train_total += len(batch_actions)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                logits = model(batch_states)
                loss = criterion(logits, batch_actions)
                val_loss += loss.item() * len(batch_states)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == batch_actions).sum().item()
                val_total += len(batch_actions)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # Save
    out_dir = Path(dataset_path).parent.parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "state_dim": state_dim,
        "num_actions": num_actions,
        "hidden_dims": (128, 128),
        "best_val_acc": best_val_acc,
    }
    torch.save(ckpt, out_dir / "imitation_agent_enhanced.pt")
    print(f"Saved: {out_dir / 'imitation_agent_enhanced.pt'}")

    return model, best_val_acc


class EnhancedImitationAgentWrapper(ImitationAgentWrapper):
    """Wrap enhanced ImitationAgent for env evaluation."""

    def decide(self, request, network_state):
        # Build original state
        state = build_state_vector_for_imitation(request, network_state, self.encoder, self.num_servers)
        # Enhance
        state = build_enhanced_state(state, request.model_name)
        action_id = self.model.predict_action(state)

        num_srv = len(self.mec.servers)
        split_id = action_id // num_srv
        server_id = action_id % num_srv
        split_id = min(split_id, len(request.model.splits) - 1)
        server_id = min(server_id, len(self.mec.servers) - 1)
        return split_id, server_id, 0.0, {"type": "imitation_enhanced", "action_id": action_id}


def evaluate_enhanced_agent():
    """Evaluate enhanced agent on fixed traces."""
    scenarios = [
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard"),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load"),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo"),
    ]
    seeds = [42, 123, 456, 789, 2024]
    checkpoint_path = "checkpoints/imitation_agent_enhanced.pt"

    all_results = {}

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label in scenarios:
        print(f"\n{'='*80}")
        print(f"ENHANCED AGENT EVAL: {label}")
        print(f"{'='*80}")

        num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
        trace_dir = Path(__file__).parent / "traces"

        for seed in seeds:
            trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
            trace_path = trace_dir / trace_fname
            if not trace_path.exists():
                trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
            requests = load_trace(trace_path)

            # Enhanced Agent
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = EnhancedImitationAgentWrapper(checkpoint_path, num_servers=num_servers)
            agent.mec = env.mec
            agent.encoder = env.encoder
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "EnhancedImitation"), []).append(metrics)

            # YinLike baseline
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = YinLikeAgent(mec)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "YinLike"), []).append(metrics)

    # Print
    print("\n" + "=" * 100)
    print("ENHANCED IMITATION AGENT CLOSED-LOOP RESULTS (Multi-Seed)")
    print("=" * 100)
    print(f"{'Scenario':<25} {'Agent':<20} {'Blocking%':>15} {'Accept%':>15} {'Reward':>15}")
    print("-" * 100)

    for label in sorted(set(l for l, _ in all_results.keys())):
        for agent_name in ["YinLike", "EnhancedImitation"]:
            if (label, agent_name) not in all_results:
                continue
            runs = all_results[(label, agent_name)]
            br = [r["blocking_rate"] for r in runs]
            ar = [r["acceptance_rate"] for r in runs]
            rw = [r["avg_reward"] for r in runs]
            print(f"{label:<25} {agent_name:<20} {np.mean(br)*100:>7.2f}±{np.std(br)*100:<5.2f}% "
                  f"{np.mean(ar)*100:>7.2f}±{np.std(ar)*100:<5.2f}% {np.mean(rw):>8.3f}±{np.std(rw):<5.3f}")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "enhanced_imitation_eval.json", "w") as f:
        serializable = {}
        for (label, agent), runs in all_results.items():
            serializable[f"{label}__{agent}"] = [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                                   for k, v in r.items()} for r in runs]
        json.dump(serializable, f, indent=2)

    print(f"\nSaved to {out_dir / 'enhanced_imitation_eval.json'}")


def main():
    print("=" * 80)
    print("PHASE B: Enhance State Vector & Retrain")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Enhance dataset
    print("\n[1/3] Enhancing dataset with split profiles...")
    dataset_path = "data/imitation_dataset.pkl"
    enhanced_data, enhanced_path = enhance_dataset(dataset_path)

    # Step 2: Retrain
    print("\n[2/3] Retraining ImitationAgent with enhanced state...")
    model, val_acc = train_enhanced_imagent(
        enhanced_path, epochs=50, batch_size=256, lr=1e-3, device=device
    )

    # Step 3: Evaluate
    print("\n[3/3] Evaluating enhanced agent in closed loop...")
    evaluate_enhanced_agent()

    print("\nDone.")


if __name__ == "__main__":
    main()
