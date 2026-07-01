"""Smoke test for event-driven Yin 2024 simulation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_mvp"))

import numpy as np

from yin2024_env import create_yin2024_env
from yin2024_requests import generate_requests
from yin_baselines import WOAgent, DFAgent, RFAgent, IWDApproxAgent
from baselines import YinLikeAgent


def smoke_test():
    print("=" * 60)
    print("SMOKE TEST: Event-driven Yin 2024 simulation")
    print("=" * 60)

    # 1. Generate requests
    server_nodes = [0, 3, 5, 6, 8]
    requests = generate_requests(15, server_nodes, seed=42,
                                  arrival_rate=5.0, avg_holding_time=5.0)
    print(f"\nGenerated {len(requests)} requests")
    print(f"  Arrival times: {[f'{r.arrival_time:.2f}' for r in requests[:5]]} ...")
    print(f"  Holding times: {[f'{r.holding_time:.2f}' for r in requests[:5]]} ...")

    # Check compute magnitude
    first_compute = requests[0].model.splits[0].compute_cost
    print(f"  First request compute_cost: {first_compute:.2e}")
    assert first_compute >= 0.5, f"compute_cost too small: {first_compute}"

    # 2. Create environment
    env, encoder, mec = create_yin2024_env("net1", frag_level=0.2, seed=42)
    print(f"\nEnv created: {env.net.num_slots} slots, {len(mec.servers)} servers")

    # 3. Run WO agent (event-driven)
    agent = WOAgent(env.mec)
    env.reset()

    print("\n--- WO Agent (event-driven) ---")
    for req in sorted(requests, key=lambda r: r.arrival_time):
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        print(f"  t={env.time:.2f} req={req.req_id} split={split_id} srv={server_id} "
              f"success={result.success} reward={result.reward:.3f} "
              f"active_conns={len(env.active_connections)}")

    metrics = env.get_metrics()
    print(f"\nWO metrics: block={metrics['blocking_rate']*100:.1f}% "
          f"accept={metrics['acceptance_rate']*100:.1f}%")

    # 4. Run DF agent (event-driven)
    env, encoder, mec = create_yin2024_env("net1", frag_level=0.2, seed=42)
    agent = DFAgent(env.mec)
    env.reset()

    print("\n--- DF Agent (event-driven) ---")
    for req in sorted(requests, key=lambda r: r.arrival_time):
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        if req.req_id <= 5:
            print(f"  t={env.time:.2f} req={req.req_id} split={split_id} srv={server_id} "
                  f"success={result.success} reward={result.reward:.3f}")

    metrics = env.get_metrics()
    print(f"\nDF metrics: block={metrics['blocking_rate']*100:.1f}% "
          f"accept={metrics['acceptance_rate']*100:.1f}%")

    # 5. Verify advance_time releases resources
    env, encoder, mec = create_yin2024_env("net1", frag_level=0.2, seed=42)
    agent = WOAgent(env.mec)
    env.reset()

    # Process first 5 requests
    for req in sorted(requests[:5], key=lambda r: r.arrival_time):
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        env.step(req, split_id, server_id)

    active_before = len(env.active_connections)
    print(f"\n--- Resource release check ---")
    print(f"  Active connections after 5 requests: {active_before}")

    # Advance time far into the future to force all releases
    env.advance_time(env.time + 1000.0)
    active_after = len(env.active_connections)
    print(f"  Active connections after +1000s: {active_after}")
    assert active_after == 0, f"Expected 0 active connections, got {active_after}"

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    smoke_test()
