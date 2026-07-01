"""Generate and save fixed request traces for fair cross-agent comparison."""
import pickle
from pathlib import Path
from traffic_generator import TrafficGenerator


def generate_and_save_trace(
    num_nodes: int,
    num_requests: int = 2000,
    arrival_rate: float = 5.0,
    avg_holding_time: float = 10.0,
    seed: int = 42,
    save_dir: Path = None,
) -> Path:
    """Generate a fixed request trace and save to disk."""
    if save_dir is None:
        save_dir = Path(__file__).parent / "traces"
    save_dir.mkdir(exist_ok=True)

    gen = TrafficGenerator(
        num_nodes=num_nodes,
        arrival_rate=arrival_rate,
        avg_holding_time=avg_holding_time,
        seed=seed,
    )
    gen.reset()

    requests = []
    t = 0.0
    for _ in range(num_requests):
        req = gen.next_request(t)
        requests.append(req)
        t = req.arrival_time

    fname = f"trace_n{num_nodes}_r{num_requests}_a{arrival_rate}_h{avg_holding_time}_s{seed}.pkl"
    fpath = save_dir / fname
    with open(fpath, "wb") as f:
        pickle.dump(requests, f)

    print(f"Generated {len(requests)} requests, total time={t:.2f}, saved to {fpath}")
    return fpath


def load_trace(fpath: Path):
    """Load a previously saved request trace."""
    with open(fpath, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=14)
    parser.add_argument("--num_requests", type=int, default=2000)
    parser.add_argument("--arrival_rate", type=float, default=5.0)
    parser.add_argument("--avg_holding_time", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_and_save_trace(
        num_nodes=args.num_nodes,
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        avg_holding_time=args.avg_holding_time,
        seed=args.seed,
    )
