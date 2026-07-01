cslots}, "
          f"Slot BW: {args.slot_bw_hz/1e9:.2f} GHz, Guard: {args.guard_band_fs}")
    print(f"Arrival interval: {args.arrival_interval}, Waste coef: {args.waste_coef}")
    print(f"Target update freq: {args.target_update_freq}, Batch size: {args.batch_size}")
    print("=" * 70)

    for episode in range(args.episodes):
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env, rng, src, args.requests_per_episode,
            arrival_interval=args.arrival_interval,
            holding_min=args.holding_min,
            holding_max=args.holding_max,
            deadline_min=args.deadline_min,
            deadline_max=args.deadline_max,
            size_min_mb=args.size_min_mb,
            size_max_mb=args.size_max_mb,
            edge_cost_min=args.edge_cost_min,
            edge_cost_max=args.edge_cost_max,
            num_splits=args.num_splits,
        )
        env.reset(requests)

        episode_reward = 0.0
        episode_blocked = 0
        episode_fs_values = []
        reason_counter = Counter()
        split_counter = Counter()
        server_counter = Counter()

        for t, req in enumerate(requests):
            obs_c = build_agent_c_observation(env, req)
            action_idx_c = agent.select_action(obs_c)

            if action_idx_c is None:
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))

            split_id, server_id = action_c
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = ksp_bf_action(obs_r)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                )

            _, _, done, info = env.step(action_c, action_r)
            server = env.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )
            episode_reward += reward

            if info.get("success", False):
                episode_fs_values.append(info.get("num_slots", 0))
            else:
                episode_blocked += 1
                reason = info.get("reason", "unknown")
                reason_counter[reason] += 1

            split_counter[split_id] += 1
            server_counter[server_id] += 1

            # Build real next_obs from the next request (or terminal dummy)
            action_features, mask = agent.build_action_features(obs_c)
            if t + 1 < len(requests):
                next_req = requests[t + 1]
                next_obs_c = build_agent_c_observation(env, next_req)
                next_action_features, next_mask = agent.build_action_features(next_obs_c)
                transition_done = False
            else:
                next_action_features = np.zeros_like(action_features)
                next_mask = np.zeros_like(mask)
                transition_done = True

            replay_buffer.push(
                action_features, mask,
                action_idx_c if action_idx_c is not None else 0,
                reward,
                next_action_features, next_mask,
                transition_done,
            )

            # Optimize
            if len(replay_buffer) >= args.learning_starts and t % 2 == 0:
                batch = replay_buffer.sample(args.batch_size)
                if batch is not None:
                    loss = agent.optimize(batch, args.batch_size)
                    if loss is not None:
                        metrics["losses"].append(loss)

            # Target network update
            if agent.step_count > 0 and agent.step_count % args.target_update_freq == 0:
                agent.update_target()

        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        metrics["episode_rewards"].append(episode_reward / args.requests_per_episode)
        metrics["episode_blocking"].append(episode_blocked / args.requests_per_episode)

        # Periodic evaluation + diagnostics
        if episode % 100 == 0 or episode == args.episodes - 1:
            eval_m = evaluate_agent(env, agent, eval_requests, args.waste_coef)
            comp_m = evaluate_baseline(env, eval_requests, args.waste_coef, "Compute-Greedy")
            spec_m = evaluate_baseline(env, eval_requests, args.waste_coef, "Spectrum-Greedy")
            rand_m = evaluate_baseline(env, eval_requests, args.waste_coef, "Random-valid-C")
            metrics["eval_rewards"].append(eval_m["avg_reward"])
            metrics["eval_blocking"].append(eval_m["blocking_rate"])

            # Best checkpoint tracking
            if eval_m["avg_reward"] > best_eval_reward:
                best_eval_reward = eval_m["avg_reward"]
                best_episode = episode
                torch.save({
                    "model_state": agent.q_net.state_dict(),
                    "target_state": agent.target_net.state_dict(),
                    "input_dim": agent.input_dim,
                    "hidden_dims": (128, 64),
                    "gamma": agent.gamma,
                    "seed": args.seed,
                    "training_metrics": metrics,
                    "args": vars(args),
                    "best_eval_reward": best_eval_reward,
                    "best_episode": best_episode,
                }, str(ckpt_best))

            avg_fs = np.mean(episode_fs_values) if episode_fs_values else 0.0
            reason_str = ", ".join(f"{k}={v}" for k, v in reason_counter.most_common(5))
            split_str = ", ".join(f"s{k}={v}" for k, v in sorted(split_counter.items()))
            server_str = ", ".join(f"sv{k}={v}" for k, v in sorted(server_counter.items()))

            print(
                f"Ep {episode:4d} | "
                f"train_r={metrics['episode_rewards'][-1]:+.3f} "
                f"train_blk={metrics['episode_blocking'][-1]:.2f} "
                f"avg_fs={avg_fs:.1f} "
                f"eps={agent.epsilon:.3f} | "
                f"eval_r={eval_m['avg_reward']:+.3f} "
                f"eval_blk={eval_m['blocking_rate']:.2f} | "
                f"comp={comp_m['avg_reward']:+.3f}/{comp_m['blocking_rate']:.2f} "
                f"spec={spec_m['avg_reward']:+.3f}/{spec_m['blocking_rate']:.2f} "
                f"rand={rand_m['avg_reward']:+.3f}/{rand_m['blocking_rate']:.2f} "
                f"| best={best_eval_reward:+.3f}@{best_episode}"
            )
            if reason_str:
                print(f"         Reasons: {reason_str}")
            print(f"         Splits: {split_str} | Servers: {server_str}")

    # Final summary
    print("=" * 70)
    print("Final vs Baselines (same eval set):")
    final_eval = evaluate_agent(env, agent, eval_requests, args.waste_coef)
    final_comp = evaluate_baseline(env, eval_requests, args.waste_coef, "Compute-Greedy")
    final_spec = evaluate_baseline(env, eval_requests, args.waste_coef, "Spectrum-Greedy")
    final_rand = evaluate_baseline(env, eval_requests, args.waste_coef, "Random-valid-C")
    print(f"  Agent-C:   reward={final_eval['avg_reward']:+.3f}, "
          f"blocking={final_eval['blocking_rate']:.2f}, "
          f"avg_waste={final_eval['avg_waste']:.3f}")
    print(f"  Compute-G: reward={final_comp['avg_reward']:+.3f}, "
          f"blocking={final_comp['blocking_rate']:.2f}, "
          f"avg_waste={final_comp['avg_waste']:.3f}")
    print(f"  Spectrum-G: reward={final_spec['avg_reward']:+.3f}, "
          f"blocking={final_spec['blocking_rate']:.2f}, "
          f"avg_waste={final_spec['avg_waste']:.3f}")
    print(f"  Random-C:  reward={final_rand['avg_reward']:+.3f}, "
          f"blocking={final_rand['blocking_rate']:.2f}, "
          f"avg_waste={final_rand['avg_waste']:.3f}")
    print(f"\nBest checkpoint: reward={best_eval_reward:+.3f} at episode {best_episode}")
    print("=" * 70)

    # Save last checkpoint
    torch.save({
        "model_state": agent.q_net.state_dict(),
        "target_state": agent.target_net.state_dict(),
        "input_dim": agent.input_dim,
        "hidden_dims": (128, 64),
        "gamma": agent.gamma,
        "seed": args.seed,
        "training_metrics": metrics,
        "args": vars(args),
        "best_eval_reward": best_eval_reward,
        "best_episode": best_episode,
    }, str(ckpt_last))
    print(f"Last checkpoint saved to {ckpt_last}")
    if ckpt_best.exists():
        print(f"Best checkpoint saved to {ckpt_best}")

    return agent, metrics


def main():
    parser = argparse.ArgumentParser(description="Agent-C DQN ablation training (no spectrum)")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--requests_per_episode", type=int, default=30)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=100)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    