#!/usr/bin/env python3
"""Train/evaluate a from-scratch pure-RMSA masked PPO-R proposer."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from sa_hmarl.agents.ppo_agents import _MaskedPPOBase
from sa_hmarl.pds_rmsa.baselines.ksp_ff import ksp_ff_action
from sa_hmarl.pds_rmsa.env.rmsa_env import PDSRMSAEnv
from sa_hmarl.pds_rmsa.env.topology import load_topology
from sa_hmarl.pds_rmsa.env.traffic_trace import generate_trace
from sa_hmarl.pds_rmsa.training.trainers import LOAD_ERLANG, NUM_NODES, NUM_SLOTS, TOPOLOGY
from sa_hmarl.pure_rmsa_v13.core import FEATURE_DIM, build_action_layout, execute

PROTOCOL = "pure_rmsa_strict_v13_proposer_ppo_v1"


def make_env(
    seed: int,
    requests: int,
    topology=TOPOLOGY,
    num_slots=NUM_SLOTS,
    load_erlang=LOAD_ERLANG,
    *,
    k_paths: int = 50,
    holding_truncation: float | None = None,
    path_sort_strategy: str = "hops",
):
    num_nodes=load_topology(topology).num_nodes
    trace=generate_trace(topology,num_nodes,requests,load_erlang,seed,
                         num_slots=num_slots,
                         holding_truncation=holding_truncation)
    return PDSRMSAEnv(
        topology,
        num_slots,
        load_erlang,
        seed,
        trace=trace,
        k_paths=k_paths,
        path_sort_strategy=path_sort_strategy,
    )


def evaluate(agent, seeds, warmup, eval_requests, topology, num_slots, load_erlang):
    rows=[]
    for seed in seeds:
        for method in ("ppo_r_top1","ksp_ff_k50_hops"):
            env=make_env(seed,warmup+eval_requests,topology,num_slots,load_erlang); blocked=0; admitted=0
            action_metrics={"path_rank":[],"path_hops":[],"path_km":[],"block_start":[],"required_fs":[]}; modulation_counts={}
            for index,request in enumerate(env.trace.requests):
                env.advance_external(request)
                if method=="ppo_r_top1":
                    features,mask,actions=build_action_layout(env,request)
                    choice,_,_=agent.select_from_features(features,mask,deterministic=True)
                    selected_action=actions[choice] if choice is not None else None
                    result=execute(env,request,selected_action)
                else:
                    selected_action=ksp_ff_action(env,request)
                    result=execute(env,request,selected_action)
                if index>=warmup:
                    admitted += int(result["success"]); blocked += int(not result["success"])
                    if result["success"]:
                        action=selected_action
                        action_metrics["path_rank"].append(action.path_rank)
                        action_metrics["path_hops"].append(len(action.path)-1)
                        action_metrics["path_km"].append(env.topology.path_length_km(action.path))
                        action_metrics["block_start"].append(action.start_slot)
                        action_metrics["required_fs"].append(action.required_fs)
                        modulation_counts[action.modulation]=modulation_counts.get(action.modulation,0)+1
            rows.append({"seed":seed,"method":method,"blocking":blocked/eval_requests,"blocked":blocked,"admitted":admitted,**{f"avg_{name}":float(np.mean(values)) if values else None for name,values in action_metrics.items()},"modulation_counts":modulation_counts})
    summary={}
    for method in ("ppo_r_top1","ksp_ff_k50_hops"):
        vals=[r["blocking"] for r in rows if r["method"]==method]
        summary[method]={"mean_blocking":float(np.mean(vals)),"std_blocking":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}
    return rows,summary


def behavior_clone_ksp(agent, seed, decisions, topology=TOPOLOGY, num_slots=NUM_SLOTS, load_erlang=LOAD_ERLANG, batch_size=128, epochs=1):
    """Warm-start the proposer on KSP-FF K=50 actions; no ranker is involved."""
    if decisions <= 0: return []
    env=make_env(seed,decisions,topology,num_slots,load_erlang); samples=[]; logs=[]
    for request in env.trace.requests[:decisions]:
        env.advance_external(request)
        features,mask,actions=build_action_layout(env,request)
        target=ksp_ff_action(env,request)
        target_index=next((i for i,a in enumerate(actions) if a is not None and a.path==target.path and a.modulation==target.modulation and a.start_slot==target.start_slot and a.required_fs==target.required_fs),None) if hasattr(target,"path") else None
        execute(env,request,target)
        if target_index is None: continue
        samples.append((features,mask,target_index))
    if not samples: return logs
    rng=np.random.default_rng(seed)
    for _ in range(epochs):
        permutation=rng.permutation(len(samples))
        for start in range(0,len(samples),batch_size):
            indices=permutation[start:start+batch_size]
            batch=[samples[int(i)] for i in indices]
            ft,mt=agent._pad_features_masks([x[0] for x in batch],[x[1] for x in batch])
            labels=torch.tensor([x[2] for x in batch],dtype=torch.long,device=agent.device)
            logits=agent.policy_net(ft).masked_fill(~mt,-1e9)
            loss=torch.nn.functional.cross_entropy(logits,labels)
            agent.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(),.5); agent.optimizer.step()
            logs.append(float(loss.item()))
    return logs


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="sa_hmarl/pure_rmsa_v13/artifacts/proposer_smoke")
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=6101)
    p.add_argument("--decisions",type=int,default=30000)
    p.add_argument("--rollout",type=int,default=256)
    p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--bc-decisions",type=int,default=0)
    p.add_argument("--bc-epochs",type=int,default=1)
    p.add_argument("--topology",default=TOPOLOGY)
    p.add_argument("--num-slots",type=int,default=NUM_SLOTS)
    p.add_argument("--load-erlang",type=float,default=LOAD_ERLANG)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    if args.smoke: args.decisions=2000; args.rollout=128; args.epochs=2
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    agent=_MaskedPPOBase(FEATURE_DIM,hidden_dims=(128,64),lr=3e-4,entropy_coef=.01,max_grad_norm=.5,device=args.device,activation="silu")
    bc_losses=behavior_clone_ksp(agent,args.seed+100000,args.bc_decisions,args.topology,args.num_slots,args.load_erlang,epochs=args.bc_epochs)
    env=make_env(args.seed,args.decisions+1,args.topology,args.num_slots,args.load_erlang); trajectory=[]; updates=[]
    for index,request in enumerate(env.trace.requests[:args.decisions]):
        env.advance_external(request)
        features,mask,actions=build_action_layout(env,request)
        choice,logp,_=agent.select_from_features(features,mask,deterministic=False)
        result=execute(env,request,actions[choice] if choice is not None else None)
        trajectory.append((features,mask,int(choice) if choice is not None else 0,float(logp),1.0 if result["success"] else 0.0))
        if len(trajectory)==args.rollout or index+1==args.decisions:
            rewards=np.asarray([x[4] for x in trajectory],dtype=np.float32)
            returns=np.zeros_like(rewards); running=0.0
            for j in range(len(rewards)-1,-1,-1): running=rewards[j]+.99*running; returns[j]=running
            advantages=(returns-returns.mean())/(returns.std()+1e-8)
            loss,entropy,kl,_=agent.optimize_ppo([x[0] for x in trajectory],[x[1] for x in trajectory],[x[2] for x in trajectory],[x[3] for x in trajectory],advantages,.2,epochs=args.epochs,target_kl=.03)
            updates.append({"decision":index+1,"reward_mean":float(rewards.mean()),"policy_loss":loss,"entropy":entropy,"approx_kl":kl})
            trajectory=[]
    eval_seeds=(8101,) if args.smoke else (8101,8102,8103,8104,8105)
    eval_requests=1000 if args.smoke else 10000; warmup=200 if args.smoke else 1000
    per_seed,summary=evaluate(agent,eval_seeds,warmup,eval_requests,args.topology,args.num_slots,args.load_erlang)
    checkpoint={"protocol":PROTOCOL,"feature_dim":FEATURE_DIM,"topology":args.topology,"num_slots":args.num_slots,"load_erlang":args.load_erlang,"k_paths":50,"path_order":"hops->km->tuple","max_blocks":10,"modulations":4,"holding_time_input":False,"state_dict":agent.policy_net.state_dict()}
    torch.save(checkpoint,out/"ppo_r_proposer.pt")
    (out/"TRAINING.json").write_text(json.dumps({"protocol":PROTOCOL,"config":vars(args),"bc_loss_mean":float(np.mean(bc_losses)) if bc_losses else None,"bc_updates":len(bc_losses),"updates":updates,"evaluation":summary,"per_seed":per_seed},indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
