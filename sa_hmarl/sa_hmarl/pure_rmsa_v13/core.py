"""Pure RMSA action layout, features, and physical execution for v1.3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sa_hmarl.pds_rmsa.env.reservations import Reservation
from sa_hmarl.pds_rmsa.env.rmsa_env import PDSRMSAEnv
from sa_hmarl.pds_rmsa.protocol import MODULATION_TABLE, Request, required_fs

MAX_PATHS, MAX_BLOCKS = 50, 10
MODS = tuple(MODULATION_TABLE)
ACTION_DIM = MAX_PATHS * len(MODS) * MAX_BLOCKS
FEATURE_DIM = 15


@dataclass(frozen=True)
class PureAction:
    flat_index: int
    path_rank: int
    path: Tuple[int, ...]
    modulation: str
    start_slot: int
    required_fs: int


def _runs(available: np.ndarray, required: int):
    result=[]; begin=None
    for index, free in enumerate(available):
        if free and begin is None: begin=index
        if not free and begin is not None:
            if index-begin >= required: result.append((begin,index-begin))
            begin=None
    if begin is not None and len(available)-begin >= required: result.append((begin,len(available)-begin))
    return result


def _global_features(env: PDSRMSAEnv):
    bitmap=env.spectrum.as_bitmap(); occ=float(bitmap.mean()); free=~bitmap
    lfb=[]; frag=[]
    for row in free:
        runs=_runs(row,1); largest=max((size for _,size in runs),default=0); count=int(row.sum())
        lfb.append(largest / max(env.num_slots,1)); frag.append(0.0 if count in (0,env.num_slots) else 1-largest/max(count,1))
    return occ, float(np.mean(lfb)), float(np.mean(frag))


def build_action_layout(env: PDSRMSAEnv, request: Request):
    """Build fixed 2000 candidate features and a legal mask with no HT fields."""
    features=np.zeros((ACTION_DIM,FEATURE_DIM),dtype=np.float32); mask=np.zeros(ACTION_DIM,dtype=bool); actions=[None]*ACTION_DIM
    occupancy,global_lfb,global_frag=_global_features(env)
    paths=env._cached_k_paths(request.src_node,request.dst_node,MAX_PATHS)
    for p_idx,path in enumerate(paths):
        length=env.topology.path_length_km(path); hops=len(path)-1; available=env.spectrum.get_available_slots(path)
        free_ratio=float(available.mean()); path_lfb=max((s for _,s in _runs(available,1)),default=0)/max(env.num_slots,1)
        path_frag=0.0 if available.sum() in (0,env.num_slots) else 1-path_lfb/max(float(available.sum())/env.num_slots,1e-8)
        for m_idx,(name,reach,se) in enumerate(MODS):
            req=required_fs(request.bitrate_gbps,se); feasible=length<=reach
            blocks=_runs(available,req)[:MAX_BLOCKS] if feasible else []
            for b_idx in range(MAX_BLOCKS):
                flat=p_idx*(len(MODS)*MAX_BLOCKS)+m_idx*MAX_BLOCKS+b_idx
                size=blocks[b_idx][1] if b_idx<len(blocks) else 0; waste=(size-req)/size if size else 1.0
                features[flat]=[length/10000,hops/10,path_lfb,free_ratio,path_frag,se/8,reach/10000,req/10,size/50,waste,occupancy,global_lfb,global_frag,request.bitrate_gbps/100,p_idx/49]
                if b_idx<len(blocks):
                    start=blocks[b_idx][0]; mask[flat]=True; actions[flat]=PureAction(flat,p_idx,path,name,start,req)
    return features,mask,actions


def execute(env: PDSRMSAEnv, request: Request, action: Optional[PureAction]):
    """Execute an action under pure RMSA semantics."""
    if action is None or not hasattr(action, "path"):
        return {"success":False,"reason":"no_legal_action"}
    if not env.spectrum.allocate(action.path,action.start_slot,action.required_fs):
        raise AssertionError("action layout emitted a non-executable action")
    env.ledger.add(Reservation(request.request_id,action.path,action.start_slot,action.required_fs,env.time+request.holding_time))
    return {"success":True,"reason":None}
