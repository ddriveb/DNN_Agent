from sa_hmarl.pds_rmsa.env.rmsa_env import PDSRMSAEnv
from sa_hmarl.pds_rmsa.env.traffic_trace import generate_trace
from sa_hmarl.pure_rmsa_v13.core import ACTION_DIM, build_action_layout, execute


def test_pure_layout_is_fixed_masked_and_executable():
    trace=generate_trace("tiny_ring3",3,2,10.0,9,num_slots=8)
    env=PDSRMSAEnv("tiny_ring3",8,10.0,9,trace=trace,k_paths=50,path_sort_strategy="hops")
    req=trace.requests[0]; env.advance_external(req)
    features,mask,actions=build_action_layout(env,req)
    assert features.shape[0] == ACTION_DIM == len(mask) == len(actions)
    index=int(mask.nonzero()[0][0]); assert actions[index] is not None
    assert execute(env,req,actions[index])["success"]
