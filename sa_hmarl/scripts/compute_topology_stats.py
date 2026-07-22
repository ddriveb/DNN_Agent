import json
import networkx as nx
import numpy as np
from collections import defaultdict
import sys
sys.path.insert(0, 'sa_hmarl')
from sa_hmarl.network.topology_data import TOPOLOGY_REGISTRY
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.network.ksp import get_k_shortest_paths

mod_reg = ModulationRegistry.from_profile('default')

def server_nodes_for(topo):
    if topo == 'snap24_gnutella_reach':
        return [6,1,7,12]
    return list(range(4))

def capacities_for(topo):
    if topo == 'snap24_gnutella_reach':
        return [90.0,55.0,32.0,85.0]
    return [50.0]*4

def compute(name):
    edges = TOPOLOGY_REGISTRY[name]
    G = nx.Graph()
    for u,v,l in edges:
        G.add_edge(u,v,length_km=l)
    N = G.number_of_nodes()
    E = G.number_of_edges()
    avg_deg = 2*E/N
    # shortest paths weighted km
    spl = dict(nx.shortest_path_length(G, weight='length_km'))
    pairs = [(i,j) for i in range(N) for j in range(i+1,N)]
    sp_km = [spl[i][j] for i,j in pairs]
    avg_sp_km = float(np.mean(sp_km))
    diam_km = float(np.max(sp_km))
    # hop shortest paths (unweighted)
    spl_hop = dict(nx.shortest_path_length(G))
    sp_hops = [spl_hop[i][j] for i,j in pairs]
    avg_sp_hops = float(np.mean(sp_hops))
    diam_hops = int(np.max(sp_hops))
    # betweenness
    eb = nx.edge_betweenness_centrality(G, weight='length_km')
    ebc = np.array(list(eb.values()))
    mean_ebc = float(np.mean(ebc))
    max_ebc = float(np.max(ebc))
    nb = nx.betweenness_centrality(G, weight='length_km')
    nbc = np.array(list(nb.values()))
    mean_nbc = float(np.mean(nbc))
    max_nbc = float(np.max(nbc))
    # global edge connectivity
    try:
        glob_edge_conn = nx.edge_connectivity(G)
    except Exception:
        glob_edge_conn = None
    # KSP counts
    ksp_km_counts = []
    ksp_hops_counts = []
    # limit src<dst ordered (src,dst) all ordered pairs except same
    ordered = [(i,j) for i in range(N) for j in range(N) if i!=j]
    for src,dst in ordered:
        ksp_km_counts.append(len(get_k_shortest_paths(G,src,dst,k=5,sort_by='km',weight='length_km')))
        ksp_hops_counts.append(len(get_k_shortest_paths(G,src,dst,k=50,sort_by='hops',weight='length_km')))
    ksp_km_counts = np.array(ksp_km_counts)
    ksp_hops_counts = np.array(ksp_hops_counts)
    # mod reach on shortest paths
    se_list = []
    feas_counts = []  # number of feasible mods per shortest path
    for i,j in pairs:
        length = spl[i][j]
        feas = mod_reg.feasible_for_path(length)
        feas_counts.append(len(feas))
        se_list.append(feas[0].spectral_efficiency if feas else 0.0)
    se_list = np.array(se_list)
    feas_counts = np.array(feas_counts)
    # server pairwise stats
    servers = server_nodes_for(name)
    server_pairs = [(a,b) for a in servers for b in servers if a!=b]
    srv_sp_km = [spl[a][b] for a,b in server_pairs]
    srv_ksp_km = [len(get_k_shortest_paths(G,a,b,k=5,sort_by='km',weight='length_km')) for a,b in server_pairs]
    srv_ksp_hops = [len(get_k_shortest_paths(G,a,b,k=50,sort_by='hops',weight='length_km')) for a,b in server_pairs]
    return {
        'name': name,
        'nodes': N,
        'edges': E,
        'avg_degree': round(avg_deg,2),
        'avg_sp_km': round(avg_sp_km,1),
        'diam_km': round(diam_km,1),
        'avg_sp_hops': round(avg_sp_hops,2),
        'diam_hops': diam_hops,
        'mean_edge_betweenness': round(mean_ebc,4),
        'max_edge_betweenness': round(max_ebc,4),
        'mean_node_betweenness': round(mean_nbc,4),
        'max_node_betweenness': round(max_nbc,4),
        'global_edge_connectivity': glob_edge_conn,
        'k5_km_count_mean': round(float(np.mean(ksp_km_counts)),2),
        'k5_km_count_min': int(np.min(ksp_km_counts)),
        'k50_hops_count_mean': round(float(np.mean(ksp_hops_counts)),2),
        'k50_hops_count_min': int(np.min(ksp_hops_counts)),
        'shortest_path_se_mean': round(float(np.mean(se_list)),2),
        'shortest_path_se_std': round(float(np.std(se_list)),2),
        'pct_16qam_shortest': round(float(np.mean(se_list==4.0))*100,1),
        'pct_8qam_shortest': round(float(np.mean(se_list==3.0))*100,1),
        'pct_qpsk_shortest': round(float(np.mean(se_list==2.0))*100,1),
        'pct_bpsk_shortest': round(float(np.mean(se_list==1.0))*100,1),
        'feas_mods_per_sp_mean': round(float(np.mean(feas_counts)),2),
        'server_avg_sp_km': round(float(np.mean(srv_sp_km)),1),
        'server_k5_km_count_mean': round(float(np.mean(srv_ksp_km)),2),
        'server_k50_hops_count_mean': round(float(np.mean(srv_ksp_hops)),2),
        'server_nodes': servers,
        'capacities': capacities_for(name),
    }

results = []
for name in ['snap24_gnutella_reach','xlron_cost239_ptrnet_real','xlron_german17','xlron_nsfnet_deeprmsa','xlron_jpn48']:
    print('computing', name, file=sys.stderr)
    results.append(compute(name))
print(json.dumps(results, indent=2))
