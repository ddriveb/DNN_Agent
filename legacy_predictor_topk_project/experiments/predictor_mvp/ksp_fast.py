"""Fast KSP implementation using graph copies (correct and simple)."""
import networkx as nx


def k_shortest_paths(G, source, target, k, weight="weight"):
    """Return up to k shortest simple paths from source to target."""
    if source == target:
        return [[source]]
    
    try:
        shortest = nx.shortest_path(G, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return []
    
    paths = [shortest]
    candidates = []
    
    for i in range(1, k):
        if i > len(paths):
            break
        prev_path = paths[i - 1]
        
        for j in range(len(prev_path) - 1):
            spur_node = prev_path[j]
            root_path = prev_path[:j + 1]
            
            # Work on a fresh copy for this spur iteration
            H = G.copy()
            
            # Remove paths that share the same root_path prefix
            for p in paths:
                if len(p) > j and p[:j + 1] == root_path:
                    u, v = p[j], p[j + 1]
                    if H.has_edge(u, v):
                        H.remove_edge(u, v)
            
            # Remove root_path nodes (except spur_node) to prevent cycles
            for node in root_path[:-1]:
                if node in H and node != spur_node:
                    H.remove_node(node)
            
            try:
                spur_path = nx.shortest_path(H, spur_node, target, weight=weight)
                total_path = root_path[:-1] + spur_path
                # Ensure simple path
                if len(total_path) == len(set(total_path)):
                    if total_path not in candidates and total_path not in paths:
                        candidates.append(total_path)
            except nx.NetworkXNoPath:
                pass
        
        if not candidates:
            break
        
        def path_length(p):
            return sum(G[u][v][weight] for u, v in zip(p[:-1], p[1:]))
        
        candidates.sort(key=path_length)
        paths.append(candidates.pop(0))
    
    return paths
