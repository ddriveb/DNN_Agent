"""
GNN-based predictor for topology-agnostic routing prediction.
End-to-end trainable: GNN processes graph structure + edge spectrum state.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNPredictor(nn.Module):
    """
    Graph Neural Network predictor.
    Input per sample: (partition, src, dst, edge_features, edge_index, num_nodes)
    - edge_features: (num_edges, edge_feat_dim) — per-edge spectrum stats
    - edge_index: (num_edges, 2) — undirected edge list
    - num_nodes: actual number of nodes in this topology
    """
    def __init__(self, max_nodes=128, edge_feat_dim=3, node_hidden=16,
                 num_partitions=3, max_servers=128, hidden_dim=64):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_hidden = node_hidden
        self.max_servers = max_servers

        # Static node embeddings (topology-agnostic initialization)
        self.node_emb = nn.Embedding(max_nodes, node_hidden)

        # Edge feature projection
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, node_hidden),
            nn.ReLU(),
        )

        # GNN layers: concat(self, neighbor_agg) -> new self
        self.gnn1 = nn.Linear(node_hidden * 2, node_hidden)
        self.gnn2 = nn.Linear(node_hidden * 2, node_hidden)

        # Action embeddings
        self.partition_embed = nn.Embedding(num_partitions, 8)
        self.src_embed = nn.Embedding(max_servers, 16)
        self.dst_embed = nn.Embedding(max_servers, 16)

        # Predictor head
        action_dim = 8 + 16 + 16
        state_dim = node_hidden * 3  # src + dst + graph
        self.mlp = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def _message_passing(self, x, edge_features, edge_index, num_nodes):
        """
        x: (num_nodes, node_hidden)
        edge_features: (num_edges, edge_feat_dim)
        edge_index: (num_edges, 2)
        Returns updated node features.
        """
        num_edges = edge_index.size(0)

        # Edge messages
        edge_msg = self.edge_proj(edge_features)  # (E, H)

        # Aggregate neighbor messages
        neighbor_agg = torch.zeros(num_nodes, self.node_hidden, device=x.device)

        src = edge_index[:, 0]  # (E,)
        dst = edge_index[:, 1]  # (E,)

        # Message from dst to src
        neighbor_agg.index_add_(0, src, edge_msg * x[dst])
        # Message from src to dst
        neighbor_agg.index_add_(0, dst, edge_msg * x[src])

        # Normalize by degree
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, src, torch.ones(num_edges, device=x.device))
        degree.index_add_(0, dst, torch.ones(num_edges, device=x.device))
        neighbor_agg = neighbor_agg / degree.clamp(min=1).unsqueeze(-1)

        # Update
        x_new = self.gnn1(torch.cat([x, neighbor_agg], dim=-1))
        x_new = F.relu(x_new)
        return x_new

    def forward(self, partition, src, dst, edge_features, edge_index, num_nodes):
        """
        Single-sample forward.
        """
        # 1. Initialize node features
        node_ids = torch.arange(num_nodes, device=edge_features.device)
        x = self.node_emb(node_ids)  # (N, H)

        # 2. Message passing
        x = self._message_passing(x, edge_features, edge_index, num_nodes)
        x = self._message_passing(x, edge_features, edge_index, num_nodes)

        # 3. Extract embeddings
        src_emb = x[src]  # (H,)
        dst_emb = x[dst]  # (H,)
        graph_emb = x.mean(dim=0)  # (H,)
        state = torch.cat([src_emb, dst_emb, graph_emb])  # (3H,)

        # 4. Action embeddings
        p_emb = self.partition_embed(partition)
        s_emb = self.src_embed(src)
        d_emb = self.dst_embed(dst)

        # 5. Predict
        x_input = torch.cat([p_emb, s_emb, d_emb, state])
        out = self.mlp(x_input)
        return out[0], out[1]  # success_logit, delay_pred

    def forward_batch(self, partition, src, dst, edge_features, edge_index, num_nodes):
        """
        Batched forward. All samples in batch must share the same edge_index and num_nodes.
        edge_features: (B, E, edge_feat_dim)
        partition, src, dst: (B,)
        edge_index: (E, 2)
        num_nodes: int
        """
        B = edge_features.size(0)
        E = edge_index.size(0)
        device = edge_features.device

        # 1. Initialize node features: (B, N, H)
        node_ids = torch.arange(num_nodes, device=device)
        x = self.node_emb(node_ids).unsqueeze(0).expand(B, -1, -1)

        # 2. Edge messages: (B, E, H)
        edge_msg = self.edge_proj(edge_features)

        src_nodes = edge_index[:, 0]
        dst_nodes = edge_index[:, 1]

        # 3. Message passing layers
        for gnn_layer in [self.gnn1, self.gnn2]:
            neighbor_agg = torch.zeros(B, num_nodes, self.node_hidden, device=device)

            # Vectorized message accumulation using index_add per edge
            # x[:, dst_nodes] is (B, E, H); edge_msg is (B, E, H)
            msgs_to_src = edge_msg * x[:, dst_nodes]
            msgs_to_dst = edge_msg * x[:, src_nodes]

            for e in range(E):
                neighbor_agg[:, src_nodes[e]] += msgs_to_src[:, e]
                neighbor_agg[:, dst_nodes[e]] += msgs_to_dst[:, e]

            # Normalize by degree
            degree = torch.zeros(num_nodes, device=device)
            degree.index_add_(0, src_nodes, torch.ones(E, device=device))
            degree.index_add_(0, dst_nodes, torch.ones(E, device=device))
            neighbor_agg = neighbor_agg / degree.clamp(min=1).unsqueeze(0).unsqueeze(-1)

            # Update
            x = gnn_layer(torch.cat([x, neighbor_agg], dim=-1))
            x = F.relu(x)

        # 4. Extract embeddings
        batch_idx = torch.arange(B, device=device)
        src_emb = x[batch_idx, src]  # (B, H)
        dst_emb = x[batch_idx, dst]  # (B, H)
        graph_emb = x.mean(dim=1)  # (B, H)
        state = torch.cat([src_emb, dst_emb, graph_emb], dim=-1)  # (B, 3H)

        # 5. Action embeddings
        p_emb = self.partition_embed(partition)  # (B, 8)
        s_emb = self.src_embed(src)  # (B, 16)
        d_emb = self.dst_embed(dst)  # (B, 16)

        x_input = torch.cat([p_emb, s_emb, d_emb, state], dim=-1)  # (B, action+state)
        out = self.mlp(x_input)  # (B, 2)
        return out[:, 0], out[:, 1]


def compute_edge_features(network):
    """
    Compute per-edge spectrum features for all edges in the network.
    Returns: edge_features (num_edges, 3), edge_index (num_edges, 2)
    Features per edge: [avail_frac, max_block_frac, frag_index]
    """
    edges = []
    features = []
    N = network.num_slots
    for (u, v), slots in network.link_states.items():
        avail = ~slots
        total_free = int(np.sum(avail))
        max_free = _max_consecutive(avail)
        frag = 1.0 - max_free / total_free if total_free > 0 and total_free < N else 0.0
        avail_frac = total_free / N
        block_frac = max_free / N
        edges.append([u, v])
        features.append([avail_frac, block_frac, frag])
    return np.array(features, dtype=np.float32), np.array(edges, dtype=np.int64)


def _max_consecutive(arr):
    if not np.any(arr):
        return 0
    max_len = curr = 0
    for v in arr:
        if v:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 0
    return max_len
