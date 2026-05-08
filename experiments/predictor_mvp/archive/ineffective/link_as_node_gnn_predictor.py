"""
Link-As-Node GAT Predictor, following Xu 2022 and Xiong 2024.

Topology transformation:
  Original graph: nodes = optical nodes, edges = fiber links
  Converted graph: nodes = original fiber links, edges = adjacency between links

Key idea from papers:
  - In RMSA, link-level spectrum states are more relevant than node features.
  - Xu 2022: GCN on converted topology + RNN for path aggregation.
  - Xiong 2024: GAT on converted topology to learn adaptive attention weights
    among neighboring links under spectrum continuity constraints.

Our implementation:
  - Link-as-node topology transformation (static, precomputed)
  - 6-dim link node features (spectrum + topology)
  - Multi-head GAT for message passing
  - Path-mask pooling for path-level embedding
  - Graph-mean pooling for global embedding
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single GAT layer with multi-head attention.
    Operates on converted topology where nodes = original links.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.dropout = dropout

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention parameters: one vector per head
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def _compute_attention(self, h, edge_index):
        """
        h: (N, num_heads, head_dim) or (B, N, num_heads, head_dim)
        edge_index: (E, 2)
        Returns alpha: same leading dims as h, shape (..., E, num_heads)
        """
        is_batched = h.dim() == 4
        if is_batched:
            B, N, H, D = h.shape
            src_idx = edge_index[:, 0]
            dst_idx = edge_index[:, 1]
            E = edge_index.size(0)

            # h[:, src_idx]: (B, E, H, D)
            att_i = (h[:, src_idx] * self.att_src).sum(dim=-1)  # (B, E, H)
            att_j = (h[:, dst_idx] * self.att_dst).sum(dim=-1)  # (B, E, H)
            e_scores = F.leaky_relu(att_i + att_j, negative_slope=0.2)  # (B, E, H)

            # Softmax over incoming edges per target node
            alpha = torch.zeros_like(e_scores)
            for j_node in range(N):
                mask = (dst_idx == j_node)
                if mask.sum() == 0:
                    continue
                alpha[:, mask, :] = F.softmax(e_scores[:, mask, :], dim=1)
            return alpha
        else:
            N, H, D = h.shape
            src_idx = edge_index[:, 0]
            dst_idx = edge_index[:, 1]
            E = edge_index.size(0)

            att_i = (h[src_idx] * self.att_src).sum(dim=-1)  # (E, H)
            att_j = (h[dst_idx] * self.att_dst).sum(dim=-1)  # (E, H)
            e_scores = F.leaky_relu(att_i + att_j, negative_slope=0.2)  # (E, H)

            alpha = torch.zeros_like(e_scores)
            for j_node in range(N):
                mask = (dst_idx == j_node)
                if mask.sum() == 0:
                    continue
                alpha[mask, :] = F.softmax(e_scores[mask, :], dim=0)
            return alpha

    def forward(self, x, edge_index):
        """
        Single-sample forward.
        x: (N, in_dim)
        edge_index: (E, 2)
        Returns: (N, out_dim)
        """
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.head_dim)

        alpha = self._compute_attention(h, edge_index)  # (E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        src_idx = edge_index[:, 0]
        dst_idx = edge_index[:, 1]

        # Aggregate: sum(alpha * h[src]) per dst
        msgs = alpha.unsqueeze(-1) * h[src_idx]  # (E, H, D)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        for i in range(edge_index.size(0)):
            out[dst_idx[i]] += msgs[i]

        return out.view(N, self.out_dim)

    def forward_batch(self, x, edge_index):
        """
        Batched forward. All samples share edge_index.
        x: (B, N, in_dim)
        edge_index: (E, 2)
        Returns: (B, N, out_dim)
        """
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.num_heads, self.head_dim)

        alpha = self._compute_attention(h, edge_index)  # (B, E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        src_idx = edge_index[:, 0]
        dst_idx = edge_index[:, 1]
        E = edge_index.size(0)

        msgs = alpha.unsqueeze(-1) * h[:, src_idx, :, :]  # (B, E, H, D)
        out = torch.zeros(B, N, self.num_heads, self.head_dim, device=x.device)
        for i in range(E):
            out[:, dst_idx[i], :, :] += msgs[:, i, :, :]

        return out.view(B, N, self.out_dim)


class LinkAsNodeGATPredictor(nn.Module):
    """
    Predictor using Link-As-Node + GAT + Path Pooling.

    Inputs per sample:
      - path_id, src, dst, bw: action descriptors
      - link_node_features: (num_links, link_feat_dim) — per-link spectrum stats
      - converted_edge_index: (num_converted_edges, 2) — adjacency in converted graph
      - path_mask: (num_links,) — binary mask indicating which links are in the candidate path
    """
    def __init__(self, num_links, link_feat_dim=6, link_hidden=32,
                 num_gat_layers=2, num_heads=4, dropout=0.1,
                 num_partitions=3, max_servers=128, hidden_dim=64):
        super().__init__()
        self.num_links = num_links
        self.link_hidden = link_hidden
        self.num_gat_layers = num_gat_layers

        # Input projection for link node features
        self.link_proj = nn.Linear(link_feat_dim, link_hidden)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = link_hidden if i == 0 else link_hidden
            self.gat_layers.append(
                GATLayer(in_dim, link_hidden, num_heads=num_heads, dropout=dropout)
            )

        # Layer normalization after each GAT layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(link_hidden) for _ in range(num_gat_layers)
        ])

        # Action embeddings
        self.path_embed = nn.Embedding(num_partitions, 8)
        self.bw_proj = nn.Linear(1, 8)
        self.src_embed = nn.Embedding(max_servers, 16)
        self.dst_embed = nn.Embedding(max_servers, 16)

        # MLP head
        # State = path_embedding (link_hidden) + graph_embedding (link_hidden)
        state_dim = link_hidden * 2
        action_dim = 8 + 8 + 16 + 16  # path + bw + src + dst
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [success_logit, delay_pred]
        )

    def _gnn_forward(self, x, converted_edge_index):
        """
        Run GAT layers. x can be (N, D) or (B, N, D).
        Returns updated x of same shape.
        """
        is_batched = x.dim() == 3
        for gat, norm in zip(self.gat_layers, self.norms):
            if is_batched:
                x_new = gat.forward_batch(x, converted_edge_index)
            else:
                x_new = gat(x, converted_edge_index)
            x = norm(x_new)
            x = F.relu(x)
        return x

    def forward(self, path_id, src, dst, bw,
                link_node_features, converted_edge_index, path_mask):
        """
        Single-sample forward.
        link_node_features: (num_links, link_feat_dim)
        converted_edge_index: (num_converted_edges, 2)
        path_mask: (num_links,)
        """
        # 1. Project link features
        x = self.link_proj(link_node_features)  # (num_links, link_hidden)
        x = F.relu(x)

        # 2. GAT layers
        x = self._gnn_forward(x, converted_edge_index)  # (num_links, link_hidden)

        # 3. Path-level pooling: mean over links in path
        mask_exp = path_mask.unsqueeze(-1).float()  # (num_links, 1)
        path_sum = (x * mask_exp).sum(dim=0)  # (link_hidden,)
        path_count = mask_exp.sum().clamp(min=1)
        path_emb = path_sum / path_count  # (link_hidden,)

        # 4. Graph-level pooling
        graph_emb = x.mean(dim=0)  # (link_hidden,)

        # 5. Action embeddings
        p_emb = self.path_embed(path_id)  # (8,) or (1,8)
        if p_emb.dim() == 2:
            p_emb = p_emb.squeeze(0)
        # Handle bw: could be scalar (0D) or 1D from dataloader
        if bw.dim() == 0:
            bw = bw.unsqueeze(0)
        bw_emb = self.bw_proj(bw.float().unsqueeze(-1))  # (1,8) if single sample
        if bw_emb.dim() == 2:
            bw_emb = bw_emb.squeeze(0)
        s_emb = self.src_embed(src)  # (16,) or (1,16)
        if s_emb.dim() == 2:
            s_emb = s_emb.squeeze(0)
        d_emb = self.dst_embed(dst)  # (16,) or (1,16)
        if d_emb.dim() == 2:
            d_emb = d_emb.squeeze(0)

        # 6. Predict
        state = torch.cat([path_emb, graph_emb])  # (link_hidden*2,)
        x_input = torch.cat([p_emb, bw_emb, s_emb, d_emb, state])
        out = self.mlp(x_input)
        return out[0], out[1]

    def forward_batch(self, path_id, src, dst, bw,
                      link_node_features, converted_edge_index, path_masks):
        """
        Batched forward. All samples share converted_edge_index.
        link_node_features: (B, num_links, link_feat_dim)
        path_masks: (B, num_links)
        converted_edge_index: (num_converted_edges, 2)
        path_id, src, dst, bw: (B,)
        """
        B = link_node_features.size(0)

        # 1. Project link features
        x = self.link_proj(link_node_features)  # (B, num_links, link_hidden)
        x = F.relu(x)

        # 2. GAT layers
        x = self._gnn_forward(x, converted_edge_index)  # (B, num_links, link_hidden)

        # 3. Path-level pooling
        mask_exp = path_masks.unsqueeze(-1).float()  # (B, num_links, 1)
        path_sum = (x * mask_exp).sum(dim=1)  # (B, link_hidden)
        path_count = mask_exp.sum(dim=1).clamp(min=1)  # (B, 1)
        path_emb = path_sum / path_count  # (B, link_hidden)

        # 4. Graph-level pooling
        graph_emb = x.mean(dim=1)  # (B, link_hidden)

        # 5. Action embeddings
        p_emb = self.path_embed(path_id)  # (B, 8)
        if bw.dim() == 1:
            bw = bw.unsqueeze(-1)
        bw_emb = self.bw_proj(bw.float())  # (B, 8)
        s_emb = self.src_embed(src)  # (B, 16)
        d_emb = self.dst_embed(dst)  # (B, 16)

        # 6. Predict
        state = torch.cat([path_emb, graph_emb], dim=-1)  # (B, link_hidden*2)
        x_input = torch.cat([p_emb, bw_emb, s_emb, d_emb, state], dim=-1)
        out = self.mlp(x_input)  # (B, 2)
        return out[:, 0], out[:, 1]


def compute_link_node_features(network, link_index_map):
    """
    Compute per-link node features for the converted topology.
    Features per link (6-dim):
      0. available_slot_ratio     = total_free / num_slots
      1. largest_free_block_ratio = max_consecutive_free / num_slots
      2. fragmentation_index      = 1 - max_free / total_free (0 if all free or all occupied)
      3. num_free_blocks_norm     = number of contiguous free blocks / num_slots
      4. avg_free_block_size_norm = (total_free / num_blocks) / num_slots
      5. link_length_norm         = link_length / max_link_length
    Returns: (num_links, 6) array
    """
    num_links = len(link_index_map)
    N = network.num_slots

    # Compute max link length for normalization
    max_len = 1.0
    for (u, v) in network.G.edges():
        max_len = max(max_len, network.G[u][v].get("length", 1.0))

    features = np.zeros((num_links, 6), dtype=np.float32)
    for link, idx in link_index_map.items():
        slots = network.link_states[link]
        avail = ~slots
        total_free = int(np.sum(avail))
        max_free = _max_consecutive(avail)

        # Count number of contiguous free blocks
        num_blocks = _count_blocks(avail)
        avg_block_size = total_free / num_blocks if num_blocks > 0 else 0.0

        frag = 1.0 - max_free / total_free if total_free > 0 and total_free < N else 0.0
        avail_frac = total_free / N
        block_frac = max_free / N
        num_blocks_norm = num_blocks / N
        avg_block_norm = avg_block_size / N

        u, v = link
        length_norm = network.G[u][v].get("length", 1.0) / max_len

        features[idx] = [
            avail_frac, block_frac, frag,
            num_blocks_norm, avg_block_norm, length_norm,
        ]
    return features


def build_converted_topology(network):
    """
    Build the converted topology for link-as-node transformation.
    Returns:
      link_index_map: dict[(min_u, max_v)] -> int
      converted_edge_index: np.ndarray (num_converted_edges, 2)
    """
    # 1. Enumerate original links with deterministic ordering
    links = []
    for (u, v) in network.link_states.keys():
        links.append((min(u, v), max(u, v)))
    links = sorted(links)
    link_index_map = {link: i for i, link in enumerate(links)}

    # 2. Build adjacency in converted graph: two links are adjacent if they share a node
    converted_edges = set()
    for i, (u1, v1) in enumerate(links):
        for j, (u2, v2) in enumerate(links):
            if i >= j:
                continue
            shared = (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2)
            if shared:
                converted_edges.add((i, j))
    converted_edge_index = np.array(list(converted_edges), dtype=np.int64)

    return link_index_map, converted_edge_index


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


def _count_blocks(arr):
    """Count number of contiguous True blocks."""
    if not np.any(arr):
        return 0
    diff = np.diff(np.concatenate([[False], arr, [False]]))
    return int(np.sum(diff == True))
