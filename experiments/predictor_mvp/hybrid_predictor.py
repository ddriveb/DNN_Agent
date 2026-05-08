"""
Hybrid Predictor: v2b path histogram + Link-As-Node GAT path embedding.

Combines the best of both worlds:
  - v2b: handcrafted path-level spectrum statistics (strong inductive bias)
  - GAT: topology-aware link aggregation (learned neighbor relationships)

Input:
  - path_id, src, dst, bw: action descriptors
  - state (z): v2b encoded histogram
  - link_node_features, converted_edge_index, path_mask: GAT inputs

Output: (success_logit, delay_pred)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from link_as_node_gnn_predictor import GATLayer


class HybridPredictor(nn.Module):
    def __init__(self, state_dim, num_links, link_feat_dim=6, link_hidden=32,
                 num_gat_layers=2, num_heads=4, dropout=0.1,
                 num_paths=3, max_servers=128, hidden_dim=64):
        super().__init__()
        self.num_links = num_links
        self.link_hidden = link_hidden

        # GAT encoder for link-as-node topology
        self.link_proj = nn.Linear(link_feat_dim, link_hidden)
        self.gat_layers = nn.ModuleList([
            GATLayer(link_hidden if i == 0 else link_hidden, link_hidden, num_heads, dropout)
            for i in range(num_gat_layers)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(link_hidden) for _ in range(num_gat_layers)
        ])

        # Action embeddings
        self.path_embed = nn.Embedding(num_paths, 8)
        self.bw_proj = nn.Linear(1, 8)
        self.src_embed = nn.Embedding(max_servers, 16)
        self.dst_embed = nn.Embedding(max_servers, 16)

        # MLP head
        # State = v2b_state (state_dim) + path_emb (link_hidden) + graph_emb (link_hidden)
        state_dim_total = state_dim + link_hidden * 2
        action_dim = 8 + 8 + 16 + 16
        self.mlp = nn.Sequential(
            nn.Linear(state_dim_total + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _gnn_forward(self, x, converted_edge_index):
        """Run GAT layers. x can be (N, D) or (B, N, D)."""
        is_batched = x.dim() == 3
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            if is_batched:
                x_new = gat.forward_batch(x, converted_edge_index)
            else:
                x_new = gat(x, converted_edge_index)
            x = norm(x_new)
            x = F.relu(x)
        return x

    def forward(self, path_id, src, dst, bw, state,
                link_node_features, converted_edge_index, path_mask):
        """Single-sample forward."""
        # Handle batch dimension from selector (state may be [1, D])
        if state.dim() == 2 and state.size(0) == 1:
            state = state.squeeze(0)

        # 1. GAT encoding
        x = self.link_proj(link_node_features)
        x = F.relu(x)
        x = self._gnn_forward(x, converted_edge_index)

        # 2. Path-level pooling
        mask_exp = path_mask.unsqueeze(-1).float()
        path_emb = (x * mask_exp).sum(dim=0) / mask_exp.sum().clamp(min=1)

        # 3. Graph-level pooling
        graph_emb = x.mean(dim=0)

        # 4. Action embeddings
        p_emb = self.path_embed(path_id)
        if p_emb.dim() == 2:
            p_emb = p_emb.squeeze(0)
        if bw.dim() == 0:
            bw = bw.unsqueeze(0)
        bw_emb = self.bw_proj(bw.float().unsqueeze(-1))
        if bw_emb.dim() == 2:
            bw_emb = bw_emb.squeeze(0)
        s_emb = self.src_embed(src)
        if s_emb.dim() == 2:
            s_emb = s_emb.squeeze(0)
        d_emb = self.dst_embed(dst)
        if d_emb.dim() == 2:
            d_emb = d_emb.squeeze(0)

        # 5. Predict
        state_combined = torch.cat([state, path_emb, graph_emb])
        x_input = torch.cat([p_emb, bw_emb, s_emb, d_emb, state_combined])
        out = self.mlp(x_input)
        return out[0], out[1]

    def forward_batch(self, path_id, src, dst, bw, state,
                      link_node_features, converted_edge_index, path_masks):
        """Batched forward."""
        B = link_node_features.size(0)

        # 1. GAT encoding
        x = self.link_proj(link_node_features)
        x = F.relu(x)
        x = self._gnn_forward(x, converted_edge_index)

        # 2. Path-level pooling
        mask_exp = path_masks.unsqueeze(-1).float()
        path_emb = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        # 3. Graph-level pooling
        graph_emb = x.mean(dim=1)

        # 4. Action embeddings
        p_emb = self.path_embed(path_id)
        s_emb = self.src_embed(src)
        d_emb = self.dst_embed(dst)
        if bw.dim() == 1:
            bw = bw.unsqueeze(-1)
        bw_emb = self.bw_proj(bw.float())

        # 5. Predict
        state_combined = torch.cat([state, path_emb, graph_emb], dim=-1)
        x_input = torch.cat([p_emb, bw_emb, s_emb, d_emb, state_combined], dim=-1)
        out = self.mlp(x_input)
        return out[:, 0], out[:, 1]
