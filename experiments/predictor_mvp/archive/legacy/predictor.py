"""PyTorch Predictor: (path_id, src, dst, bw, z) -> (success_prob, delay)
Supports unified max_servers for cross-topology transfer.
DNN-offload-aware: bw (slots required) is an explicit input.
"""
import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, state_dim, num_paths=3, max_servers=64, hidden_dim=64):
        super().__init__()
        self.max_servers = max_servers
        self.path_embed = nn.Embedding(num_paths, 8)
        self.bw_proj = nn.Linear(1, 8)          # bw in slots -> 8D embedding
        self.src_embed = nn.Embedding(max_servers, 16)
        self.dst_embed = nn.Embedding(max_servers, 16)
        action_dim = 8 + 8 + 16 + 16           # path + bw + src + dst
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, path_id, src, dst, bw, state):
        p_emb = self.path_embed(path_id)
        # bw can be [B] or [B,1]; ensure [B,1] for Linear
        if bw.dim() == 1:
            bw = bw.unsqueeze(-1)
        bw_emb = self.bw_proj(bw.float())
        s_emb = self.src_embed(src)
        d_emb = self.dst_embed(dst)
        x = torch.cat([p_emb, bw_emb, s_emb, d_emb, state], dim=-1)
        out = self.mlp(x)
        return out[:, 0], out[:, 1]
