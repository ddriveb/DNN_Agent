"""PyTorch Predictor: (partition, src, dst, z) -> (success_prob, delay)
Supports unified max_servers for cross-topology transfer.
"""
import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, state_dim, num_partitions=3, max_servers=64, hidden_dim=64):
        super().__init__()
        self.max_servers = max_servers
        self.partition_embed = nn.Embedding(num_partitions, 8)
        self.src_embed = nn.Embedding(max_servers, 16)
        self.dst_embed = nn.Embedding(max_servers, 16)
        action_dim = 8 + 16 + 16
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, partition, src, dst, state):
        p_emb = self.partition_embed(partition)
        s_emb = self.src_embed(src)
        d_emb = self.dst_embed(dst)
        x = torch.cat([p_emb, s_emb, d_emb, state], dim=-1)
        out = self.mlp(x)
        return out[:, 0], out[:, 1]
