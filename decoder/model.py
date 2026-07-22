import math

import torch.nn as nn
import torch

N_Q_HEADS = 32
N_KV_HEADS = 8
HIDDEN_SIZE = 768
HEAD_DIM = 24


class SelfAttn(nn.Module):
    def __init__(
        self, n_q_heads: int, n_kv_heads: int, hidden_size: int, head_dim: int
    ):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "N_KV_HEADS must divide N_Q_HEADS"

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        # params
        self.wq = nn.Linear(hidden_size, n_q_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)

        self.wo = nn.Linear(n_q_heads * head_dim, hidden_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_TOKENS, HIDDEN_SIZE]
        n_batches = x.shape[0]
        n_tokens = x.shape[1]

        # Q: [B, N_TOKENS, N_Q_HEADS * HEAD_DIM]
        # K: [B, N_TOKENS, N_KV_HEADS * HEAD_DIM]
        # V: [B, N_TOKENS, N_KV_HEADS * HEAD_DIM]
        Q: torch.Tensor = self.wq(x)
        K: torch.Tensor = self.wk(x)
        V: torch.Tensor = self.wv(x)

        # note: could use interleave for GQA but it materializes a copy
        # instead, reshape Q so that broadcasting happens
        n_reps = self.n_q_heads // self.n_kv_heads
        Q = Q.view(n_batches, n_tokens, self.n_kv_heads, n_reps, self.head_dim).permute(
            0, 2, 3, 1, 4
        )  # [B, N_KV_HEADS, N_REPS, N_TOKENS, HEAD_DIM]
        K = K.view(n_batches, n_tokens, self.n_kv_heads, 1, self.head_dim).permute(
            0, 2, 3, 1, 4
        )  # [B, N_KV_HEADS, 1, N_TOKENS, HEAD_DIM]
        V = V.view(n_batches, n_tokens, self.n_kv_heads, 1, self.head_dim).permute(
            0, 2, 3, 1, 4
        )  # [B, N_KV_HEADS, 1, N_TOKENS, HEAD_DIM]

        attn = torch.matmul(
            Q, K.transpose(-2, -1)
        )  # [B, N_KV_HEADS, N_REPS, N_TOKENS, N_TOKENS]
        attn = attn / math.sqrt(self.head_dim)
        mask = torch.tril(
            torch.ones(n_tokens, n_tokens, device=x.device, dtype=torch.bool)
        )
        masked_attn = attn.masked_fill(~mask, float("-inf"))
        masked_attn = self.softmax(masked_attn)

        out_vals = torch.matmul(
            masked_attn, V
        )  # [B, N_KV_HEADS, N_REPS, N_TOKENS, HEAD_DIM]
        out_vals = (
            out_vals.view(n_batches, self.n_q_heads, n_tokens, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # [B, N_TOKENS, N_Q_HEADS, HEAD_DIM]
        out_vals = out_vals.view(
            n_batches, n_tokens, self.n_q_heads * self.head_dim
        )  # [B, N_TOKENS, N_Q_HEADS * HEAD_DIM]
        out = self.wo(out_vals)  # [B, N_TOKENS, HIDDEN_SIZE]
        return out


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ff_size: int):
        super().__init__()
        self.up = nn.Linear(hidden_size, ff_size, bias=False)
        self.activation = nn.ReLU()
        self.down = nn.Linear(ff_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        return self.down(self.activation(self.up(x)))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        n_q_heads: int,
        n_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        ff_size: int,
    ):
        super().__init__()
        self.attn = SelfAttn(n_q_heads, n_kv_heads, hidden_size, head_dim)
        self.norm_attn = nn.RMSNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ff_size)
        self.norm_ffn = nn.RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        ff_size: int,
        vocab_size: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(n_q_heads, n_kv_heads, hidden_size, head_dim, ff_size)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return self.output_proj(x)  # [B, N_TOKENS, VOCAB_SIZE]
