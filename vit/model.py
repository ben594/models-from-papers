import math

import torch.nn as nn
import torch

from loaders import IMAGE_DIM

HIDDEN_SIZE = 768
MLP_SIZE = 3072
N_HEADS = 12
N_LAYERS = 12
D_HEAD = HIDDEN_SIZE // N_HEADS
PATCH_DIM = 16
BATCH_SIZE = 32


class Patches(nn.Module):
    def __init__(
        self,
        image_dim: int = IMAGE_DIM,
        hidden_size: int = HIDDEN_SIZE,
        patch_dim: int = PATCH_DIM,
    ):
        super().__init__()
        self.image_dim = image_dim
        self.hidden_size = hidden_size
        assert image_dim % patch_dim == 0
        self.n_patches = (image_dim // patch_dim) ** 2
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_dim,
            stride=patch_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x C x H x W
        conv_result: torch.Tensor = self.conv(
            x
        )  # B x HIDDEN_SIZE x (image_dim/patch_dim) x (image_dim/patch_dim)

        # flatten last 2 dims
        flattened = torch.flatten(
            conv_result, start_dim=-2
        )  # B X HIDDEN_SIZE X n_patches
        patches = flattened.permute(0, 2, 1).contiguous()

        return patches


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x N_TOKENS x HIDDEN_SIZE
        return self.layer_norm(x)


class MLP(nn.Module):
    def __init__(
        self,
        mlp_size: int = MLP_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x N_TOKENS x HIDDEN_SIZE
        return self.layer(x)


class MHA(nn.Module):
    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        n_heads: int = N_HEADS,
        d_head: int = D_HEAD,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = d_head
        assert hidden_size == n_heads * d_head

        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x N_TOKENS x HIDDEN_SIZE
        n_batches = x.shape[0]
        n_tokens = x.shape[1]
        Q = self.w_q(x)  # B x N_TOKENS x HIDDEN_SIZE
        K = self.w_k(x)  # B x N_TOKENS x HIDDEN_SIZE
        V = self.w_v(x)  # B x N_TOKENS x HIDDEN_SIZE

        # reshape so that attn head is the second dimension
        Q = Q.view(n_batches, n_tokens, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = K.view(n_batches, n_tokens, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = V.view(n_batches, n_tokens, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)
        )  # B x N_HEADS x N_TOKENS x N_TOKENS
        attn_scores = attn_scores / math.sqrt(self.d_head)
        attn_weights = self.dropout(self.softmax(attn_scores))

        weighted_vals = torch.matmul(attn_weights, V)  # B x N_HEADS x N_TOKENS x D_HEAD
        weighted_vals = (
            weighted_vals.permute(0, 2, 1, 3)
            .contiguous()
            .view(n_batches, n_tokens, self.hidden_size)
        )
        # TODO dropout?
        out = self.w_o(weighted_vals)
        return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        mlp_size: int = MLP_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        n_heads: int = N_HEADS,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_layer_norm = LayerNorm(hidden_size)
        self.attn = MHA(hidden_size, n_heads, dropout=dropout)
        self.mlp_layer_norm = LayerNorm(hidden_size)
        self.mlp = MLP(mlp_size, hidden_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x N_TOKENS x HIDDEN_SIZE
        x = x + self.attn(self.attn_layer_norm(x))
        x = x + self.mlp(self.mlp_layer_norm(x))
        return x


class Classifier(nn.Module):
    def __init__(
        self, n_classes: int, hidden_size: int = HIDDEN_SIZE, mlp_size: int = MLP_SIZE
    ):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.mlp = self.layer = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x HIDDEN_SIZE
        # x contains first token from each batch (learnable classification token)
        return self.mlp(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.n_patches = n_patches
        self.embeddings = nn.Parameter(
            torch.randn(1, n_patches + 1, hidden_size), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x N_TOKENS x HIDDEN_SIZE
        return x + self.embeddings


class ViT(nn.Module):
    def __init__(
        self,
        n_classes: int,
        image_dim: int = IMAGE_DIM,
        patch_dim: int = PATCH_DIM,
        n_layers: int = N_LAYERS,
        mlp_size: int = MLP_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        n_heads: int = N_HEADS,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patcher = Patches(
            image_dim=image_dim,
            hidden_size=hidden_size,
            patch_dim=patch_dim,
        )
        self.class_token = nn.Parameter(
            torch.zeros(1, 1, hidden_size), requires_grad=True
        )
        assert image_dim % patch_dim == 0
        n_patches = (image_dim // patch_dim) ** 2
        self.positional_embedding = PositionalEmbedding(
            n_patches=n_patches, hidden_size=hidden_size
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    mlp_size=mlp_size,
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.encoder_norm = LayerNorm(hidden_size)
        self.classifier = Classifier(
            n_classes=n_classes, hidden_size=hidden_size, mlp_size=mlp_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x C x H x W
        patches = self.patcher(x)  # B x N_TOKENS x HIDDEN_SIZE
        # add classification token in first position
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        patches = torch.cat((class_token, patches), dim=1)
        patches = self.positional_embedding(patches)
        for layer in self.layers:
            patches = layer(patches)
        patches = self.encoder_norm(patches)
        first_tokens = patches[:, 0, :]
        classifier_result = self.classifier(first_tokens)
        return classifier_result
