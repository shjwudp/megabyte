# MIT License
# Copyright (c) 2023 Jianbin Chang

"""
The implementation of the transformer is adapted from 
https://github.com/lucidrains/MEGABYTE-pytorch/blob/main/MEGABYTE_pytorch/megabyte.py
these codes follow the license:
"""

# MIT License
# Copyright (c) 2023 Phil Wang

from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from model.attend import Attend


MegabyteConfig = namedtuple(
    "MegabyteConfig",
    [
        "V", "P", "D_G", "D_L", "T_MAX",
        "g_nheads", "g_nlayers",
        "l_nheads", "l_nlayers",
        "initializer_range",
        "pad_id",
    ]
)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.attend = Attend(
            causal = True,
            flash = flash,
            dropout = dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, attn_bias = None):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        out = self.attend(q, k, v, attn_bias = attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
    )


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        flash_attn = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        self.norm = RMSNorm(dim)

    def forward(self, x):
        n = x.shape[-2]
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x) + x

        return self.norm(x)


class Megabyte(nn.Module):
    """
    notation
    V - vocabulary size
    P - patch size
    D_G - global dimension
    D_L - local dimension
    T - sequence length
    """

    def __init__(
        self,
        config: MegabyteConfig,
    ):
        super().__init__()
        self.config = config
        P = config.P
        V = config.V
        D_G = config.D_G
        D_L = config.D_L

        self.g_embedder = nn.Embedding(V, D_G)
        self.g_pos_embedder = nn.Embedding(config.T_MAX, D_G)
        self.g_transformer = Transformer(
            dim=config.P*config.D_G,
            layers=config.g_nlayers,
            dim_head=(config.P*config.D_G)//config.g_nheads,
            heads=config.g_nheads,
        )
        self.gl_linear = nn.Sequential(
            Rearrange("... (P D_G) -> ... P D_G", P=P, D_G=D_G),
            nn.Linear(D_G, D_L),
            Rearrange("... P D_L -> (...) P D_L", P=P, D_L=D_L),
        )

        self.l_embedder = nn.Embedding(V, D_L)
        self.l_transformer = Transformer(
            dim=config.D_L,
            layers=config.l_nlayers,
            dim_head=config.D_L//config.l_nheads,
            heads=config.l_nheads,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def patch_embed(self, ids):
        B, K, P = ids.shape
        T = K*P
        embedder = self.g_embedder
        pos_embedder = self.g_pos_embedder

        # add spatial tokens
        ids = F.pad(ids[:, :K-1, :], (0, 0, 1, 0), value=self.config.pad_id)

        # embedding = tokens embedding + absolute position embedding
        tokens_embed = embedder(rearrange(ids, "B ... -> B (...)", B=B))
        pos = torch.cat([torch.arange(T) for _ in range(B)]).reshape(B, T)
        pos_embed = pos_embedder(pos)
        h = tokens_embed + pos_embed
        h = rearrange(h, "B (K P) ... -> B K (P ...)", B=B, K=K, P=P)

        return h

    def forward(self, ids):
        """
        Input "ids" is for pretraining language model, ids shape is [B, K*P].
        
        In the global model, input/output ids[:, :K*(P-1)]/ids[:, :] trains the 
        ability of global model to predict next patch hidden states.

        In the local model, ids are rewritten into shape [B, K, P], and 
        input/output ids[:, :, :P-1]/ids[:, :, :] is used to train the local 
        model's ability to predict the next token.
        """
        B, T = ids.shape
        P = self.config.P
        K = T//P

        global_in = self.patch_embed(
            ids=rearrange(ids, "... (K P) -> ... K P", K=K, P=P)
        )
        global_out = self.g_transformer(global_in)

        l_input_ids = rearrange(ids, "B (K P) -> (B K) P", B=B, K=K, P=P)
        l_input_ids = F.pad(l_input_ids[:, :P-1], (1, 0), value=self.config.pad_id)
        l_embed = self.l_embedder(l_input_ids)
        local_in = self.gl_linear(global_out) + l_embed
        local_out = self.l_transformer(local_in)

        lm_logits = F.linear(local_out, self.l_embedder.weight)
        labels = ids
        loss = F.cross_entropy(
            rearrange(lm_logits, "... V -> (...) V", V=self.config.V),
            rearrange(labels, "... -> (...)"),
            ignore_index=self.config.pad_id,
        )

        return loss


if __name__ == "__main__":
    V = 512
    P = 4
    D_G = 512
    D_L = 128
    T = 1024
    B = 2
    K = T//P
    PAD_ID = 257

    config = MegabyteConfig(
        V=V,
        P=P,
        D_G=D_G,
        D_L=D_L,
        T_MAX=T,
        initializer_range=0.02,
        g_nlayers=4,
        g_nheads=16,
        l_nlayers=2,
        l_nheads=8,
        pad_id=PAD_ID
    )
    megabyte = Megabyte(config)
    input_ids = torch.randint(0, 255, (B, T))
    loss = megabyte(input_ids)
    loss.backward()

    print(loss.norm())
