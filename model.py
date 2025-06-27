import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

""" Bavarian City Name GPT // core classes to setup transformer nn"""


@dataclass
class GPTconfig:
    """
    - configuration class for model architecture hyperparameters
    - mandatory as input to instanciate core GTP class
    - training & sampling config params in separate files in config dir
    """

    context_len: int = 64  # block_size
    vocab_size: int = 61
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.2
    ffw_widen: int = 4  # factor to widen linear layer in ffw module
    a_bias: bool = True  # bias true for q, k, v, proj in attention layers
    ffw_bias: bool = True  # bias true for lin layers in ffw modules;
    lm_head_bias: bool = False


class Head(nn.Module):
    """
    - single self-attention head; called from multi-head-attention class
    - pre-registered full-size buffer for triangular masking
    """

    def __init__(self, config: GPTconfig, h_size: int):
        super().__init__()
        self.query = nn.Linear(config.n_embd, h_size, bias=config.a_bias)
        self.key = nn.Linear(config.n_embd, h_size, bias=config.a_bias)
        self.value = nn.Linear(config.n_embd, h_size, bias=config.a_bias)
        # helper matrix for triangular masking; all zero values above diagonal
        self.register_buffer("tril", torch.tril(torch.ones(config.context_len, config.context_len)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x)  # B,T,H
        k = self.key(x)  # B,T,H
        wei = q @ torch.transpose(k, dim0=-1, dim1=-2) * C**-0.5  # B,T,T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B,T,T
        wei = F.softmax(wei, dim=-1)  # B,T,T
        # dropout after softmax
        wei = self.dropout(wei)  # B,T,T
        v = self.value(x)  # B,T,H
        out = wei @ v  # B,T,H
        return out


class MultiHeadAttention(nn.Module):
    """steering multiple heads of self-attention in parallel; returning cat heads output after"""

    def __init__(self, config: GPTconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size: int = config.n_embd // config.n_head
        self.heads = nn.ModuleList(Head(config, self.head_size) for _ in range(config.n_head))
        # linear projection layer to blend all cat head outputs
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.a_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        # cat each head out along last dim
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Ffw(nn.Module):
    """
    - mlp layer with relu non-linearity
    - widened by tbd factor; dropout before returning output
    - explicit layer naming to catch proj weights for weight init
    """

    def __init__(self, config: GPTconfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * config.ffw_widen, bias=config.ffw_bias)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(config.n_embd * config.ffw_widen, config.n_embd, bias=config.ffw_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    - communication in multi-head-attention, computation in ffw layers
    - layernorm -> attention -> skip -> layernorm -> ffw -> skip
    """

    def __init__(self, config: GPTconfig):
        super().__init__()
        self.multi_head_sa = MultiHeadAttention(config)
        self.ffw = Ffw(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x) -> torch.Tensor:
        x = x + self.multi_head_sa((self.ln1(x)))
        x = x + self.ffw(self.ln2(x))
        return x


# Core GPT logic setting up NN
class GPT(nn.Module):
    """central class setting up the NN"""

    def __init__(self, config: GPTconfig):
        super().__init__()
        self.config = config
        # embeddings & transformer blocks
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.context_len, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # output layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.lm_head_bias)
        # weight tying between token embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight
        # trigger weight init
        self.apply(self._init_weights)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.tensor]]:

        # derive device from idx arg
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.context_len
        # creates 1D-tensor with values from 0 - context_len; T
        pos_idx = torch.arange(0, T, dtype=torch.long, device=device)

        # setup embeddings
        tok_emb = self.transformer.wte(idx)  # B,T,C
        pos_emb = self.transformer.wpe(pos_idx)  # position embeddings; T,C
        # dropout on combined embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        # forward through hidden layers & layernorm
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # calc loss if targets are available; otherwise loss is None
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # flatten logits into B*T, C
            logits = logits.view(B * T, C)
            # flatten targets into B*T
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def _init_weights(self, module) -> None:
        """standard initfor lin / embd layers; scaled residual init for projection layers"""
        if isinstance(module, nn.Linear):
            # init weights with small normal distribution (GPT-2 standard)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # init embd with same std as linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # scaled init to residual projections
        for name, param in self.named_parameters():
            # catches both attention and ffw projections
            if name.endswith("proj.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )
