from model import GPTconfig, GPT, TransformerBlock, Ffw, MultiHeadAttention, Head
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from config import TrainConfig


@pytest.fixture
def min_config():
    return GPTconfig(
        context_len=4,
        vocab_size=10,
        n_embd=8,
        n_head=2,
        n_layer=2,
        dropout=0.2,
        ffw_widen=4,  # factor to widen linear layer in ffw module
        a_bias=True,  # bias true for q, k, v, proj in attention layers
        ffw_bias=True,  # bias true for lin layers in ffw modules;
        lm_head_bias=False,
    )

@pytest.fixture
def min_tensor():
    return torch.stack((torch.arange(4), torch.arange(4)))


def test_init_GPT_wrong_config():
    with pytest.raises(TypeError):
        m = GPT(TrainConfig())


def test_GPT_init(min_config):
    m = GPT(min_config, init_weights=False)
    assert isinstance(m.config, GPTconfig)
    assert hasattr(m, "_parameters")
    assert isinstance(m.transformer, nn.ModuleDict)