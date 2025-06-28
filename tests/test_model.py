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
def min_x_tensor():
    return torch.stack((torch.arange(0, 4), torch.arange(4, 8)))


@pytest.fixture
def min_y_tensor():
    return torch.stack((torch.arange(1, 5), torch.arange(5, 9)))


def test_init_GPT_wrong_config():
    with pytest.raises(TypeError):
        m = GPT(TrainConfig())


def test_init_GPT_wrong_embd_head_ratio(min_config):
    """ n_embd / n_head must have no remainder """
    with pytest.raises(ValueError):
        config = min_config
        config.n_head = 3
        m = GPT(config)


def test_GPT_init(min_config):
    m = GPT(min_config, init_weights=False)
    assert isinstance(m.config, GPTconfig)
    assert hasattr(m, "_parameters")
    assert isinstance(m.transformer, nn.ModuleDict)
    assert m.config.n_embd % m.config.n_head == 0


def test_GPT_forward_wrong_T(min_config):
    """ T dim of input idx may not be greater than context_len"""
    with pytest.raises(ValueError):
        m = GPT(min_config)
        idx = torch.stack((torch.zeros(1, 5), torch.zeros(1, 5)))
        logits, loss = m(idx)


def test_GPT_forward_infer_basic(min_config, min_x_tensor):
    m = GPT(min_config)
    logits, loss = m(min_x_tensor)
    assert logits is not None
    assert loss is None

