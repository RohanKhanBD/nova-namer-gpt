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
def min_idx_tensor():
    return torch.stack((torch.arange(0, 4), torch.arange(4, 8)))


@pytest.fixture
def min_targets_tensor():
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


def test_GPT_init_num_params():
    config = GPTconfig()
    m = GPT(config, init_weights=False)
    assert m.get_num_params() == 6334208


def test_GPT_forward_T_greater_context_len(min_config):
    """ T dim of input idx may not be greater than context_len"""
    with pytest.raises(ValueError):
        m = GPT(min_config)
        idx = torch.stack((torch.zeros(1, 5), torch.zeros(1, 5)))
        logits, loss = m(idx)


def test_GPT_forward_train_base(min_config, min_idx_tensor, min_targets_tensor):
    """ when training with targets, logits shape gets flattened to (B*T, V) for loss calculation """
    m = GPT(min_config)
    B_idx, T_idx = min_idx_tensor.shape
    logits, loss = m(min_idx_tensor, min_targets_tensor)
    T_l, V_l = logits.shape
    # T flattened out in train case
    assert T_l == (B_idx * T_idx)
    assert V_l == m.config.vocab_size
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0


def test_GPT_forward_infer_base(min_config, min_idx_tensor):
    """ when infering with idx input shape (B,T) logits shape must be (B,T,V) V:vocab_size"""
    m = GPT(min_config)
    B_idx, T_idx = min_idx_tensor.shape
    logits, loss = m(min_idx_tensor)
    B_l, T_l, V_l = logits.shape
    assert B_idx == B_l == 2
    assert T_idx == T_l == m.config.context_len == 4
    assert V_l == m.config.vocab_size
    assert loss is None


