from model import GPTconfig, GPT, TransformerBlock, Ffw, MultiHeadAttention
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
        dropout=0.0,  # Set to 0 for deterministic tests
        ffw_widen=4,  
        a_bias=True,
        ffw_bias=True,  
        lm_head_bias=False,
    )


@pytest.fixture
def min_idx_tensor():
    return torch.stack((torch.arange(0, 4), torch.arange(4, 8)))


@pytest.fixture
def min_targets_tensor():
    return torch.stack((torch.arange(1, 5), torch.arange(5, 9)))


def test_GPT_init_validations():
    with pytest.raises(AssertionError, match="Invalid config type."):
        GPT(TrainConfig())
    with pytest.raises(AssertionError, match="Ratio n_embd / n_head must have no remainder."):
        GPT(GPTconfig(n_head=3))
    with pytest.raises(AssertionError, match="n_head must be positive"):
        GPT(GPTconfig(n_head=0))
    with pytest.raises(AssertionError, match="n_layer must be positive"):
        GPT(GPTconfig(n_layer=0))
    with pytest.raises(AssertionError, match="vocab_size must be positive"):
        GPT(GPTconfig(vocab_size=0))


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
    with pytest.raises(AssertionError):
        m = GPT(min_config)
        idx = torch.zeros(2, 5, dtype=torch.long)
        m(idx)


def test_GPT_forward_train_base(min_config, min_idx_tensor, min_targets_tensor):
    """ when training with targets, logits shape gets flattened to (B*T, C) for loss calculation """
    m = GPT(min_config)
    B_idx, T_idx = min_idx_tensor.shape
    logits, loss = m(min_idx_tensor, min_targets_tensor)
    B_l, T_l, C_l = logits.shape
    # flattened out into (B*T, C) only for cross_entropy calc; returned as (B,T,C)
    assert B_l == B_idx
    assert T_l == T_idx
    assert C_l == m.config.vocab_size
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0


def test_GPT_forward_infer_base(min_config, min_idx_tensor):
    """ when infering with idx input shape (B,T) logits shape must be (B,T,C) C:vocab_size"""
    m = GPT(min_config)
    B_idx, T_idx = min_idx_tensor.shape
    logits, loss = m(min_idx_tensor)
    B_l, T_l, C_l = logits.shape
    assert B_idx == B_l == 2
    # only last T is taken, but dim preserved with 1 for downstream services
    assert T_l == 1
    assert C_l == m.config.vocab_size
    assert loss is None


def test_MultiHeadAttention(min_config):
    batch_size = 2
    mha = MultiHeadAttention(min_config)
    x = torch.randn(batch_size, min_config.context_len, min_config.n_embd)
    out = mha(x)
    assert out.shape == x.shape


def test_MultiHeadAttention_qvc_projection(min_config):
    mha = MultiHeadAttention(min_config)
    B, T, C = 2, 4, 8
    x = torch.randn(B, T, C)
    qkv_out = mha.qkv(x)
    assert qkv_out.shape == (B, T, 3 * C)
    # test reshape & permute
    qkv = qkv_out.reshape(B, T, 3, mha.n_head, mha.head_size).permute(2, 0, 3, 1, 4)
    assert qkv.shape == (3, B, mha.n_head, T, mha.head_size)
    q, k, v = qkv[0], qkv[1], qkv[2]
    assert q.shape == (B, mha.n_head, T, mha.head_size)


def test_attention_mask_causal(min_config):
    mha = MultiHeadAttention(min_config)
    B, T = 1, 4
    x = torch.ones(B, T, min_config.n_embd)
    assert torch.all(mha.tril[:T, :T] == torch.tril(torch.ones(T, T)))
    out = mha(x)
    assert out.shape == x.shape


def test_weight_tying():
    config = GPTconfig(vocab_size=20, n_embd=16)
    m = GPT(config, init_weights=False)
    assert m.transformer.wte.weight is m.lm_head.weight
    assert m.transformer.wte.weight.shape == (20, 16)
    assert m.lm_head.weight.shape == (20, 16)


def test_positional_encoding_device_placement(min_config):
    m = GPT(min_config)
    # test CPU
    idx_cpu = torch.randint(0, 10, (2, 4))
    logits_cpu, _ = m(idx_cpu)
    assert logits_cpu.device == idx_cpu.device
    # test GPU if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        m_mps = GPT(min_config).to(device)
        idx_mps = idx_cpu.to(device)
        logits_mps, _ = m_mps(idx_mps)
        assert logits_mps.device == idx_mps.device


def test_ffw_widening_factor(min_config):
    ffw = Ffw(min_config)
    assert ffw.c_fc.out_features == min_config.n_embd * min_config.ffw_widen
    assert ffw.proj.in_features == min_config.n_embd * min_config.ffw_widen
    assert ffw.proj.out_features == min_config.n_embd


def test_gradient_flow(min_config, min_idx_tensor, min_targets_tensor):
    """ check gradients flow through key parameters """
    m = GPT(min_config)
    _, loss = m(min_idx_tensor, min_targets_tensor)
    loss.backward()
    assert m.transformer.wte.weight.grad is not None
    assert m.transformer.wpe.weight.grad is not None
    assert m.lm_head.weight.grad is not None
    for block in m.transformer.h:
        assert block.multi_head_sa.qkv.weight.grad is not None
        assert block.multi_head_sa.proj.weight.grad is not None


def test_transformer_block_residual_connections(min_config):
    block = TransformerBlock(min_config)
    x = torch.randn(2, 4, 8)
    x_copy = x.clone()
    out = block(x)
    # uutput should be different from input due to transformations
    assert not torch.allclose(out, x_copy)
    assert out.shape == x_copy.shape

