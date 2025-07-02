from config import TrainConfig, DataConfig
from model import GPTconfig, GPT
from train import NameGPTTrainer
import torch
import torch.nn as nn
import numpy as np
import pickle
import pytest


@pytest.fixture
def temp_data_files(tmp_path):
    """
    creates minimal test data files that _load_data() expects:
    - train.bin: 12 tokens as uint16 (enough for context_len=4, batch_size=2)
    - dev.bin: 8 tokens as uint16 (smaller dev set)
    - meta.pkl: vocab mappings for 4 characters
    """
    # minimal vocabulary: 4 characters (newline, a, b, c)
    vocab = {'\n': 0, 'a': 1, 'b': 2, 'c': 3}
    itos = {i: c for c, i in vocab.items()}
    stoi = vocab
    # create train data: "abc\nabc\nabc\n" -> 12 tokens
    train_tokens = np.array([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.uint16)
    train_tokens.tofile(tmp_path / "train.bin")
    # create dev data: "ab\nab\n" -> 8 tokens  
    dev_tokens = np.array([1, 2, 0, 1, 2, 0, 1, 2], dtype=np.uint16)
    dev_tokens.tofile(tmp_path / "dev.bin")
    # create metadata pickle with vocab mappings
    meta = {
        "vocab_size": 4,
        "itos": itos,
        "stoi": stoi,
    }
    with open(tmp_path / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    return tmp_path  # Return path so tests can set data_dir


@pytest.fixture
def min_model_config():
    return GPTconfig(
        context_len=4,
        vocab_size=10,
        n_embd=8,
        n_head=2,
        n_layer=2,
        dropout=0.2,
        ffw_widen=4,  
        a_bias=True,
        ffw_bias=True,  
        lm_head_bias=False,
    )


@pytest.fixture
def min_train_config():
    return TrainConfig(
        batch_size=2,
        learning_rate=3e-4,
        train_iter=1,
        eval_iter=5,
        eval_interval=500,
        device="mps",
        data_dir="data",
        save_model=True,
        model_save_dir="saved_models",
        model_name="bavGPT",
        seed=42,
        sample_after_train=True,
        num_samples=5,
    )


def test_NameGPTTrainer_init_wrong_configs(min_train_config, min_model_config):
    with pytest.raises(AssertionError, match="Invalid train config type."):
        NameGPTTrainer(train_config=DataConfig(), model_config=min_model_config)
    with pytest.raises(AssertionError, match="Invalid model config type."):
        NameGPTTrainer(train_config=min_train_config, model_config=DataConfig())


def test_NameGPTTrainer_init(min_train_config, min_model_config, temp_data_files):
    t_config = min_train_config
    t_config.data_dir = temp_data_files
    trainer = NameGPTTrainer(t_config, min_model_config)
    assert isinstance(trainer.train_data, torch.Tensor)
    assert isinstance(trainer.dev_data, torch.Tensor)
    assert hasattr(trainer.model, "_parameters")
    assert isinstance(trainer.model.transformer, nn.ModuleDict)


def test_load_data_invalid_bbin_file_path(min_train_config, min_model_config):
    with pytest.raises(AssertionError, match=".bin file not found."):
        t_config = min_train_config
        t_config.data_dir = "invalid_file_path"
        NameGPTTrainer(t_config, min_model_config)


def test_load_data_vocab_size(min_train_config, min_model_config, temp_data_files):
    """ model_config vocab_size: 10; meta.pkl testfile: 4 -> update to 4!! """
    t_config = min_train_config
    m_config = min_model_config
    assert m_config.vocab_size == 10
    t_config.data_dir = temp_data_files
    t = NameGPTTrainer(t_config, m_config)
    assert t.model_config.vocab_size == 4


def test_load_data_base(min_train_config, min_model_config, temp_data_files):
    t_config = min_train_config
    t_config.data_dir = temp_data_files
    t = NameGPTTrainer(t_config, min_model_config)
    assert isinstance(t.train_data, torch.Tensor) and len(t.train_data) == 12
    assert isinstance(t.dev_data, torch.Tensor) and len(t.dev_data) == 8













