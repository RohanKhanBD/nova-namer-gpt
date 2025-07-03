from config import TrainConfig, DataConfig, SampleConfig
from model import GPTconfig, GPT
from train import NameGPTTrainer
from sample import NameGPTSampler
import torch
import torch.nn as nn
import numpy as np
import pickle
import pytest
from typing import Dict
import math
import os
import json
from io import StringIO
import sys


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
        context_len=2,
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
def min_train_config(tmp_path):
    return TrainConfig(
        batch_size=2,
        learning_rate=3e-4,
        train_iter=2,
        eval_iter=5,
        eval_interval=500,
        device="cpu",
        data_dir="data",
        model_save_dir=str(tmp_path),
        model_name="test_bavGPT",
        seed=42,
        sample_after_train=False,
        num_samples=5,
    )


def test_NameGPTTrainer_init_wrong_configs(min_train_config, min_model_config):
    with pytest.raises(AssertionError, match="Invalid train config type."):
        NameGPTTrainer(train_config=DataConfig(), model_config=min_model_config)
    with pytest.raises(AssertionError, match="Invalid model config type."):
        NameGPTTrainer(train_config=min_train_config, model_config=DataConfig())


def test_NameGPTTrainer_init(min_train_config, min_model_config, temp_data_files):
    t_config = min_train_config
    t_config.data_dir = str(temp_data_files)
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
    t_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(t_config, m_config)
    assert t.model_config.vocab_size == 4


def test_load_data_base(min_train_config, min_model_config, temp_data_files):
    t_config = min_train_config
    t_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(t_config, min_model_config)
    assert isinstance(t.train_data, torch.Tensor) and len(t.train_data) == 12
    assert isinstance(t.dev_data, torch.Tensor) and len(t.dev_data) == 8


def test_get_batch(min_train_config, min_model_config, temp_data_files):
    """ context_len: 2; batch_size = 2"""
    t_config = min_train_config
    t_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(t_config, min_model_config)
    x_tr, y_tr = t._get_batch(t.train_data)
    x_dev, y_dev = t._get_batch(t.dev_data)
    assert isinstance(x_tr, torch.Tensor) and x_tr.shape == (2, 2)
    assert isinstance(y_tr, torch.Tensor) and y_tr.shape == (2, 2)
    assert isinstance(x_dev, torch.Tensor) and x_dev.shape == (2, 2)
    assert isinstance(y_dev, torch.Tensor) and y_dev.shape == (2, 2)
    assert not torch.equal(x_tr, y_tr)
    # check if first value of first batch x_tr is not element in y_tr; vs second value is element
    first_xtr = x_tr[0][0]
    assert not (y_tr[0] == first_xtr).any()
    second_xtr = x_tr[0][1]
    assert (y_tr[0] == second_xtr).any()
    # check for last value of y_tr
    last_ytr = y_tr[-1][-1]
    assert not (x_tr[-1] == last_ytr).any()
    sec_last_ytr = y_tr[-1][-2]
    assert (x_tr[-1] == sec_last_ytr).any()


def test_estimate_loss(min_train_config, min_model_config, temp_data_files):
    t_config = min_train_config
    t_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(t_config, min_model_config)
    # switch model into eval mode, to check if _estimate_loss switches is back into train
    t.model.eval()
    losses = t._estimate_loss()
    assert t.model.training
    assert isinstance(losses, Dict) and len(losses) == 2
    assert isinstance(losses["train"], float)
    assert isinstance(losses["dev"], float)
    # defaul avg nlll at vocab_size=4 equals -1.38; add tolerance of 0.05
    tolerance = 0.1
    assert 0 < losses["train"] <= math.log(4) + tolerance
    assert 0 < losses["dev"] <= math.log(4) + tolerance


def test_train_base(min_train_config, min_model_config, temp_data_files):
    t_config = min_train_config
    t_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(t_config, min_model_config)
    t.train_model()
    assert len(t.training_results) > 0
    assert "train_loss" in t.training_results[0]


def test_save_checkpoint_model(min_train_config, min_model_config, temp_data_files, tmp_path):
    t_config = min_train_config
    t_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(min_train_config, min_model_config)
    # check if model_save_path was updated from None to str after train_model()
    assert t.model_save_path is None
    t.train_model()
    assert isinstance(t.model_save_path, str)
    # setup fresh model, load state_dict into it and compare with original state
    m = GPT(min_model_config)
    checkpoint = torch.load(os.path.join(t.model_save_path, "model.pt"), map_location="cpu")
    m.load_state_dict(checkpoint)
    # Compare state dicts directly
    original_state = {k: v.cpu() for k, v in t.model.state_dict().items()}
    loaded_state = {k: v.cpu() for k, v in m.state_dict().items()}
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key])


def test_save_checkpoint_metadata(min_train_config, min_model_config, temp_data_files, tmp_path):
    t_config = min_train_config
    t_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(min_train_config, min_model_config)
    t.train_model()
    json_path = os.path.join(t.model_save_path, "config.json")
    with open(json_path, mode="r") as f:
        saved_config = json.load(f)
    assert saved_config["model_config"]["ffw_widen"] == 4
    assert len(saved_config["training_results"]["detailed_training_results"]) > 0


# def test_sample_after_train(min_train_config, min_model_config, temp_data_files):
#     t_config = min_train_config
#     t_config.data_dir = str(temp_data_files)
#     t_config.sample_after_train = True
#     t = NameGPTTrainer(t_config, min_model_config)
#     # capture stdout for sampling to console
#     captured_output = StringIO()
#     sys.stdout = captured_output
#     t.train_model()
#     sys.stdout = sys.__stdout__
#     output = captured_output.getvalue()
#     assert "1." in output #....