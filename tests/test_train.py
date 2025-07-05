from config import TrainConfig, DataConfig, SampleConfig
from model import GPTconfig, GPT
from train import NameGPTTrainer
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
    # create train/dev data
    train_tokens = np.array([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.uint16)
    dev_tokens = np.array([1, 2, 0, 1, 2, 0, 1, 2], dtype=np.uint16)
    for tokens, name in [(train_tokens, "train.bin"), (dev_tokens, "dev.bin")]:
        tokens.tofile(tmp_path / name)
    # create metadata pickle with vocab mappings
    with open(tmp_path / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 4, "itos": itos, "stoi": vocab}, f)
    return tmp_path


@pytest.fixture
def configs(tmp_path):
    """ combined fixture for model and train configs """
    model_config = GPTconfig(
        context_len=2, vocab_size=10, n_embd=8, n_head=2, n_layer=2,
        dropout=0.2, ffw_widen=4, a_bias=True, ffw_bias=True, lm_head_bias=False
    )
    train_config = TrainConfig(
        batch_size=2, learning_rate=3e-4, train_iter=2, eval_iter=5,
        eval_interval=500, device="cpu", data_dir="data",
        saved_models_root=str(tmp_path), model_name="test_bavGPT",
        seed=42, num_samples=2
    )
    return train_config, model_config

# @pytest.fixture
# def sample_config(tmp_path):
#     return SampleConfig(device="cpu", num_samples=5, max_length=10, temperature=1.0)

# @pytest.fixture
# def SampleConfig


def test_NameGPTTrainer_init_wrong_configs(configs):
    train_config, model_config = configs
    with pytest.raises(AssertionError, match="Invalid train config type."):
        NameGPTTrainer(train_config=DataConfig(), model_config=model_config)
    with pytest.raises(AssertionError, match="Invalid model config type."):
        NameGPTTrainer(train_config=train_config, model_config=DataConfig())


def test_NameGPTTrainer_init(configs, temp_data_files):
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    trainer = NameGPTTrainer(train_config, model_config)
    assert isinstance(trainer.train_data, torch.Tensor)
    assert isinstance(trainer.dev_data, torch.Tensor)
    assert hasattr(trainer.model, "_parameters")
    assert isinstance(trainer.model.transformer, nn.ModuleDict)


def test_get_device_fallback(configs, temp_data_files, monkeypatch):
    # use monkeypatch.setattr to modify the function
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    train_config.device = "cuda"
    t = NameGPTTrainer(train_config, model_config)
    assert t.device == "cpu"


def test_load_data_invalid_bin_file_path(configs):
    with pytest.raises(AssertionError, match=".bin file not found."):
        train_config, model_config = configs
        train_config.data_dir = "invalid_file_path"
        NameGPTTrainer(train_config, model_config)


def test_load_data_missing_meta_file(configs, temp_data_files):
    # remove the meta.pkl file from the temporary directory
    os.remove(temp_data_files / "meta.pkl")
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    with pytest.raises(AssertionError, match="meta.pkl file not found"):
        NameGPTTrainer(train_config, model_config)


def test_load_data_vocab_size(configs, temp_data_files):
    """ model_config vocab_size: 10; meta.pkl testfile: 4 -> update to 4!! """
    train_config, model_config = configs
    assert model_config.vocab_size == 10
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    assert t.model_config.vocab_size == 4


def test_load_data_base(configs, temp_data_files):
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    assert isinstance(t.train_data, torch.Tensor) and len(t.train_data) == 12
    assert isinstance(t.dev_data, torch.Tensor) and len(t.dev_data) == 8


def test_get_batch(configs, temp_data_files):
    """ context_len: 2; batch_size = 2"""
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    x_tr, y_tr = t._get_batch(t.train_data)
    x_dev, y_dev = t._get_batch(t.dev_data)
    for x, y in [(x_tr, y_tr), (x_dev, y_dev)]:
        assert isinstance(x, torch.Tensor) and x.shape == (2, 2)
        assert isinstance(y, torch.Tensor) and y.shape == (2, 2)
    assert not torch.equal(x_tr, y_tr)
    # check offset relationship between x and y
    assert not (y_tr[0] == x_tr[0][0]).any()
    assert (y_tr[0] == x_tr[0][1]).any()
    assert not (x_tr[-1] == y_tr[-1][-1]).any()
    assert (x_tr[-1] == y_tr[-1][-2]).any()


def test_estimate_loss(configs, temp_data_files):
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    # switch model into eval mode, to check if _estimate_loss switches is back into train
    t.model.eval()
    losses = t._estimate_loss()
    assert t.model.training
    assert isinstance(losses, Dict) and len(losses) == 2
    # defaul avg nlll at vocab_size=4 equals -1.38; add tolerance
    tolerance = 0.1
    for split in ["train", "dev"]:
        assert isinstance(losses[split], float)
        assert 0 < losses[split] <= math.log(4) + tolerance


def test_train_base(configs, temp_data_files):
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    t.train_model()
    assert len(t.training_results) > 0
    assert "train_loss" in t.training_results[0]


def test_save_checkpoint_model(configs, temp_data_files):
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    # check if model_save_path was updated from None to str after train_model()
    assert t.model_save_dir is None
    t.train_model()
    assert isinstance(t.model_save_dir, str)
    # setup fresh model, load state_dict into it and compare with original state
    m = GPT(model_config)
    checkpoint = torch.load(os.path.join(t.model_save_dir, "model.pt"), map_location="cpu")
    m.load_state_dict(checkpoint)
    # Compare state dicts directly
    original_state = {k: v.cpu() for k, v in t.model.state_dict().items()}
    loaded_state = {k: v.cpu() for k, v in m.state_dict().items()}
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key])


def test_save_checkpoint_metadata(configs, temp_data_files):
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    t.train_model()
    json_path = os.path.join(t.model_save_dir, "config.json")
    with open(json_path, mode="r") as f:
        saved_config = json.load(f)
    assert saved_config["model_config"]["ffw_widen"] == 4
    assert len(saved_config["training_results"]["detailed_training_results"]) > 0


def test_sample_after_train(configs, temp_data_files):
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    # capture stdout for sampling to console
    captured_output = StringIO()
    sys.stdout = captured_output
    t.train_model()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "1." in output
    assert "2." in output

