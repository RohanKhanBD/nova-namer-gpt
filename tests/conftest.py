import pytest
import tempfile
import os
import torch
from config import DataConfig, TrainConfig, SampleConfig
from model import GPTconfig
import numpy as np
import pickle


""" fixtures for test_sample.py """


@pytest.fixture
def sample_cfg(tmp_path):
    return SampleConfig(
        device="cpu",
        num_samples=3,
        max_length=10,
        temperature=1.0,
        enforce_novelty=False,
        saved_samples_root=str(tmp_path),
    )


""" fixtures for test_train.py """


@pytest.fixture
def mock_train_data(tmp_path):
    """
    creates minimal test data files that _load_data() expects:
    - train.bin: 12 tokens as uint16 (enough for context_len=4, batch_size=2)
    - dev.bin: 8 tokens as uint16 (smaller dev set)
    - meta.pkl: vocab mappings for 4 characters
    """
    # minimal vocabulary: 4 characters (newline, a, b, c)
    vocab = {'\n': 0, 'a': 1, 'b': 2, 'c': 3}
    itos = {i: c for c, i in vocab.items()}
    training_names_set = {"a", "b", "c"}
    # create train/dev data
    train_tokens = np.array([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.uint16)
    dev_tokens = np.array([1, 2, 0, 1, 2, 0, 1, 2], dtype=np.uint16)
    for tokens, name in [(train_tokens, "train.bin"), (dev_tokens, "dev.bin")]:
        tokens.tofile(tmp_path / name)
    # create metadata pickle with vocab mappings
    with open(tmp_path / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 4, "itos": itos, "stoi": vocab, "training_names": training_names_set}, f)
    return tmp_path


@pytest.fixture
def train_cfg(mock_train_data, tmp_path):
    return TrainConfig(
        batch_size=2,
        learning_rate=3e-4,
        train_iter=2,
        eval_iter=5,
        eval_interval=500,
        device="cpu",
        data_dir=str(mock_train_data),
        saved_models_root=str(tmp_path),
        model_name="test_bavGPT",
        model_filename="model.pt",
        seed=42,
        num_samples=2,
    )


""" fixtures for test_model.py """


@pytest.fixture
def model_cfg():
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


""" fixtures for test_prepare_data.py """


@pytest.fixture
def data_cfg(mock_names_file_valid, tmp_path):
    cfg = DataConfig(
        input_file=mock_names_file_valid,
        output_dir=tmp_path,
        seed=42,
        min_name_length=3,
        max_name_length=50,
        train_size=0.8,
        dev_size=0.1,
        test_size=0.1,
    )
    return cfg


@pytest.fixture
def mock_names_file_valid():
    test_data = "München\nNür\nAugsburg\n" + "VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhnot\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_data)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def mock_names_file_invalid():
    test_data = "A\n" + "VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhhhhh\n" + "Berlin\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_data)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def mock_vocab_cases():
    """ essential test cases for vocabulary building """
    return {
        'basic': ["abc\n", "def\n"],  # expected: ['\n', 'a', 'b', 'c', 'd', 'e', 'f'] = 7 chars
        'duplicates': ["aaa\n", "abc\n"],  # expected: ['\n', 'a', 'b', 'c'] = 4 chars
        'german': ["München\n", "Nürnberg\n"]  # test umlauts + real data
    }
