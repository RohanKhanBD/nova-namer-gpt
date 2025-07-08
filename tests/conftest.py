import pytest
import torch
from config import DataConfig, TrainConfig, SampleConfig
from model import GPTconfig
import numpy as np
import pickle


@pytest.fixture
def sample_cfg(tmp_path):
    return SampleConfig(
        device="cpu",
        num_samples=3,
        max_length=5,
        temperature=1.0,
        enforce_novelty=False,
        saved_samples_root=str(tmp_path),
    )


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
    data_files = [
        (np.array([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.uint16), "train.bin"),
        (np.array([1, 2, 0, 1, 2, 0, 1, 2], dtype=np.uint16), "dev.bin")
    ]
    for tokens, filename in data_files:
        tokens.tofile(tmp_path / filename)
    # create metadata pickle with vocab mappings
    with open(tmp_path / "meta.pkl", "wb") as f:
        pickle.dump({
            "vocab_size": 4,
            "itos": itos,
            "stoi": vocab,
            "training_names": training_names_set
        }, f)
    return tmp_path


@pytest.fixture
def min_idx_tensor():
    return torch.stack((torch.arange(0, 4), torch.arange(4, 8)))


@pytest.fixture
def min_targets_tensor():
    return torch.stack((torch.arange(1, 5), torch.arange(5, 9)))


@pytest.fixture
def mock_names_file_valid(tmp_path):
    test_data = "München\nNür\nAugsburg\nVeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhnot\n"
    test_file = tmp_path / "test_names.txt"
    test_file.write_text(test_data, encoding="utf-8")
    return str(test_file)


@pytest.fixture
def mock_names_file_invalid(tmp_path):
    test_data = "A\nVeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhhhhh\nBerlin\n"
    test_file = tmp_path / "test_names_invalid.txt"
    test_file.write_text(test_data, encoding="utf-8")
    return str(test_file)


@pytest.fixture
def mock_vocab_cases():
    return {
        'basic': ["abc\n", "def\n"],
        'duplicates': ["aaa\n", "abc\n"],
        'german': ["München\n", "Nürnberg\n"]
    }
