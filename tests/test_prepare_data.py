from config import DataConfig, TrainConfig
from prepare_data import NameProcessor
import pytest
import numpy as np
import pickle

""" fixtures for data_config with tmp_path, mock names.txt, vocab cases on conftest.py """


def test_NameProcessor_init_invalid_config():
    """ check for error when init NameProcessor with wrong config object """
    with pytest.raises(AssertionError):
        NameProcessor(TrainConfig())


def test_NameProcessor_init(data_cfg):
    p = NameProcessor(data_cfg)
    assert isinstance(p.config, DataConfig)
    assert p.config.input_file.endswith("txt")
    assert p.vocab_size == 0


def test_load_raw_data_no_input_file(data_cfg):
    with pytest.raises(AssertionError):
        config = data_cfg
        config.input_file = "this_file_does_not_exist.txt"
        p = NameProcessor(config)
        p._load_raw_data()


def test_load_raw_data_valid(data_cfg):
    config = data_cfg
    p = NameProcessor(config)
    names = p._load_raw_data()
    assert isinstance(names, list)
    assert all(isinstance(item, str) for item in names)
    assert len(names) == 4
    assert "München\n" in names


def test_load_raw_data_invalid(data_cfg, mock_names_file_invalid):
    config = data_cfg
    config.input_file = str(mock_names_file_invalid)
    p = NameProcessor(config)
    names = p._load_raw_data()
    assert len(names) == 1
    assert "Berlin\n" in names


def test_is_valid_name(data_cfg):
    """ test name len boundaries from DataConfig """
    config = data_cfg
    p = NameProcessor(config)
    assert p._is_valid_name("hah")
    assert p._is_valid_name("VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhnot\n")
    assert not p._is_valid_name("hu")
    assert not p._is_valid_name("VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhhhhh\n")


def test_shuffle_names(data_cfg):
    config = data_cfg
    # first processor
    p1 = NameProcessor(config)
    names1 = p1._load_raw_data()
    shuffled1 = p1._shuffle_names(names1)
    # second processor with SAME seed
    p2 = NameProcessor(config)
    names2 = p2._load_raw_data()
    shuffled2 = p2._shuffle_names(names2)
    assert names1 is not shuffled1
    assert names2 is not shuffled2
    assert shuffled1 == shuffled2


def test_vocabulary_size(data_cfg, mock_vocab_cases):
    """ test that vocabulary counts unique characters correctly """
    config = data_cfg
    p = NameProcessor(config)
    p._build_vocabulary(mock_vocab_cases["duplicates"])
    assert p.vocab_size == 4  # \n, a, b, c
    p._build_vocabulary(mock_vocab_cases["basic"])
    assert p.vocab_size == 7  # \n, a, b, c, d, e, f


def test_vocabulary_mappings(data_cfg, mock_vocab_cases):
    config = data_cfg
    p = NameProcessor(config)
    p._build_vocabulary(mock_vocab_cases["basic"])
    # test bidirectional mapping
    for char, idx in p.stoi.items():
        assert p.itos[idx] == char


def test_encode_decode_roundtrip(data_cfg, mock_vocab_cases):
    config = data_cfg
    p = NameProcessor(config)
    p._build_vocabulary(mock_vocab_cases["german"])
    test_string = "München\n"
    encoded = p.encode(test_string)
    decoded = p.decode(encoded)
    assert decoded == test_string
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)


def test_unknown_character_in_encode(data_cfg):
    config = data_cfg
    p = NameProcessor(config)
    p._build_vocabulary(["abc\n"])
    with pytest.raises(KeyError):
        p.encode("xyz")


def test_create_splits(data_cfg):
    config = data_cfg
    p = NameProcessor(config)
    idx = list(range(1, 11))
    train, dev, test = p._create_splits(idx)
    assert len(train) == 8 and len(dev) == 1 and len(test) == 1
    assert train + dev + test == idx


def test_export_data(data_cfg):
    """ equally E2E test of execute() method """
    config = data_cfg
    p = NameProcessor(config)
    p.execute()
    # check if bin & metadata paths exist
    for filename in ["train.bin", "dev.bin", "test.bin", "meta.pkl"]:
        assert (config.output_dir / filename).exists()
    # check bin file contents
    for split in ["train", "dev", "test"]:
        data = np.fromfile(config.output_dir / f"{split}.bin", dtype=np.uint16)
        assert len(data) > 0
    # check meta file contents
    with open(config.output_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert all(key in meta for key in ["vocab_size", "itos", "stoi", "training_names"])
    assert meta["vocab_size"] > 0


def test_export_data_training_names(data_cfg):
    """ check if name from mock file in training_names meta.pkl """
    cfg = data_cfg
    p = NameProcessor(cfg)
    p.execute()
    with open(cfg.output_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert "München" in meta["training_names"]
