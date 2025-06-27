from config import DataConfig, TrainConfig
from prepare_data import NameProcessor
import pytest
import tempfile
import os
import numpy as np
import pickle


@pytest.fixture
def temp_names_file_valid():
    """ temp text file with 5 names in accepted length"""
    test_data = "München\nNür\nAugsburg\n" + "VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhnot\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_data)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_names_file_invalid():
    """ 3 min length, 50 max"""
    test_data = "A\n" + "VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhhhhh\n" + "Berlin\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_data)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def vocab_test_cases():
    """ essential test cases for vocabulary building """
    return {
        'basic': ["abc\n", "def\n"],  # expected: ['\n', 'a', 'b', 'c', 'd', 'e', 'f'] = 7 chars
        'duplicates': ["aaa\n", "abc\n"],  # expected: ['\n', 'a', 'b', 'c'] = 4 chars
        'german': ["München\n", "Nürnberg\n"]  # test umlauts + real data
    }


def test_NameProcessor_init_invalid_config():
    """ check for error when init NameProcessor with wrong config object """
    with pytest.raises(TypeError):
        config = TrainConfig()
        processor = NameProcessor(config)


def test_NameProcessor_init():
    """ test itit of nameprocessor objects """
    config = DataConfig()
    p = NameProcessor(config)
    assert isinstance(p.config, DataConfig)
    assert hasattr(p.config, "input_file")
    assert p.config.input_file.endswith("txt")
    assert hasattr(p, "stoi")
    assert hasattr(p, "itos")
    assert p.vocab_size == 0
    assert hasattr(p, "rng")


def test_load_raw_data_no_input_file():
    with pytest.raises(FileNotFoundError):
        config = DataConfig()
        config.input_file = "this_file_does_not_exist.txt"
        p = NameProcessor(config)
        p._load_raw_data()


def test_load_raw_data_valid(temp_names_file_valid):
    config = DataConfig()
    config.input_file = temp_names_file_valid
    p = NameProcessor(config)
    names = p._load_raw_data()
    assert isinstance(names, list)
    assert all(isinstance(item, str) for item in names)
    assert len(names) == 4
    assert "München\n" in names


def test_load_raw_data_invalid(temp_names_file_invalid):
    config = DataConfig()
    config.input_file = temp_names_file_invalid
    p = NameProcessor(config)
    names = p._load_raw_data()
    assert len(names) == 1
    assert "Berlin\n" in names


def test_is_valid_name():
    """ test name len boundaries from DataConfig """
    config = DataConfig()
    p = NameProcessor(config)
    assert p._is_valid_name("hah")
    assert p._is_valid_name("VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhnot\n")
    assert not p._is_valid_name("hu")
    assert not p._is_valid_name("VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhhhhh\n")


def test_shuffle_names(temp_names_file_valid):
    config = DataConfig()
    config.input_file = temp_names_file_valid
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


def test_vocabulary_size(vocab_test_cases):
    """ test that vocabulary counts unique characters correctly """
    config = DataConfig()
    p = NameProcessor(config)
    p._build_vocabulary(vocab_test_cases["duplicates"])
    assert p.vocab_size == 4  # \n, a, b, c
    p._build_vocabulary(vocab_test_cases["basic"])
    assert p.vocab_size == 7  # \n, a, b, c, d, e, f


def test_vocabulary_mappings(vocab_test_cases):
    """ test that stoi and itos are consistent inverses """
    config = DataConfig()
    p = NameProcessor(config)
    p._build_vocabulary(vocab_test_cases["basic"])
    # test bidirectional mapping
    for char, idx in p.stoi.items():
        assert p.itos[idx] == char


def test_encode_decode_roundtrip(vocab_test_cases):
    """ test that encode/decode works correctly """
    config = DataConfig()
    p = NameProcessor(config)
    p._build_vocabulary(vocab_test_cases["german"])
    test_string = "München\n"
    encoded = p.encode(test_string)
    decoded = p.decode(encoded)
    assert decoded == test_string
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)


def test_create_splits():
    """
    - test for splitting up list of ints into 80/10/10 splits 
    - names: List[int]) -> Tuple[List[int], List[int], List[int]
    """
    config = DataConfig()
    p = NameProcessor(config)
    # check boundaries of sample_config
    assert p.config.train_size + p.config.dev_size + p.config.test_size == 1.0
    idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    splits = p._create_splits(idx)
    assert isinstance(splits, tuple)
    assert all(isinstance(item, list) for item in splits)
    assert all(isinstance(idx, int) for item in splits for idx in item)
    assert len(splits[0]) == 8 and len(splits[1]) == 1 and len(splits[2]) == 1


def test_export_data(tmp_path, temp_names_file_valid):
    """
    - splits: Tuple[List[int], List[int], List[int]]) -> None
    - tests execute() method equally E2E
    """
    config = DataConfig()
    config.input_file = temp_names_file_valid
    config.output_dir = str(tmp_path)
    p = NameProcessor(config)
    p.execute()
    # check if bin & metadata paths exist
    assert (tmp_path / "train.bin").exists()
    assert (tmp_path / "dev.bin").exists()
    assert (tmp_path / "test.bin").exists()
    assert (tmp_path / "meta.pkl").exists()
    # check bin file contents
    train_data = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    dev_data = np.fromfile(tmp_path / "dev.bin", dtype=np.uint16)
    test_data = np.fromfile(tmp_path / "test.bin", dtype=np.uint16)
    assert len(train_data) > 0 and len(dev_data) > 0 and len(test_data) > 0
    # check meta file contents
    with open(tmp_path / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert "vocab_size" in meta
    assert "itos" in meta
    assert "stoi" in meta

