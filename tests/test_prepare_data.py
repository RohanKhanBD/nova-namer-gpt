from config import DataConfig, TrainConfig
from prepare_data import NameProcessor
import pytest
import tempfile
import os
import numpy as np
import pickle


@pytest.fixture
def temp_names_file_valid():
    test_data = "München\nNür\nAugsburg\n" + "VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhnot\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_data)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_names_file_invalid():
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
    with pytest.raises(AssertionError):
        p = NameProcessor(TrainConfig())


def test_NameProcessor_init():
    p = NameProcessor(DataConfig())
    assert isinstance(p.config, DataConfig)
    assert p.config.input_file.endswith("txt")
    assert p.vocab_size == 0


def test_load_raw_data_no_input_file():
    with pytest.raises(AssertionError):
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
    config.input_file = str(temp_names_file_invalid)
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
    config.input_file = str(temp_names_file_valid)
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
    config = DataConfig()
    p = NameProcessor(config)
    p._build_vocabulary(vocab_test_cases["basic"])
    # test bidirectional mapping
    for char, idx in p.stoi.items():
        assert p.itos[idx] == char


def test_encode_decode_roundtrip(vocab_test_cases):
    config = DataConfig()
    p = NameProcessor(config)
    p._build_vocabulary(vocab_test_cases["german"])
    test_string = "München\n"
    encoded = p.encode(test_string)
    decoded = p.decode(encoded)
    assert decoded == test_string
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)


def test_unknown_character_in_encode():
    config = DataConfig()
    p = NameProcessor(config)
    p._build_vocabulary(["abc\n"])
    with pytest.raises(KeyError):
        p.encode("xyz")


def test_create_splits():
    config = DataConfig()
    p = NameProcessor(config)
    idx = list(range(1, 11))
    train, dev, test = p._create_splits(idx)
    assert len(train) == 8 and len(dev) == 1 and len(test) == 1
    assert train + dev + test == idx


def test_export_data(tmp_path, temp_names_file_valid):
    """ equally E2E test of execute() method """
    config = DataConfig()
    config.input_file = str(temp_names_file_valid)
    config.output_dir = str(tmp_path)
    p = NameProcessor(config)
    p.execute()
    # check if bin & metadata paths exist
    for filename in ["train.bin", "dev.bin", "test.bin", "meta.pkl"]:
        assert (tmp_path / filename).exists()
    # check bin file contents
    for split in ["train", "dev", "test"]:
        data = np.fromfile(tmp_path / f"{split}.bin", dtype=np.uint16)
        assert len(data) > 0
    # check meta file contents
    with open(tmp_path / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert all(key in meta for key in ["vocab_size", "itos", "stoi", "training_names"])
    assert meta["vocab_size"] > 0

