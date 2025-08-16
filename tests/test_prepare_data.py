from config import DataConfig, TrainConfig
from prepare_data import NameProcessor
import pytest
import numpy as np
import pickle


def test_NameProcessor_init_invalid_config():
    with pytest.raises(AssertionError):
        NameProcessor(TrainConfig())


def test_NameProcessor_init(data_cfg):
    p = NameProcessor(data_cfg)
    assert isinstance(p.config, DataConfig)
    assert p.config.input_file.endswith("txt")
    assert p.vocab_size == 0


def test_load_raw_data_no_input_file(data_cfg):
    data_cfg.input_file = "this_file_does_not_exist.txt"
    with pytest.raises(AssertionError):
        NameProcessor(data_cfg)._load_raw_data()


def test_load_raw_data_valid(data_cfg):
    p = NameProcessor(data_cfg)
    names = p._load_raw_data()
    assert isinstance(names, list) and len(names) == 4
    assert all(isinstance(item, str) for item in names)
    assert "München\n" in names


def test_load_raw_data_invalid(data_cfg, mock_names_file_invalid):
    data_cfg.input_file = str(mock_names_file_invalid)
    p = NameProcessor(data_cfg)
    names = p._load_raw_data()
    assert len(names) == 1 and "Berlin\n" in names


def test_is_valid_name(data_cfg):
    """ test name len boundaries from DataConfig """
    p = NameProcessor(data_cfg)
    valid_cases = ["hah", "VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhnot\n"]
    invalid_cases = ["hu", "VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhhhhh\n"]
    assert all(p._is_valid_name(name) for name in valid_cases)
    assert not any(p._is_valid_name(name) for name in invalid_cases)


def test_shuffle_names(data_cfg):
    p1, p2 = NameProcessor(data_cfg), NameProcessor(data_cfg)
    names1, names2 = p1._load_raw_data(), p2._load_raw_data()
    shuffled1, shuffled2 = p1._shuffle_names(names1), p2._shuffle_names(names2)
    assert names1 is not shuffled1 and names2 is not shuffled2
    assert shuffled1 == shuffled2


def test_vocabulary_size(data_cfg, mock_vocab_cases):
    """ test that vocabulary counts unique characters correctly """
    p = NameProcessor(data_cfg)
    p._build_vocabulary(mock_vocab_cases["duplicates"])
    assert p.vocab_size == 4  # \n, a, b, c
    p._build_vocabulary(mock_vocab_cases["basic"])
    assert p.vocab_size == 7  # \n, a, b, c, d, e, f


def test_vocabulary_mappings(data_cfg, mock_vocab_cases):
    p = NameProcessor(data_cfg)
    p._build_vocabulary(mock_vocab_cases["basic"])
    # test bidirectional mapping
    for char, idx in p.stoi.items():
        assert p.itos[idx] == char


def test_encode_decode_roundtrip(data_cfg, mock_vocab_cases):
    p = NameProcessor(data_cfg)
    p._build_vocabulary(mock_vocab_cases["german"])
    test_string = "München\n"
    encoded = p.encode(test_string)
    decoded = p.decode(encoded)
    assert decoded == test_string
    assert isinstance(encoded, list) and all(isinstance(x, int) for x in encoded)


def test_unknown_character_in_encode(data_cfg):
    p = NameProcessor(data_cfg)
    p._build_vocabulary(["abc\n"])
    with pytest.raises(KeyError):
        p.encode("xyz")


def test_create_splits(data_cfg):
    p = NameProcessor(data_cfg)
    idx = list(range(1, 11))
    train, dev, test = p._create_splits(idx)
    assert (len(train), len(dev), len(test)) == (8, 1, 1)
    assert train + dev + test == idx


def test_export_data(data_cfg):
    """ equally E2E test of execute() method """
    p = NameProcessor(data_cfg)
    p.execute()
    # check if bin & metadata paths exist
    expected_files = ["train.bin", "dev.bin", "test.bin", "vocab_meta.pkl"]
    assert all((data_cfg.output_dir / f).exists() for f in expected_files)
    # check bin file contents
    for split in ["train", "dev", "test"]:
        data = np.fromfile(data_cfg.output_dir / f"{split}.bin", dtype=np.uint16)
        assert len(data) > 0
    # check meta file contents
    with open(data_cfg.output_dir / "vocab_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert all(key in meta for key in ["vocab_size", "itos", "stoi", "training_names"])
    assert meta["vocab_size"] > 0


def test_export_data_training_names(data_cfg):
    """ check if name from mock file in training_names vocab_meta.pkl """
    p = NameProcessor(data_cfg)
    p.execute()
    with open(data_cfg.output_dir / "vocab_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    assert "München" in meta["training_names"]


def test_empty_input_file(data_cfg, tmp_path):
    """ handle empty input files gracefully """
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    data_cfg.input_file = str(empty_file)
    p = NameProcessor(data_cfg)
    names = p._load_raw_data()
    assert names == []


def test_vocab_consistency_after_splits(data_cfg):
    """ ensure vocab built from full dataset, not just training split """
    p = NameProcessor(data_cfg)
    p.execute()
    with open(data_cfg.output_dir / "vocab_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    # check that vocab includes characters from all splits
    all_chars = set()
    for split in ["train", "dev", "test"]:
        data = np.fromfile(data_cfg.output_dir / f"{split}.bin", dtype=np.uint16)
        for token in data:
            all_chars.add(meta["itos"][token])
    assert len(all_chars) == meta["vocab_size"]
