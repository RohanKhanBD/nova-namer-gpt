from config import DataConfig, TrainConfig
from prepare_data import NameProcessor
import pytest
import tempfile
import os

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

def test_NameProcessor_init_invalid_config():
    with pytest.raises(TypeError):
        config = TrainConfig()
        processor = NameProcessor(config)

def test_NameProcessor_init():
    config = DataConfig()
    processor = NameProcessor(config)
    assert isinstance(processor.config, DataConfig)
    assert hasattr(processor.config, "input_file")
    assert hasattr(processor, "stoi")
    assert hasattr(processor, "itos")
    assert processor.vocab_size == 0
    assert hasattr(processor, "rng")

def test_load_raw_data_no_input_file():
    with pytest.raises(FileNotFoundError):
        config = DataConfig()
        config.input_file = "this_file_does_not_exist.txt"
        processor = NameProcessor(config)
        processor._load_raw_data()

def test_load_raw_data_valid(temp_names_file_valid):
    config = DataConfig()
    config.input_file = temp_names_file_valid
    processor = NameProcessor(config)
    names = processor._load_raw_data()
    assert len(names) == 4
    assert "München\n" in names

def test_load_raw_data_invalid(temp_names_file_invalid):
    config = DataConfig()
    config.input_file = temp_names_file_invalid
    processor = NameProcessor(config)
    names = processor._load_raw_data()
    assert len(names) == 1
    assert "Berlin\n" in names

def test_is_valid_name():
    config = DataConfig()
    processor = NameProcessor(config)
    assert processor._is_valid_name("hah")
    assert processor._is_valid_name("VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhnot\n")
    assert not processor._is_valid_name("hu")
    assert not processor._is_valid_name("VeryLongCityNameThatExceedsMaxLengthhhhhhhhhhhhhhh\n")



