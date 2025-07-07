import pytest
import tempfile
import os
from config import DataConfig


""" fixtures for test_model.py """




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