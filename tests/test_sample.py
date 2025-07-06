import pytest
import os
import torch
from config import SampleConfig, TrainConfig
from sample import NameGPTSampler
from train import NameGPTTrainer
from test_train import temp_data_files, configs


"""
- configs from test_train is ready to use pytest fixture with mock train_config and model_config
- temp_data_files is ready to use pytest fixture with mock data .bin and metadata.pkl files
"""


@pytest.fixture
def sample_config(tmp_path):
    return SampleConfig(
        device="cpu",
        num_samples=3,
        max_length=10,
        temperature=1.0,
    )


def test_NameGPTSample_init_wrong_config():
    with pytest.raises(AssertionError, match="Invalid sample config type."):
        NameGPTSampler(sample_config=TrainConfig(), model_dir=None, model=None, itos=None, save_samples=None)


def test_NameGPTSample_init_saved_model(temp_data_files, configs, sample_config):
    """ test only init from saved model; _sample_after_train way already tested in test_train.py"""
    train_config, model_config = configs
    train_config.data_dir = str(temp_data_files)
    t = NameGPTTrainer(train_config, model_config)
    t.train_model()
    # take path to which the model was saved to by NameGPTTrainer
    #model_dir = os.path.join(t.model_save_dir, "model.pt")
    s = NameGPTSampler.for_saved_model(sample_config, t.model_save_dir)
    assert all(torch.equal(p0, p1) for p0, p1 in zip(t.model.state_dict().values(), s.model.state_dict().values()))



