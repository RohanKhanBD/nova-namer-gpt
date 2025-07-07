import pytest
import torch
from config import SampleConfig, TrainConfig
from sample import NameGPTSampler
from train import NameGPTTrainer


""" fixtures for sample_config, train_config, model_config, test_train_data on conftest.py """


def test_NameGPTSampler_init_wrong_config():
    with pytest.raises(AssertionError, match="Invalid sample config type."):
        NameGPTSampler(sample_config=TrainConfig(), model_dir=None, model=None, enforce_novelty=None, save_samples=None)


def test_NameGPTSampler_from_training_init(train_cfg, model_cfg, sample_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    sample_config = sample_cfg
    sample_config.enforce_novelty = True
    s = NameGPTSampler.from_training(sample_config, t.model_save_dir, t.model)
    # called via from_training, cls method must setup obj with certain attributes
    assert not s.enforce_novelty
    assert not s.save_samples
    assert s.training_names == set()


def test_NameGPTSampler_from_saved_model_init(train_cfg, model_cfg, sample_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    sample_config = sample_cfg
    sample_config.enforce_novelty = True
    s = NameGPTSampler.from_saved_model(sample_cfg, t.model_save_dir)
    assert s.enforce_novelty
    assert s.save_samples
    # check if weights are identical in models saved at trainer & sampler
    assert all(torch.equal(p0, p1) for p0, p1 in zip(t.model.state_dict().values(), s.model.state_dict().values()))


def test_NameGPTSampler_load_meta(train_cfg, model_cfg, sample_cfg, mock_train_data):
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    s = NameGPTSampler.from_training(sample_cfg, t.model_save_dir, t.model)
    pass




def test_train_and_create_save_dir(sample_cfg):
    """ dir name created in config property depending on time """
    new_file = sample_cfg.save_sample_filename
    assert new_file.startswith("samples_")
    assert new_file.endswith(".txt")
