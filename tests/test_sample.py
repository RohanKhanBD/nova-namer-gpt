import pytest
import torch
from config import TrainConfig
from sample import NameGPTSampler
from train import NameGPTTrainer


def test_NameGPTSampler_init_wrong_config():
    with pytest.raises(AssertionError, match="Invalid sample config type."):
        NameGPTSampler(
            sample_config=TrainConfig(),
            model_dir=None, 
            model=None,
            metadata=None,
            enforce_novelty=None,
            save_samples=None
        )


def test_NameGPTSampler_from_training_init(train_cfg, model_cfg, sample_cfg):
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    sample_cfg.enforce_novelty = True
    s = NameGPTSampler.from_training(sample_cfg, t.model_save_dir, t.model)
    # called via from_training, cls method must setup obj with certain attributes
    assert not s.enforce_novelty and not s.save_samples and s.training_names == set()


def test_NameGPTSampler_from_saved_model_init(train_cfg, model_cfg, sample_cfg):
    """ tests also _load_model """
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    sample_cfg.enforce_novelty = True
    s = NameGPTSampler.from_saved_model(sample_cfg, t.model_save_dir)
    assert s.enforce_novelty and s.save_samples
    # check if weights are identical in models saved at trainer & sampler
    assert all(
        torch.equal(p0, p1) for p0, p1 in zip(
            t.model.state_dict().values(), s.model.state_dict().values()
            )
        )


def test_NameGPTSampler_from_training_load_meta(train_cfg, model_cfg, sample_cfg):
    """ coming from training, only itos dict is saved at obj """
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    s = NameGPTSampler.from_training(sample_cfg, t.model_save_dir, t.model)
    assert isinstance(s.itos, dict) and s.itos[1] == "a" and s.training_names == set()


def test_NameGPTSampler_from_saved_model_load_meta(train_cfg, model_cfg, sample_cfg):
    """ coming from saved model, itos dict and training names (if flag set) are saved at obj """
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    sample_cfg.enforce_novelty = True
    s = NameGPTSampler.from_saved_model(sample_cfg, t.model_save_dir)
    assert isinstance(s.itos, dict) and s.itos[1] == "a" and "b" in s.training_names


def test_NameGPTSample_generate(train_cfg, model_cfg, sample_cfg):
    """ tests also gen_single_name """
    t = NameGPTTrainer(train_cfg, model_cfg)
    t.train_model()
    s = NameGPTSampler.from_saved_model(sample_cfg, t.model_save_dir)
    samples = s.generate()
    assert len(samples) == sample_cfg.num_samples
    assert all((len(sample) <= sample_cfg.max_length) for sample in samples)
    # check saved file
    with open(s.saved_samples_path, mode="r") as f:
        saved_samples = f.read()
    assert all((sample in saved_samples) for sample in samples)
    assert all((num in saved_samples) for num in ["1.", "2.", "3."])


def test_train_and_create_save_dir(sample_cfg):
    """ dir name created in config property depending on time """
    new_file = sample_cfg.save_sample_filename
    assert new_file.startswith("samples_") and new_file.endswith(".txt")
