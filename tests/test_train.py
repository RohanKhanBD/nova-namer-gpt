from config import TrainConfig, DataConfig
from model import GPTconfig, GPT
from train import NameGPTTrainer
import numpy as np
import pickle
import torch
import pytest


@pytest.fixture
def min_model_config():
    return GPTconfig(
        context_len=4,
        vocab_size=10,
        n_embd=8,
        n_head=2,
        n_layer=2,
        dropout=0.2,
        ffw_widen=4,  
        a_bias=True,
        ffw_bias=True,  
        lm_head_bias=False,
    )


@pytest.fixture
def min_train_config():
    return TrainConfig(
        batch_size=2,
        learning_rate=3e-4,
        train_iter=1,
        eval_iter=5,
        eval_interval=500,
        device="mps",
        data_dir="data",
        save_model=True,
        model_save_dir="saved_models",
        model_name="bavGPT",
        seed=42,
        sample_after_train=True,
        num_samples=5,
    )


def test_NameGPTTrainer_init_wrong_configs(min_train_config, min_model_config):
    with pytest.raises(AssertionError, match="Invalid train config type."):
        NameGPTTrainer(train_config=DataConfig(), model_config=min_model_config)
    with pytest.raises(AssertionError, match="Invalid model config type."):
        NameGPTTrainer(train_config=min_train_config, model_config=DataConfig())