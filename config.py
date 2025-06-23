from dataclasses import dataclass

""" 
Bavarian City Name GPT // config classes for:
- training
- sampling
- data processing
"""


@dataclass
class TrainConfig:
    """ training configuration"""
    batch_size: int = 64
    learning_rate: float = 3e-4
    train_iter: int = 10000
    eval_iter: int = 150
    eval_interval: int = 500
    device: str = "mps"
    # dir with bin / meta files for training
    data_dir: str = "data"
    # flag to save model
    save_model: bool = True
    model_save_dir: str = "saved_models"
    model_name: str = "bavGPT"
    # seed for torch
    seed: int = 42
    # create samples after training optionally
    sample_after_train: bool = True
    num_samples: int = 20


@dataclass
class SampleConfig:
    """ 
    - sampling configuration 
    - samples are always saved as .txt in model dir
    """
    device: str = "mps"
    model_path: str = "saved_models/bavGPT_20250623_115225/model.pt"
    num_samples: int = 50
    max_length: int = 50
    temperature: float = 1.0
    seed: int = 42


@dataclass
class DataConfig:
    """
    - path to load in raw input data
    - path to load processed datasets so
    - other data processing config
    """
    # data processing
    input_file: str = "data/names.txt"
    output_dir: str = "data"
    # seed for shuffling names
    seed: int = 42
    # raw data validation
    min_name_length: int = 3
    max_name_length: int = 50
    # split sizes for train / dev; rest it test
    train_size: float = 0.8
    dev_size: float = 0.9
