from dataclasses import dataclass
import os
import datetime


@dataclass
class TrainConfig:
    """training configuration"""

    batch_size: int = 64
    learning_rate: float = 3e-4
    train_iter: int = 1
    eval_iter: int = 150
    eval_interval: int = 500
    device: str = "mps"
    # dir with bin / meta files for training
    data_dir: str = "data"
    # save model after train
    saved_models_root: str = "saved_models"
    model_name: str = "bavGPT"
    model_filename: str = "model.pt"
    seed: int = 42
    # print samples after training
    num_samples: int = 20

    @property
    def save_dir_current(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.saved_models_root, f"{self.model_name}_{ts}")


@dataclass
class SampleConfig:
    """
    - sampling configuration
    - samples derived from saved models are always saved as .txt in model dir
    - enforce novelty only relevant and configurable at sampling from saved model
    """

    device: str = "mps"
    num_samples: int = 50
    max_length: int = 50
    temperature: float = 1
    # discards 1to1 copies of training data values
    enforce_novelty: bool = True
    saved_samples_root: str = "saved_samples"

    @property
    def save_sample_filename(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"samples_{ts}.txt"


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
    # name validation
    min_name_length: int = 3
    max_name_length: int = 50
    train_size: float = 0.8
    dev_size: float = 0.1
    test_size: float = 0.1
