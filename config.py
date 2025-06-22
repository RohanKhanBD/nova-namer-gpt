import os
from dataclasses import dataclass

""" 
Bavarian City Name GPT // config classes for:
- training
- sampling
- data processing

"""


@dataclass
class DataConfig:
    """
    - path to load in raw input data
    - path to load processed datasets so
    - other data processing config
    """
    # data processing
    input_file: str = "data/names.txt"
    output_dir: str = "data/"
    # seed for shuffling names
    seed: int = 42
    # raw data validation
    min_name_length: int = 3
    max_name_length: int = 50
    # split sizes for train / dev; rest it test
    train_size: int = 0.8
    dev_size: int = 0.9


