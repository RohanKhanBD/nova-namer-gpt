"""
NovaNamerGPT Data Processing Pipeline

This module handles raw text data ingestion, processing, and conversion to binary
format for efficient training. Supports any character-level text dataset with
configurable validation, shuffling, and train/dev/test splitting.

In the default state it works with a .txt file with continuous stream of name strings sep by \n.
Specify the input file and other params in config.py DataConfig class.

Classes:
    NameProcessor: Core data processing pipeline for character-level datasets
"""

import os
import random
import numpy as np
from config import DataConfig
from typing import Dict, List, Tuple, Set
import pickle


class NameProcessor:
    """
    character-level data processing pipeline for text datasets
    - loads and validates raw text data from input file
    - builds character vocabulary with encoding/decoding mappings
    - creates reproducible train/dev/test splits
    - exports binary files and metadata for efficient training
    """

    def __init__(self, config: DataConfig):
        assert isinstance(config, DataConfig), "Invalid config type."
        self.config = config
        self.stoi: Dict[str, int] = {}  # string to index mapping
        self.itos: Dict[int, str] = {}  # index to string mapping
        self.vocab_size: int = 0
        self.rng = random.Random(self.config.seed)  # seeded rng for reproducibility
        self.training_names_set: Set[str] = set()  # for novelty checking during sampling

    def _load_raw_data(self) -> List[str]:
        """
        load all names from input file with validation
        - reads text file line by line preserving newlines
        - filters names based on length criteria
        - returns validated list of names
        """
        print(f"Loading data from {self.config.input_file}")
        assert os.path.exists(self.config.input_file), f"File not found: {self.config.input_file}"
        with open(self.config.input_file, mode="r", encoding="utf-8") as file:
            # preserve newline chars - they're important for model training
            names = file.readlines()
        # filter names based on validation criteria
        names = [name for name in names if self._is_valid_name(name)]
        print(f"Loaded {len(names)} valid names")
        print(f"Sample of first 5 names: {names[:5]}")
        return names

    def _is_valid_name(self, name: str) -> bool:
        """ check if name meets length criteria """
        return self.config.min_name_length <= len(name) <= self.config.max_name_length

    def _shuffle_names(self, names: List[str]) -> List[str]:
        """ shuffle names using seeded rng for reproducibility """
        shuffled_names = names.copy()
        self.rng.shuffle(shuffled_names)
        return shuffled_names

    def _build_vocabulary(self, names: List[str]) -> None:
        """
        create character vocabulary from all names
        - extracts unique characters and sorts them
        - builds bidirectional mapping dictionaries
        - sets vocabulary size for model configuration
        """
        all_chars = sorted(set(''.join(names)))  # get unique chars and sort
        self.itos = {i: s for i, s in enumerate(all_chars)}  # index to string
        self.stoi = {s: i for i, s in self.itos.items()}  # string to index
        self.vocab_size = len(all_chars)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"All Characters: {repr(''.join(all_chars))}")

    def encode(self, text: str) -> List[int]:
        """ convert string to list of character indices """
        return [self.stoi[i] for i in text]

    def decode(self, indexes: List[int]) -> str:
        """ convert list of indices back to string """
        return "".join([self.itos[i] for i in indexes])

    def _create_splits(self, names: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        split encoded data into train/dev/test sets
        - uses configured split ratios from dataconfig
        - returns three lists of token indices
        """
        boundary_1 = int(self.config.train_size * len(names))
        boundary_2 = int((self.config.train_size + self.config.dev_size) * len(names))
        boundary_3 = int((1 - self.config.test_size) * len(names))
        train = names[:boundary_1]
        dev = names[boundary_1:boundary_2]
        test = names[boundary_3:]
        print(f"Train has {len(train):,} tokens")
        print(f"Dev has {len(dev):,} tokens")
        print(f"Test has {len(test):,} tokens")
        return train, dev, test

    def _export_data(self, splits: Tuple[List[int], List[int], List[int]]) -> None:
        """
        save processed data to binary files and metadata
        - converts to uint16 numpy arrays for memory efficiency
        - saves train.bin, dev.bin, test.bin files
        - exports vocabulary mappings and training names set
        """
        split_names = ["train", "dev", "test"]
        # save binary files for fast loading during training
        for name, data in zip(split_names, splits):
            np.array(data, dtype=np.uint16).tofile(
                os.path.join(self.config.output_dir, f"{name}.bin")
            )
        # save metadata for model initialization and sampling
        meta = {
            "vocab_size": self.vocab_size,
            "itos": self.itos,
            "stoi": self.stoi,
            "training_names": self.training_names_set,
        }
        with open(os.path.join(self.config.output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    def execute(self) -> None:
        """
        main processing pipeline execution
        - loads and shuffles raw data
        - builds character vocabulary
        - encodes text to indices and creates splits
        - exports binary files and metadata
        """
        names = self._load_raw_data()
        names = self._shuffle_names(names)
        # store training names for novelty checking during sampling
        self.training_names_set = set(name.strip() for name in names)
        self._build_vocabulary(names)
        encoded_data = self.encode("".join(names))  # join all names into one sequence
        train, dev, test = self._create_splits(encoded_data)
        self._export_data((train, dev, test))
        # print final statistics
        print("\nData processing completed successfully!")
        print(f"Total tokens: {len(encoded_data):,}")


def main() -> None:
    """
    main entry point for data processing
    - creates dataconfig instance with default parameters
    - initializes nameprocessor and runs complete pipeline
    """
    processor = NameProcessor(DataConfig())
    processor.execute()


if __name__ == "__main__":
    main()
