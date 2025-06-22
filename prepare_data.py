import os
import random
import numpy as np
from config import DataConfig
from typing import Dict, List, Tuple
import pickle


""" 
Bavarian City Name GPT // data prep & encoding
- read in dataset of 60k bavarian city names and do some processing
- create encoding / decoding functions
- save train / dev / test splits as np bin files

"""


class NameProcessor:
    """ processes bavarian names dataset for GPT training """

    def __init__(self, config: DataConfig):
        self.config = config
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size: int = 0

    def _load_raw_data(self) -> List[str]:
        """
        - loads all names into memory as list of str
        - checks names for len boundaries
        - shuffles them with seed from config
        """
        print(f"Loading data from {self.config.input_file}")
        # throw error if file not found
        if not os.path.exists(self.config.input_file):
            raise FileNotFoundError(f"Data file not found: {self.config.input_file}")
        # load all names into memory
        with open(self.config.input_file, mode="r", encoding="utf-8") as file:
            names = file.readlines()
        # check names for certain criteria
        names = [name for name in names if self._is_valid_name(name)]
        # shuffle with seed
        random.seed(self.config.seed)
        random.shuffle(names)
        # print and return valid names
        print(f"Loaded {len(names)} valid names")
        print(f"Sample of first 5 names: {names[:5]}")
        return names

    def _is_valid_name(self, name: str) -> bool:
        """ simple check if name meets certain criteria; as of now, check only len """
        if len(name) < self.config.min_name_length or len(name) > self.config.max_name_length:
            return False
        else:
            return True
        
    def _build_vocabulary(self, names: List[str]) -> None:
        """ creates mapping dicts from all distinct chars """
        all_chars = list(sorted(set([("".join(char)) for name in names for char in name])))
        # create mappings
        self.itos = {i: s for i, s in enumerate(all_chars)}
        self.stoi = {s: i for i, s in self.itos.items()}
        self.vocab_size = len(all_chars)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"All Characters: {repr(''.join(all_chars))}")

    def encode(self, text: str) -> List[int]:
        """ encodes str into list of indexes with mapping dict """
        return [self.stoi[i] for i in text]
    
    def decode(self, indexes: List[int]) -> str:
        """ returns joined string for list of indexes with mapping dict """
        return "".join([self.itos[i] for i in indexes])
    
    def _create_splits(self, names: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """ split names into train / dev / test """
        boundary_1 = int(self.config.train_size * len(names))
        boundary_2 = int(self.config.dev_size * len(names))
        train = names[:boundary_1]
        dev = names[boundary_1: boundary_2]
        test = names[boundary_2:]
        print(f"Train has {len(train):,} tokens")
        print(f"Dev has {len(dev):,} tokens")
        print(f"Test has {len(test):,} tokens")
        return train, dev, test
    
    def _export_data(self, splits: Tuple[List[int], List[int], List[int]]) -> None:
        """
        - convert splits into np uint16
        - save processed data to bin files at dir defined in config
        - save related metadata
        """
        train, dev, test = splits
        # convert into np uint16
        train = np.array(train, dtype=np.uint16)
        dev = np.array(dev, dtype=np.uint16)
        test = np.array(test, dtype=np.uint16)
        # save binary data
        os.makedirs(self.config.output_dir, exist_ok=True)
        train_path = os.path.join(self.config.output_dir, "train.bin")
        # create pathes
        dev_path = os.path.join(self.config.output_dir, "dev.bin")
        test_path = os.path.join(self.config.output_dir, "test.bin")
        # export
        train.astype(np.uint16).tofile(train_path)
        dev.astype(np.uint16).tofile(dev_path)
        test.astype(np.uint16).tofile(test_path)
        # export metadata
        meta = {
            "vocab_size": self.vocab_size,
            "itos": self.itos,
            "stoi": self.stoi,
        }
        with open(os.path.join(self.config.output_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    
    def execute(self) -> None:
        """
        - load raw names from file
        - build mapping dicts from names
        - encode the data into indexes from joined string
        - create splits
        - export splits to np bin files & save some metadata
        """
        names = self._load_raw_data()
        self._build_vocabulary(names)
        encoded_data = self.encode("".join(names))
        train, dev, test = self._create_splits(encoded_data)
        self._export_data((train, dev, test))
        # print stats
        print("\nData processing completed successfully!")
        print(f"Total tokens: {len(encoded_data):,}")


def main():
    """
    main entry point; execute data processing by:
        1. creating instance of DataConfig
        2. creating instance of NameProcessor with config
        3. call NameProcessor execute method
    """
    config = DataConfig
    processor = NameProcessor(config)
    processor.execute()


if __name__ == "__main__":
    main()
