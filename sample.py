import os
import torch
import torch.nn.functional as F
import pickle
import json
import sys
import argparse
from datetime import datetime
from config import SampleConfig
from model import GPTconfig, GPT
from typing import List, Dict, Tuple


"""
Bavarian City Name GPT // inference script
- sampling from saved models and sample after training
- both options use NameGPTSampler class
- setup params in config.py
"""


class NameGPTSampler:

    """
    - generate names from 2 class_method entries: in-mem model (trainer) / saved checkpoint (CLI)
    - samples from in-mem model are only print; from saved checkpoint are always saved as .txt
    - from_training: pass call further as is for in-mem model with added save_samples argument
    - from_saved_model: load model from checkpoint and pass with added save_samples argument
    - 
    """

    @classmethod
    def from_training(cls, sample_config: SampleConfig, model_dir: str, model: GPT):
        return cls(sample_config, model_dir, model, save_samples=False)

    @classmethod
    def from_saved_model(cls, sample_config: SampleConfig, model_dir: str):
        model = cls._load_model(model_dir, sample_config.device)
        return cls(sample_config, model_dir, model, save_samples=True)

    def __init__(
        self,
        sample_config: SampleConfig,
        model_dir: str,
        model: GPT,
        save_samples: bool,
    ):
        assert isinstance(sample_config, SampleConfig), "Invalid sample config type."
        self.config = sample_config
        self.model_dir = model_dir
        self.device = sample_config.device if torch.backends.mps.is_available() else "cpu"
        self.model = model
        # load itos with same process for both entry points
        self.itos: Dict = self._load_vocab(self.model_dir)
        self.save_samples: bool = save_samples

    @staticmethod
    def _load_model(model_dir: str, device) -> Tuple[GPT, Dict]:
        """ load saved model from checkpoint"""
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            saved_config = json.load(f)
        model_filename = saved_config["train_config"]["model_filename"]
        model_path = os.path.join(model_dir, model_filename)
        model_config_dict = saved_config["model_config"]
        model_config = GPTconfig(**model_config_dict)
        model = GPT(model_config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model
    
    @staticmethod
    def _load_vocab(model_dir: str) -> Dict:
        """ used in both sample_after_train and inference mode """
        # load config.json in dir of saved model
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            saved_config = json.load(f)
        # get in there location of data_dir, the reference to the meta.pkl containing vocab
        data_dir = saved_config["train_config"]["data_dir"]
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        return meta["itos"]


    def _generate_single_name(self) -> str:
        """ generate a single name with current model; single name can have max_length chars """
        name = []
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device) # (B,T)
        for _ in range(self.config.max_length):
            context_cut = context[:, -self.model.config.context_len:]
            logits, _ = self.model(context_cut)
            # take last time step by -1, squeeze this dim away
            logits = logits[:, -1, :] / self.config.temperature  # (B,C)
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            # stop on linebreak
            if idx == 0:
                break
            name.append(self.itos[idx])
            context = torch.cat([context, torch.tensor([[idx]], device=self.device)], dim=1)
        return "".join(name)

    @torch.no_grad()
    def generate(self, num_samples: int) -> List[str]:
        """
        - sample n amount of names and print results
        - when sampling from checkpoint: additionally save to file
        """
        print(f"\nGenerating {num_samples} sample names:")
        self.model.eval()
        names = []
        for i in range(num_samples):
            name = self._generate_single_name()
            names.append(name)
            print(f"{i+1:2d}. {name}")
        if self.save_samples:
            self._save_samples(names)
        self.model.train()
        return names

    def _save_samples(self, samples: List[str]) -> None:
        """
        - save generated samples to a text file with sequential numbering and line breaks
        - filename format: samples_YYYYMMDD_HHMMSS.txt derived from config.py getter
        """
        samples_dir = os.path.join(self.model_dir, self.config.saved_samples_root)
        os.makedirs(samples_dir, exist_ok=True)
        filepath = os.path.join(samples_dir, self.config.save_sample_filename)
        # Format samples with sequential numbering and save to .txt
        with open(filepath, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples, 1):
                f.write(f"{i}. {sample}\n")
        print(f"Samples saved to: {filepath}")


def main():
    """central entry point with command line arguments"""
    if len(sys.argv) > 3:
        print("Too many command line arguments. Only --outdir + value accepted.")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="Specify model_dir to sample from.")
    # demo path as default
    parser.add_argument("--out_dir", default="saved_models/demo")
    args = parser.parse_args()
    # pass the model_path directly to the sampler
    sample_config = SampleConfig()
    sampler = NameGPTSampler.from_saved_model(sample_config, model_dir=args.out_dir)
    names = sampler.generate(sample_config.num_samples)
    print(f"\nGenerated {len(names)} Bavarian city names.")


if __name__ == "__main__":
    main()
