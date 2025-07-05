import os
import torch
import torch.nn.functional as F
import pickle
import json
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

    # insert class method factories!!
    
    def __init__(
        self,
        sample_config: SampleConfig,
        model_path=None,
        model=None,
        data_dir=None,
    ):
        assert isinstance(sample_config, SampleConfig), "Invalid sample config type."
        self.config = sample_config
        # received from main() as default or command-line argument
        self.model_path = model_path
        # if device not as argument from training, take from sample config
        self.device = sample_config.device if torch.backends.mps.is_available() else "cpu"
        # determine mode: after_training (with model+itos) vs from_file
        self.is_after_training = model is not None and data_dir is not None
        # case 1: sample_after_train -> print some samples after each training run
        if self.is_after_training:
            self.model = model
            self.itos = self._load_vocab(data_dir)
        # case 2: sample from saved model -> infer from loaded model and saves samples as .txt
        else:
            # Load from file (standalone usage)
            self.model, self.itos = self._load_model()

    def _load_vocab(self, data_dir: str) -> Dict:
        """ only used in mode: sample_after_train; load vocab from data dir meta.pkl directly """
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        return meta["itos"]

    def _load_model(self) -> Tuple[GPT, Dict]:
        """load saved model and metadata"""
        # load model config from JSON
        config_path = os.path.dirname(self.model_path)
        with open(os.path.join(config_path, "config.json"), "r") as f:
            saved_config = json.load(f)
        # create model config
        model_config_dict = saved_config["model_config"]
        model_config = GPTconfig(**model_config_dict)
        # Load vocabulary
        data_dir = saved_config["train_config"]["data_dir"]
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        # Init and load model
        model = GPT(model_config)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        return model, meta["itos"]

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
        - sample n amount of names
        - when sampling from file: always save to file
        - when sampling after training: only print, never save
        """
        print(f"\nGenerating {num_samples} sample names:")
        self.model.eval()
        names = []
        for i in range(num_samples):
            name = self._generate_single_name()
            names.append(name)
            print(f"{i+1:2d}. {name}")
        if not self.is_after_training:
            self._save_samples(names)
        self.model.train()
        return names

    def _save_samples(self, samples: List[str]) -> None:
        """
        save generated samples to a text file
        - creates samples directory in model folder if doesn't exist
        - saves samples with sequential numbering and line breaks
        - filename format: samples_YYYYMMDD_HHMMSS.txt
        - samples from "sample_after_train" are not saved, only printed
        """
        model_dir = os.path.dirname(self.model_path)
        samples_dir = os.path.join(model_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(samples_dir, f"samples_{timestamp}.txt")
        # Format samples with sequential numbering and save to .txt
        with open(filepath, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples, 1):
                f.write(f"{i}. {sample}\n")
        print(f"Samples saved to: {filepath}")


def main():
    """central entry point with command line arguments"""

    parser = argparse.ArgumentParser(description="Generate Bavarian city names")
    # demo path as default
    parser.add_argument("--out_dir", default="saved_models/demo")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    # build the model path from the directory
    model_path = os.path.join(args.out_dir, "model.pt")
    # create config and set the model path
    config = SampleConfig()
    config.num_samples = args.num_samples
    config.temperature = args.temperature
    # pass the model_path directly to the sampler
    sampler = NameGPTSampler(config, model_path=model_path)
    names = sampler.generate(config.num_samples)
    print(f"\nGenerated {len(names)} Bavarian city names.")


if __name__ == "__main__":
    main()
