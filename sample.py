"""
NovaNamerGPT Text Generation and Sampling

This module handles text generation from trained GPT models with support for
both in-memory models (post-training) and saved checkpoints (standalone inference).
Features temperature-controlled sampling, novelty enforcement, and automatic file management.

Classes:
    NameGPTSampler: Text generation engine with flexible model loading and output management
"""

import os
import torch
import torch.nn.functional as F
import pickle
import json
import argparse
from config import SampleConfig
from model import GPTconfig, GPT
from typing import List, Dict, Tuple, Any


class NameGPTSampler:
    """
    - generate names from 2 class_method entries: in-mem model (trainer) / saved checkpoint (CLI)
    - samples from in-mem model are only print; from saved checkpoint are always saved as .txt
    - from_training: pass call further as is for in-mem model with added save_samples argument
    - from_saved_model: load model from checkpoint and pass with added save_samples argument
    """

    @classmethod
    def from_training(cls, sample_config: SampleConfig, model_dir: str, model: GPT):
        """
        create sampler from in-memory model after training
        - used by training pipeline for immediate sample generation
        - skips novelty enforcement and file saving
        - load metadata once and pass it to obj creation -> consistency to "from_saved_model"-flow
        """
        metadata = cls._load_meta(model_dir, load_training_names=False)
        return cls(
            sample_config,
            model_dir,
            model,
            metadata,
            enforce_novelty=False,
            save_samples=False
        )

    @classmethod
    def from_saved_model(cls, sample_config: SampleConfig, model_dir: str):
        """
        create sampler from saved model checkpoint
        - loads model state from disk for standalone inference
        - enables always file saving and novelty enforcement depending on sample_config
        - load metadata & model, pass it further to obj creation
        """
        metadata = cls._load_meta(model_dir, load_training_names=sample_config.enforce_novelty)
        model = cls._load_model(model_dir, sample_config.device, metadata)
        return cls(
            sample_config,
            model_dir,
            model,
            metadata,
            enforce_novelty=sample_config.enforce_novelty,
            save_samples=True
        )

    def __init__(
        self,
        sample_config: SampleConfig,
        model_dir: str,
        model: GPT,
        metadata: Dict[str, Any],
        enforce_novelty: bool,
        save_samples: bool,
    ):
        assert isinstance(sample_config, SampleConfig), "Invalid sample config type."
        self.config = sample_config
        self.model_dir = model_dir
        # fallback to cpu if requested device unavailable
        self.device = sample_config.device if torch.backends.mps.is_available() else "cpu"
        self.model = model
        self.enforce_novelty: bool = enforce_novelty  # set depending on entry point & SampleConfig
        self.save_samples: bool = save_samples  # flag set depending on entry point (cls methods)
        # load vocabulary and - if necessary - training data for novelty checking
        self.itos: Dict = metadata["itos"]
        self.training_names = set(metadata.get("training_names", ())) if enforce_novelty else set()
        # populated after saving for samples from saved model checkpoints
        self.saved_samples_path: str = ""

    @staticmethod
    def _load_meta(model_dir: str, load_training_names: bool) -> Dict[str, Any]:
        """
        load metadata from model directory
        - reads config.json for model configuration
        - loads meta.pkl for vocabulary mappings
        - optionally loads training names for novelty checking
        """
        # load model configuration
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        # load vocabulary metadata
        with open(os.path.join(config["train_config"]["data_dir"], "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        result = {"config": config, "itos": meta["itos"]}
        # optionally include training names for duplicate detection
        if load_training_names:
            result["training_names"] = meta["training_names"]
        return result

    @staticmethod
    def _load_model(model_dir: str, device: str, metadata: Dict) -> Tuple[GPT, Dict]:
        """
        load trained model from checkpoint using pre-loaded metadata
        - reconstructs model architecture from provided config
        - loads state dict and moves to target device
        """
        cfg = metadata["config"]
        # reconstruct model architecture
        model_filename = cfg["train_config"]["model_filename"]
        model_config = GPTconfig(**cfg["model_config"])
        model = GPT(model_config)
        # load trained weights
        model_path = os.path.join(model_dir, model_filename)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model

    def _gen_single_name(self) -> str:
        """
        generate single text sample using autoregressive sampling
        - starts with empty context and builds character by character
        - stops at newline character or maximum length
        - uses temperature-controlled multinomial sampling
        """
        name = []
        # start with empty context (single zero token)
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)  # (B,T)

        # inference for max len chars - if not intercepted by learned linebreak char
        for _ in range(self.config.max_length):

            # truncate context to model's maximum length
            context_cut = context[:, -self.model.config.context_len:]

            # get next token logits
            logits, _ = self.model(context_cut)
            # take last time step by -1, squeeze this dim away
            logits = logits[:, -1, :] / self.config.temperature  # (B,C)

            # sample next token
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()

            # stop generation at newline character (index 0)
            if idx == 0:
                break

            # add character to name and update context
            name.append(self.itos[idx])
            context = torch.cat([context, torch.tensor([[idx]], device=self.device)], dim=1)

        return "".join(name)

    @torch.no_grad()
    def generate(self) -> List[str]:
        """
        generate multiple text samples with optional novelty enforcement
        - produces requested number of unique samples
        - optionally filters out training data duplicates
        - when sampling from checkpoint: additionally save to file
        """
        print(f"\nGenerating {self.config.num_samples} sample names:")
        self.model.eval()
        names = []
        # setup duplicate counter & inform user about novelty enforcement
        if self.enforce_novelty:
            duplicate_counter = 0
            print("1to1 duplicates to training data are discarded.\n")
        # generate until enough unique samples are produced
        while len(names) < self.config.num_samples:
            name = self._gen_single_name()
            # check for duplicates; if duplicate discard, reset counter & jump to next iteration
            if self.enforce_novelty and name in self.training_names:
                duplicate_counter += 1
                continue
            names.append(name)
            print(f"{len(names):2d}. {name}")
        if self.enforce_novelty:
            print(f"A total of {duplicate_counter} duplicates were discarded.")
        # save samples to file for "from_saved_model" requests
        if self.save_samples:
            self.saved_samples_path = self._save_samples(names)
        self.model.train()
        print(f"\nGenerated {len(names)} Bavarian city names.")
        return names

    def _save_samples(self, samples: List[str]) -> str:
        """
       save generated samples to timestamped text file
        - creates samples directory within model directory
        - formats samples with sequential numbering
        - uses timestamp for unique filenames
        """
        samples_dir = os.path.join(self.model_dir, self.config.saved_samples_root)
        os.makedirs(samples_dir, exist_ok=True)
        filepath = os.path.join(samples_dir, self.config.save_sample_filename)
        # format samples with sequential numbering
        with open(filepath, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples, 1):
                f.write(f"{i}. {sample}\n")
        print(f"Samples saved to: {filepath}")
        return filepath


def main():
    """
    command line interface for standalone text generation
    - accepts model directory path as ONLY argument
    - uses demo model dir as default for easy sampling
    - all other sampling options must be set in SampleConfig
    """
    parser = argparse.ArgumentParser(description="generate samples from trained model.")
    # demo path as default
    parser.add_argument(
        "out_dir",
        nargs="?",
        default="saved_models/demo",
        help="Path to saved model directory (default: saved_models/demo).",
    )
    args = parser.parse_args()

    # create sampler and generate samples
    sampler = NameGPTSampler.from_saved_model(SampleConfig(), model_dir=args.out_dir)
    sampler.generate()


if __name__ == "__main__":
    main()
