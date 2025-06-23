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
- easy sampling from saved models
- optional possible to sample after training
- both options use NameGPTSampler class
- setup params in config.py
"""


class NameGPTSampler:

    def __init__(
        self,
        sample_config: SampleConfig,
        model_path=None,
        model=None,
        itos=None,
        device=None,
    ):
        self.config = sample_config
        # received from main() as default or command-line argument
        self.model_path = model_path
        # if device not as argument from training, take from sample config
        self.device = device or (
            sample_config.device if torch.backends.mps.is_available() else "cpu"
        )
        # determine mode: after_training (with model+itos) vs from_file
        self.is_after_training = model is not None and itos is not None
        if self.is_after_training:
            # Use provided model and vocab from training
            self.model, self.itos = model, itos
        else:
            # Load from file (standalone usage)
            self.model, self.itos = self._load_model()

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
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model, meta["itos"]

    def _save_samples(self, samples: List[str]) -> None:
        """
        save generated samples to a text file
        - creates samples directory in model folder if doesn't exist
        - saves samples with sequential numbering and line breaks
        - filename format: samples_YYYYMMDD_HHMMSS.txt
        """
        # get dir from model path
        model_dir = os.path.dirname(self.model_path)
        samples_dir = os.path.join(model_dir, "samples")
        # Create samples directory if it doesn't exist
        os.makedirs(samples_dir, exist_ok=True)
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"samples_{timestamp}.txt"
        filepath = os.path.join(samples_dir, filename)
        # Format samples with sequential numbering
        formatted_samples = [
            f"{i}. {sample}" for i, sample in enumerate(samples, 1)
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(formatted_samples))
        print(f"Samples saved to: {filepath}")

    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
    ) -> List[str]:
        """
        generate names:
        - context: start always with 0 context for linebreak as first char;
        forward pass expects shape of (1, 1) to work
        - single name can have max_length chars
        - crop idx to the last block_size tokens; if longer idx than
        - context_len is used -> no sense & generation index errors
        - when sampling from file: always save to file
        - when sampling after training: only print, never save
        """

        out = []
        # sample num_sample new names
        for i in range(num_samples):
            name = []
            context = torch.zeros(
                (1, 1), dtype=torch.long, device=self.device
            )  # B,T
            for _ in range(self.config.max_length):
                # context cut
                context_cut = context[:, -self.model.config.context_len :]
                logits, _ = self.model(context_cut)
                # take last time step by -1, squeeze this dim away
                logits = logits[:, -1, :] / self.config.temperature  # B,C
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).item()
                # stop sampling name when linebreak
                if idx == 0:
                    break
                name.append(self.itos[idx])
                # append sampled index as tensor to running sequence along T
                context = torch.cat(
                    [context, torch.tensor([[idx]], device=self.device)], dim=1
                )

            generated_name = "".join(name)
            out.append(generated_name)
            print(f"{i+1:2d}. {generated_name}")

        # Auto-save behavior based on mode
        if not self.is_after_training:
            # Sampling from file -> always save
            self._save_samples(out)

        return out


def main():
    """central entry point with command line arguments"""

    parser = argparse.ArgumentParser(description="Generate Bavarian city names")
    # demo path as default
    parser.add_argument(
        "--out_dir",
        default="saved_models/demo",
        help="Directory containing model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of names to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )

    args = parser.parse_args()
    # build the model path from the directory
    model_path = os.path.join(args.out_dir, "model.pt")
    # create config and set the model path
    config = SampleConfig()
    config.num_samples = args.num_samples
    config.temperature = args.temperature
    # pass the model_path directly to the sampler
    sampler = NameGPTSampler(config, model_path=model_path)
    sampler.config.model_path = model_path
    names = sampler.generate(config.num_samples)
    print(f"\nGenerated {len(names)} Bavarian city names.")


if __name__ == "__main__":
    main()
