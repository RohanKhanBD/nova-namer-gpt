import os
import torch
import torch.nn.functional as F
import pickle
import json
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
    
    def __init__(self, sample_config: SampleConfig, model=None, itos=None, device=None):
        self.config = sample_config
        # if device not as argument from training, take from sample config
        self.device = device or (sample_config.device if torch.backends.mps.is_available() else "cpu")
        # use provided model and vocab from training
        if model is not None and itos is not None:
            self.model, self.itos = model, itos
        else:
            # Load from file (standalone usage)
            self.model, self.itos = self._load_model()
    
    def _load_model(self) -> Tuple[GPT, Dict]:
        """ load saved model and metadata """
        # load model config from JSON
        config_path = os.path.dirname(self.config.model_path)
        with open(os.path.join(config_path, "config.json"), 'r') as f:
            saved_config = json.load(f)
        # create model config
        model_config_dict = saved_config["model_config"]
        model_config = GPTconfig(**model_config_dict)
        # Load vocabulary
        data_dir = saved_config["train_config"]["data_dir"]
        with open(os.path.join(data_dir, 'meta.pkl'), "rb") as f:
            meta = pickle.load(f)
        # Init and load model
        model = GPT(model_config)
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model, meta["itos"]

    def generate(self, num_samples: int, print_results: bool = False) -> List[str]:
        """ 
        generate names:
        - context: start always with 0 context for linebreak as first char; 
        forward pass expects shape of (1, 1) to work
        - single name can have max_length chars
        - crop idx to the last block_size tokens; if longer idx than 
        context_len is used -> no sense & generation index errors

         
        """
        torch.manual_seed(self.config.seed)
        out = []
        # sample num_sample new names
        for i in range(num_samples):
            name = []
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)  # B,T
            for _ in range(self.config.max_length):
                # context cut
                context_cut = context[:, -self.model.config.context_len:]
                logits, _ = self.model(context_cut)
                # take last time step by -1, squeeze this dim away
                logits = logits[:, -1, :] / self.config.temperature  # B,C
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).item()
                # stop sampling name when linebreak
                if idx == 0:
                    break    
                name.append(self.itos[idx])
                # append sampled index as tensor to the running sequence along time / sequence axis
                context = torch.cat([context, torch.tensor([[idx]], device=self.device)], dim=1)
            
            generated_name = "".join(name)
            out.append(generated_name)
            
            if print_results:
                print(f"{i+1:2d}. {generated_name}")
        
        return out     


def main():
    """ central entry point"""
    config = SampleConfig()
    sampler = NameGPTSampler(config)
    names = sampler.generate(config.num_samples, print_results=True)
    print(f"\nGenerated {len(names)} Bavarian city names.")
    print(names)


if __name__ == "__main__":
    main()












#     # generate names of tbd amount; name ends at first line break char
#     def generate(self, amount_names):
#         out = []
#         for _ in range(amount_names):
#             name = []
#             # start always with 0 context for linebreak as first char; forward pass expects shape of (1, 1) to work
#             context = torch.zeros((1, 1), dtype=torch.long)
#             context = context.to(device)
#             while True:
#                 # context must not be greater than context_len, otherwise mat mul in forward pass does not work; cut max latest context
#                 context_cut = context[:, -context_len:]
#                 logits, _ = self(context_cut)
#                 # grab logits at last timestep
#                 logits = logits[:, -1, :]
#                 logits = F.softmax(logits, dim=-1)
#                 idx = torch.multinomial(logits, num_samples=1, replacement=True).item()
#                 name.append(itos[idx])
#                 # end name gen when first linebreak is sampled
#                 if idx == 0:
#                     break
#                 else:
#                     # as long as no linebreak is hit, add last idx to context and sample next char for name
#                     context = torch.cat((context, torch.tensor([[idx]], dtype=torch.long, device=device)), dim=1)
#             out.append("".join(name))
#         return out


# # sample from model with amount names
# m.generate(50)