
import os
import torch
import torch.optim as Optim
import numpy as np
import pickle
from config import TrainConfig
from model import GPTconfig, GPT
from typing import Tuple, Dict

""" Bavarian City Name GPT // lightweight training scrip """


class NameGPTTrainer:

    def __init__(self, train_config: TrainConfig, model_config: GPTconfig):
        self.train_config = train_config
        self.model_config = model_config
        # set device mps; if not available cpu
        self.device = train_config.device if torch.backends.mps.is_available() else "cpu"
        # load data
        self.train_split, self.dev_split = self._load_data()
        # init model
        self.model = self._init_model()
        # init optimizer
        self.optim = Optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)
        self._print_model_stats()
    
    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ load train and val data"""
        data_dir = self.train_config.data_dir
        # Load binary data
        train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        dev_data = np.memmap(os.path.join(data_dir, 'dev.bin'), dtype=np.uint16, mode='r')
        # Convert to torch tensors
        train_split = torch.from_numpy(train_data.astype(np.int64))
        dev_split = torch.from_numpy(dev_data.astype(np.int64))
        # Load vocabulary metadata
        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # Update model config with actual vocab size
        self.model_config.vocab_size = meta["vocab_size"]
        print(f"Loaded data: train={len(train_split):,} tokens, dev={len(dev_split):,} tokens")
        print(f"Vocabulary size: {meta['vocab_size']}")
        return train_split, dev_split
    
    def _init_model(self) -> GPT:
        """ init GPT model with updated configs and port to device"""
        model = GPT(self.model_config)
        m = model.to(self.device)
        return m
    
    def _print_model_stats(self) -> None:
        """ print total model params after model init"""
        total_params = sum(p.nelement() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def _get_batch(self, split: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ delivers a batch of X, Y tensors for specified split """
        batch_borders = torch.randint(
            0,
            len(split) - self.model_config.context_len,
            (self.train_config.batch_size, )
        )
        x = torch.stack([split[t: t+self.model_config.context_len] for t in batch_borders])
        y = torch.stack([split[t+1: t+self.model_config.context_len+1] for t in batch_borders])
        return x, y
    
    @torch.no_grad()
    def check_loss(self) -> Dict:
        """ 
        - validate loss outsite backprop; 
        - called from training function after defined training steps 
        """
        # activate eval mode
        self.model.eval()
        out = {}
        # calc train & dev loss as averages after defined eval steps
        for split in [self.train_split, self.dev_split]:
            losses = torch.zeros(self.train_config.eval_iter)
            # calc loss for every batch and save result into tensor
            for i in range(self.train_config.eval_iter):
                x, y = self._get_batch(split)
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train_model(self):
        """ train model over defined train steps """
        # seed for torch from train_config
        torch.manual_seed(self.train_config.seed)
        # training loop
        for i in range(self.train_config.train_iter):
            
            # eval loss & print after certain amount of train steps
            if i % self.train_config.eval_interval == 0:
                losses = self.check_loss()
                print(f"loss after {i} iterations: train_loss {losses[self.train_split]}; eval_loss {losses[self.dev_split]}")
            
            # forward pass
            Xtr, Ytr = self._get_batch(self.train_split)
            Xtr, Ytr = Xtr.to(self.device), Ytr.to(self.device)
            _, loss = self.model(Xtr, Ytr)

            # backward pass
            self.optim.zero_grad()
            loss.backward()

            # update params
            self.optim.step()

            break


def main():
    """ main entry point"""
    train_config = TrainConfig()
    model_config = GPTconfig()
    trainer = NameGPTTrainer(train_config, model_config)
    trainer.train_model()



if __name__ == "__main__":
    main()
