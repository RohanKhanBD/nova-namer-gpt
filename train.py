
import os
from datetime import datetime
import torch
import torch.optim as Optim
import numpy as np
import pickle
import json
from dataclasses import asdict
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
    
    def _save_model(self, final_train_loss: float, final_dev_loss: float, training_time: float) -> None:
        """
        save trained model with configuration metadata
        """
        # Create timestamp for unique model naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(
            self.train_config.model_save_dir, 
            f"{self.train_config.model_name}_{timestamp}"
        )
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        # save model_dir for _sample_after_train
        self.model_dir = model_dir
        # Save model checkpoint
        model_path = os.path.join(model_dir, "model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, model_path)
        # Save configuration and metadata as JSON
        config_data = {
            "train_config": asdict(self.train_config),
            "model_config": asdict(self.model_config),
            "training_metadata": {
                "final_train_loss": float(final_train_loss),
                "final_dev_loss": float(final_dev_loss),
                "training_time": round(training_time / 60, 2),
                "total_parameters": sum(p.nelement() for p in self.model.parameters()),
                "device_used": str(self.device),
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__
            }
        }
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"Model saved to: {model_dir}")
        print(f"  - Model checkpoint: {model_path}")
        print(f"  - Configuration: {config_path}")
    
    def train_model(self):
        """ train model over defined train steps """
        # Record training start time
        start_time = datetime.now()
        # seed for torch from train_config
        torch.manual_seed(self.train_config.seed)
        # training loop
        for i in range(self.train_config.train_iter):

            # eval loss & print after certain amount of train steps
            if i % self.train_config.eval_interval == 0:
                losses = self.check_loss()
                print(f"loss after {i} iterations: train_loss {losses[self.train_split]:.4f}; eval_loss {losses[self.dev_split]:.4f}")
            
            # forward pass
            Xtr, Ytr = self._get_batch(self.train_split)
            Xtr, Ytr = Xtr.to(self.device), Ytr.to(self.device)
            _, loss = self.model(Xtr, Ytr)

            # backward pass
            self.optim.zero_grad()
            loss.backward()

            # update params
            self.optim.step()

            #break

        # final evaluation after training
        final_losses = self.check_loss()
        print(f"Final losses: train_loss {final_losses[self.train_split]:.4f}; eval_loss {final_losses[self.dev_split]:.4f}")
        # calculate training time
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        print(f"Training completed in {training_time:.2f} seconds")
        # save model if enabled with config flag
        if self.train_config.save_model:
            self._save_model(
                final_train_loss=final_losses[self.train_split],
                final_dev_loss=final_losses[self.dev_split],
                training_time=training_time
            )
        # sample after training if enabled
        if self.train_config.sample_after_train:
            self._sample_after_train()
    
    def _sample_after_train(self):
        """
        - generate optionally samples after training with sample.py 
        - samples of this mode are only print, not saved
        """
        from sample import NameGPTSampler
        from config import SampleConfig
        print(f"\nGenerating {self.train_config.num_samples} sample names:")
        # Load vocabulary for sampler
        data_dir = self.train_config.data_dir
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        # Create sampler with current model
        sample_config = SampleConfig()
        sample_config.seed = self.train_config.seed
        sampler = NameGPTSampler(
            sample_config,
            model=self.model,
            itos=meta["itos"],
            device=self.device,
        )
        # Generate and print names
        self.model.eval()
        # pass model dir 
        sampler.generate(self.train_config.num_samples)
        self.model.train()


def main():
    """ main entry point"""
    train_config = TrainConfig()
    model_config = GPTconfig()
    trainer = NameGPTTrainer(train_config, model_config)
    trainer.train_model()


if __name__ == "__main__":
    main()
