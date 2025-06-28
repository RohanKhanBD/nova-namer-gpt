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
from typing import Tuple, Dict, List

""" Bavarian City Name GPT // lightweight training scrip """


class NameGPTTrainer:

    def __init__(self, train_config: TrainConfig, model_config: GPTconfig):
        self.train_config = train_config
        self.model_config = model_config
        # set device mps; if not available cpu
        self.device = (
            train_config.device if torch.backends.mps.is_available() else "cpu"
        )
        # load data
        self.train_split, self.dev_split = self._load_data()
        # init components
        self.model_manager = ModelManager(train_config, model_config)
        # init model
        self.model = self.model_manager.create_model()
        self.optimizer = Optim.Adam(
            self.model.parameters(), lr=self.train_config.learning_rate
        )
        self.metrics = TrainingMetrics(
            self.model, self.device, self.train_config.eval_iter
        )
        self._print_model_stats()

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """load train and val data"""
        data_dir = self.train_config.data_dir
        # Load binary data
        train_data = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        dev_data = np.memmap(
            os.path.join(data_dir, "dev.bin"), dtype=np.uint16, mode="r"
        )
        # Convert to torch tensors
        train_split = torch.from_numpy(train_data.astype(np.int64))
        dev_split = torch.from_numpy(dev_data.astype(np.int64))
        # Load vocabulary metadata
        meta_path = os.path.join(data_dir, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        # Update model config with actual vocab size
        self.model_config.vocab_size = meta["vocab_size"]
        print(
            f"Loaded data: train={len(train_split):,} tokens, dev={len(dev_split):,} tokens"
        )
        print(f"Vocabulary size: {meta['vocab_size']}")
        return train_split, dev_split

    def _print_model_stats(self) -> None:
        """print total model params after model init"""
        total_params = self.model.get_num_params()
        print(f"Model parameters: {total_params:,}")

    def _get_batch(self, split: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """delivers a batch of X, Y tensors for specified split"""
        batch_borders = torch.randint(0,
            len(split) - self.model_config.context_len,
            (self.train_config.batch_size,),
        )
        x = torch.stack([split[t: t + self.model_config.context_len] for t in batch_borders])
        y = torch.stack([split[t + 1: t + self.model_config.context_len + 1]for t in batch_borders])
        return x, y

    def train_model(self):
        """train model over defined train steps"""
        # Record training start time
        start_time = datetime.now()
        # seed for torch from train_config
        torch.manual_seed(self.train_config.seed)
        # training loop
        for i in range(self.train_config.train_iter):

            # eval loss & print after certain amount of train steps
            if i % self.train_config.eval_interval == 0:
                losses = self.metrics.evaluate_loss(
                    self.train_split, self.dev_split, self._get_batch
                )
                self.metrics.log_progress(i, losses)

            # forward pass
            Xtr, Ytr = self._get_batch(self.train_split)
            Xtr, Ytr = Xtr.to(self.device), Ytr.to(self.device)
            _, loss = self.model(Xtr, Ytr)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # update params
            self.optimizer.step()

        # Final evaluation and saving
        self._finalize_training(start_time)

    def _finalize_training(self, start_time: datetime):
        """handle final evaluation, saving, and sampling"""
        # final evaluation
        final_losses = self.metrics.evaluate_loss(
            self.train_split, self.dev_split, self._get_batch
        )
        train_loss = list(final_losses.values())[0]
        dev_loss = list(final_losses.values())[1]
        print(f"Final losses: train_loss {train_loss:.5f}; eval_loss {dev_loss:.5f}")
        # calc train time
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        print(f"Training completed in {training_time:.2f} seconds")
        # save model if enabled in config
        if self.train_config.save_model:
            model_dir = self.model_manager.save_model(
                self.model,
                self.optimizer,
                train_loss,
                dev_loss,
                training_time,
                self.metrics.get_detailed_results(),
            )
        # sample from model after training if enabled in config
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
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
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


class ModelManager:
    """handles model init and saving ops"""

    def __init__(self, train_config: TrainConfig, model_config: GPTconfig):
        self.train_config = train_config
        self.model_config = model_config
        self.device = (
            train_config.device if torch.backends.mps.is_available() else "cpu"
        )

    def create_model(self) -> GPT:
        """init and return GPT model onto device"""
        model = GPT(self.model_config)
        return model.to(self.device)

    def save_model(
        self,
        model: GPT,
        optimizer,
        final_train_loss: float,
        final_dev_loss: float,
        training_time: float,
        detailed_results: list,
    ) -> str:
        """save model and return the model directory path"""
        # create unique model dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(
            self.train_config.model_save_dir,
            f"{self.train_config.model_name}_{timestamp}",
        )
        os.makedirs(model_dir, exist_ok=True)
        # save model checkpoint in model_dir
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_path,
        )
        # save combined config with training results
        config_path = os.path.join(model_dir, "config.json")
        config_data = {
            "train_config": asdict(self.train_config),
            "model_config": asdict(self.model_config),
            "training_results": {
                "final_train_loss": round(float(final_train_loss), 3),
                "final_dev_loss": round(float(final_dev_loss), 3),
                "training_time": f"{round(training_time / 60, 2)} min",
                "total_parameters": f"{(model.get_num_params()):,}",
                "device_used": str(self.device),
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "detailed_training_results": detailed_results,
            },
        }
        # save file & print stats
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"Model saved to: {model_dir}")
        print(f"  - Model checkpoint: {model_path}")
        print(f"  - Configuration: {config_path}")
        return model_dir


class TrainingMetrics:
    """handles loss tracking and evaluation during training"""

    def __init__(self, model, device: str, eval_iter: int):
        self.model = model
        self.device = device
        self.eval_iter = eval_iter
        self.detailed_results: List[str] = []

    @torch.no_grad()
    def evaluate_loss(
        self, train_split: torch.Tensor, dev_split: torch.Tensor, get_batch_fn
    ) -> Dict[torch.Tensor, float]:
        """evaluate loss on train and dev splits"""
        self.model.eval()
        losses = {}
        # calc train & dev loss as averages after defined eval steps
        for split in [train_split, dev_split]:
            split_losses = torch.zeros(self.eval_iter)
            # calc loss for every batch and save result into tensor
            for i in range(self.eval_iter):
                x, y = get_batch_fn(split)
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                split_losses[i] = loss.item()
            losses[split] = split_losses.mean()
        # change to train mode after eval
        self.model.train()
        return losses

    def log_progress(self, iteration: int, losses: Dict) -> None:
        """log training progress and store for detailed results"""
        train_loss = list(losses.values())[0]
        dev_loss = list(losses.values())[1]

        result_line = (
            f"loss after {iteration} iterations: train_loss {train_loss:.5f}; "
            f"eval_loss {dev_loss:.5f}"
        )
        self.detailed_results.append(result_line)
        print(result_line)

    def get_detailed_results(self) -> List[str]:
        """return all logged training progress"""
        return self.detailed_results


def main():
    """main entry point"""
    train_config = TrainConfig()
    model_config = GPTconfig()
    trainer = NameGPTTrainer(train_config, model_config)
    trainer.train_model()


if __name__ == "__main__":
    main()
