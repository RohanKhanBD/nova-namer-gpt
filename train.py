import os
from datetime import datetime
import torch
import torch.optim as Optim
import numpy as np
import pickle
import json
from dataclasses import asdict
from config import TrainConfig, SampleConfig
from model import GPTconfig, GPT
from sample import NameGPTSampler
from typing import Tuple, Dict


""" Bavarian City Name GPT // lightweight training scrip """


class NameGPTTrainer:

    def __init__(self, train_config: TrainConfig, model_config: GPTconfig):
        assert isinstance(train_config, TrainConfig), "Invalid train config type."
        assert isinstance(model_config, GPTconfig), "Invalid model config type."
        self.train_config = train_config
        self.model_config = model_config
        self.device = self._get_device()
        # load data
        self.train_data, self.dev_data = self._load_data()
        # init model
        self.model = GPT(self.model_config).to(self.device)
        self.optimizer = Optim.Adam(self.model.parameters(), lr=self.train_config.learning_rate)
        # will be updated after training
        self.training_results = []
        self.final_losses = {}
        self.model_save_dir = None

        # print amount params on after model init
        print(f"Successful model init with {self.model.get_num_params():,} parameters.")

    def _get_device(self) -> str:
        """ get best available device based on train config and hardware """
        if self.train_config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.train_config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """load train and val data"""
        # load bins, do memmap, return as tensors
        data_dir = self.train_config.data_dir
        splits = {}
        for split in ["train", "dev"]:
            # bin_path = os.path.join(data_dir, f"{split}.bin")
            assert os.path.exists(os.path.join(data_dir, f"{split}.bin")), ".bin file not found."
            data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
            splits[split] = torch.from_numpy(data.astype(np.int64))
        # load metadata & compare vocab of gptconfig & dataset
        assert os.path.exists(os.path.join(data_dir, "meta.pkl")), "meta.pkl file not found"
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        actual_vocab = meta["vocab_size"]
        # warn if necessary and proceed anyway with updating model config vocab size
        if self.model_config.vocab_size and self.model_config.vocab_size != actual_vocab:
            print(f"Warning: Model config vocab_size ({self.model_config.vocab_size}) "
                  f"doesn't match data vocab_size ({actual_vocab}). Using data vocab_size.")
        self.model_config.vocab_size = actual_vocab
        print(f"Loaded data: train={len(splits["train"]):,} tok, dev={len(splits["dev"]):,} tok")
        print(f"Vocabulary size: {meta['vocab_size']}")
        return splits["train"], splits["dev"]

    def _get_batch(self, split: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """delivers a batch of X, Y tensors for specified split"""
        batch_borders = torch.randint(
            0,
            len(split) - self.model_config.context_len,
            (self.train_config.batch_size,),
        )
        x = torch.stack([split[t: t + self.model_config.context_len] for t in batch_borders])
        y = torch.stack([split[t + 1: t + self.model_config.context_len + 1] for t in batch_borders])
        return x, y

    def train_model(self):
        """train model over defined train steps"""
        start_time = datetime.now()
        # seed for torch from train_config
        torch.manual_seed(self.train_config.seed)
        # training loop
        print("Training started...")
        for i in range(self.train_config.train_iter):

            # eval loss & print after certain amount of train steps
            if i % self.train_config.eval_interval == 0:
                losses = self._estimate_loss()
                result = f"loss after {i} iterations: train_loss {losses["train"]:.5f}; eval_loss {losses["dev"]:.5f}"
                self.training_results.append(result)
                print(result)

            # forward pass
            Xtr, Ytr = self._get_batch(self.train_data)
            Xtr, Ytr = Xtr.to(self.device), Ytr.to(self.device)
            _, loss = self.model(Xtr, Ytr)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # update params
            self.optimizer.step()

        # Final evaluation and saving
        self._finalize_training(start_time)

    @torch.no_grad()
    def _estimate_loss(self) -> Dict[str, float]:
        """ evaluate loss on train and dev splits """
        self.model.eval()
        losses = {}
        for split_name, split_data in [("train", self.train_data), ("dev", self.dev_data)]:
            split_losses = torch.zeros(self.train_config.eval_iter)
            for i in range(self.train_config.eval_iter):
                x, y = self._get_batch(split_data)
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                split_losses[i] = loss.item()
            losses[split_name] = split_losses.mean().item()
        self.model.train()
        return losses

    def _finalize_training(self, start_time: datetime) -> None:
        """handle final evaluation, saving, and sampling"""
        # save final losses into obj, then unpack & print
        self.final_losses = self._estimate_loss()
        train_loss, dev_loss = self.final_losses["train"], self.final_losses["dev"]
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"Final losses: train_loss {train_loss:.5f}; eval_loss {dev_loss:.5f}")
        print(f"Training completed in {training_time:.2f} seconds")
        # save model after training & return path
        self.model_save_dir = self._save_checkpoint(train_loss, dev_loss, training_time)
        # print samples after training
        self._sample_after_train()

    def _save_checkpoint(self, train_loss: float, dev_loss: float, training_time: float) -> str:
        """ save model state dicts, config data and training results"""
        save_dir = self.train_config.save_dir_current
        os.makedirs(save_dir, exist_ok=True)
        # inference only saving model -> only state_dict
        torch.save(self.model.state_dict(), os.path.join(save_dir, self.train_config.model_filename))
        # save config separately from torch.save
        config_data = {
            "train_config": asdict(self.train_config),
            "model_config": asdict(self.model_config),
            "training_results": {
                "final_train_loss": round(float(train_loss), 3),
                "final_dev_loss": round(float(dev_loss), 3),
                "training_time": f"{round(training_time / 60, 2)} min",
                "total_parameters": f"{self.model.get_num_params():,}",
                "device_used": str(self.device),
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "detailed_training_results": self.training_results,
            },
        }
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"Model saved to: {save_dir}")
        return save_dir

    def _sample_after_train(self) -> None:
        """always print some samples after training"""
        # create sampler with current model using same device as training
        sampler = NameGPTSampler.from_training(
            sample_config=SampleConfig(
                device=self.train_config.device,
                num_samples=self.train_config.num_samples
            ),
            model_dir=self.model_save_dir,
            model=self.model,
        )
        sampler.generate()


def main():
    """main entry point"""
    trainer = NameGPTTrainer(TrainConfig(), GPTconfig())
    trainer.train_model()


if __name__ == "__main__":
    main()
