import os
import torch
import json
from datetime import datetime
from dataclasses import asdict
from model import GPTconfig, GPT
from config import TrainConfig
from typing import Dict, List, Tuple


""" Bavarian City Name GPT // handles model initialization and saving operations """

class ModelManager:
    """  """
    
    def __init__(self, train_config: TrainConfig, model_config: GPTconfig):
        self.train_config = train_config
        self.model_config = model_config
        self.device = train_config.device if torch.backends.mps.is_available() else "cpu"
    
    def create_model(self) -> GPT:
        """Initialize and return GPT model"""
        model = GPT(self.model_config)
        return model.to(self.device)
    
    def save_model(self, model: GPT, optimizer, final_train_loss: float, 
                   final_dev_loss: float, training_time: float, detailed_results: list) -> str:
        """Save model and return the model directory path"""
        # Create unique model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(
            self.train_config.model_save_dir, 
            f"{self.train_config.model_name}_{timestamp}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model checkpoint
        model_path = os.path.join(model_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        
        # Save combined config with training results
        config_path = os.path.join(model_dir, "config.json")
        config_data = {
            "train_config": asdict(self.train_config),
            "model_config": asdict(self.model_config),
            "training_results": {
                "final_train_loss": round(float(final_train_loss), 3),
                "final_dev_loss": round(float(final_dev_loss), 3),
                "training_time": f"{round(training_time / 60, 2)} min",
                "total_parameters": f"{sum(p.nelement() for p in model.parameters()):,}",
                "device_used": str(self.device),
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "detailed_training_results": detailed_results
            }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"Model saved to: {model_dir}")
        print(f"  - Model checkpoint: {model_path}")
        print(f"  - Configuration: {config_path}")
        return model_dir


class TrainingMetrics:
    """Handles loss tracking and evaluation during training"""
    
    def __init__(self, model, device: str, eval_iter: int):
        self.model = model
        self.device = device
        self.eval_iter = eval_iter
        self.detailed_results: List[str] = []
    
    @torch.no_grad()
    def evaluate_loss(self, train_split: torch.Tensor, dev_split: torch.Tensor, 
                     get_batch_fn) -> Dict[torch.Tensor, float]:
        """Evaluate loss on train and dev splits"""
        self.model.eval()
        losses = {}
        
        for split in [train_split, dev_split]:
            split_losses = torch.zeros(self.eval_iter)
            for i in range(self.eval_iter):
                x, y = get_batch_fn(split)
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                split_losses[i] = loss.item()
            losses[split] = split_losses.mean()
        
        self.model.train()
        return losses
    
    def log_progress(self, iteration: int, losses: Dict) -> None:
        """Log training progress and store for detailed results"""
        train_loss = list(losses.values())[0]  # First is train
        dev_loss = list(losses.values())[1]    # Second is dev
        
        result_line = f"loss after {iteration} iterations: train_loss {train_loss:.5f}; eval_loss {dev_loss:.5f}"
        self.detailed_results.append(result_line)
        print(result_line)
    
    def get_detailed_results(self) -> List[str]:
        """Return all logged training progress"""
        return self.detailed_results