
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from config import TrainConfig
from model import GPTconfig, GPT

""" Bavarian City Name GPT // lightweight training scrip """

class NameGPTTrainer:

    def __init__(self, train_config: TrainConfig, model_config: GPTconfig):
        self.train_config = train_config
        self.model_config = model_config
        # set device mps; if not available cpu
        self.device = train_config.device if torch.backends.mps.is_available() else "cpu"
        # load data
        self.train_split, self.dev_split = self._load_data()
    
    def _load_data(self):
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


def main():
    """ main entry point"""
    train_config = TrainConfig()
    model_config = GPTconfig()
    test_run = NameGPTTrainer(train_config, model_config)
    


if __name__ == "__main__":
    main()




# device = "mps" if torch.backends.mps.is_available() else "cpu"

# # data loading: deliver batches of X, Y tensors for chosen split
# torch.manual_seed(42)
# def get_batch(split):
#     """ delivers a batch of X, Y tensors for specified split"""
#     # get random numbers (in amount of "batch_size") within split boundaries to grab data for the batch samples
#     batch_borders = torch.randint(0, len(split)-context_len, (batch_size,))
#     x = torch.stack([split[t : t+context_len] for t in batch_borders])
#     y = torch.stack([split[t+1 : t+context_len+1] for t in batch_borders])
#     return x, y
# x, y = get_batch(train_split)
# print(x.shape, y.shape)


# # validate loss function outsite backprop; called from training function after defined training steps
# @torch.no_grad()
# def check_loss():
#     m.eval()
#     out = {}
#     # calc train & dev loss as averages after defined eval steps
#     for split in [train_split, dev_split]:
#         losses = torch.zeros(eval_iter)
#         # calc loss for every batch and save result into tensor
#         for i in range(eval_iter):
#             x, y = get_batch(split)
#             x, y = x.to(device), y.to(device)
#             _, loss = m(x, y)
#             losses[i] = loss.item()
#         out[split] = losses.mean() 
#     m.train()
#     return out

# # train model over defined train steps
# def train_model():

#     for i in range(train_iter):
    
#         # eval loss & print after certain amount of train steps
#         if i % eval_interval == 0:
#             losses = check_loss()
#             print(f"loss after {i} iterations: train_loss {losses[train_split]}; eval_loss {losses[dev_split]}")
        
#         # forward pass
#         Xtr, Ytr = get_batch(train_split)
#         Xtr, Ytr = Xtr.to(device), Ytr.to(device)
#         _, loss = m(Xtr, Ytr)

#         # backward pass
#         optimizer.zero_grad()
#         loss.backward()

#         # update params
#         optimizer.step()

# train_model()

# # init model, port to gpu, init optimizer, print model params
# model = GPT()
# m = model.to(device)
# optimizer = Optim.Adam(m.parameters(), lr=learning_rate)
# parameters = m.parameters()
# print(sum(p.nelement() for p in parameters))