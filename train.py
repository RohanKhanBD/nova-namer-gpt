# validate loss function outsite backprop; called from training function after defined training steps
@torch.no_grad()
def check_loss():
    m.eval()
    out = {}
    # calc train & dev loss as averages after defined eval steps
    for split in [train_split, dev_split]:
        losses = torch.zeros(eval_iter)
        # calc loss for every batch and save result into tensor
        for i in range(eval_iter):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            _, loss = m(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean() 
    m.train()
    return out

# train model over defined train steps
def train_model():

    for i in range(train_iter):
    
        # eval loss & print after certain amount of train steps
        if i % eval_interval == 0:
            losses = check_loss()
            print(f"loss after {i} iterations: train_loss {losses[train_split]}; eval_loss {losses[dev_split]}")
        
        # forward pass
        Xtr, Ytr = get_batch(train_split)
        Xtr, Ytr = Xtr.to(device), Ytr.to(device)
        _, loss = m(Xtr, Ytr)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update params
        optimizer.step()

train_model()

# init model, port to gpu, init optimizer, print model params
model = GPT()
m = model.to(device)
optimizer = Optim.Adam(m.parameters(), lr=learning_rate)
parameters = m.parameters()
print(sum(p.nelement() for p in parameters))