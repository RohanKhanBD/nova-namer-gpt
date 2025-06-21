# import text & shuffle set
with open("./assets/names.txt", mode="r", encoding="utf-8") as file:
    names = file.readlines()
# shuffle it
random.seed(42)
random.shuffle(names)
# dataset size
print(len(names))
print(names[:10])

# setup vocabulary
all_chars = list(sorted(set([("".join(char)) for name in names for char in name])))
print(all_chars)
vocab_size = len(all_chars)
print(vocab_size)

# vocabulary mapping dicts
itos = {i:s for i, s in enumerate(all_chars)}
stoi = {s:i for i, s in itos.items()}
#print(itos)
#print(stoi)
# voc encoding / decoding functions
encode = lambda input: [stoi[i] for i in input]
decode = lambda input: "".join([itos[i] for i in input])
#print(encode(names[0]))
#print(decode(encode(names[0])))

# convert names list to data: concat text, encode it, tensor it
data = torch.tensor(encode("".join(names)), dtype=torch.long)
# split data into train / dev / test with 0.8 / 0.1 / 0.1
border_1 = int(0.8 * len(data))
border_2 = int(0.9 * len(data))
train_split = data[:border_1]
dev_split = data[border_1:border_2]
test_split = data[border_2:]
print(len(train_split), len(dev_split), len(test_split))

# data loading: deliver batches of X, Y tensors for chosen split
torch.manual_seed(42)
def get_batch(split):
    """ delivers a batch of X, Y tensors for specified split"""
    # get random numbers (in amount of "batch_size") within split boundaries to grab data for the batch samples
    batch_borders = torch.randint(0, len(split)-context_len, (batch_size,))
    x = torch.stack([split[t : t+context_len] for t in batch_borders])
    y = torch.stack([split[t+1 : t+context_len+1] for t in batch_borders])
    return x, y
x, y = get_batch(train_split)
print(x.shape, y.shape)