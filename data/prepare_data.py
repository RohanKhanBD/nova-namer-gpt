import os


"""
- read in dataset of 60k bavarian city names and do some processing
- create encoding / decoding functions
- save train / dev / test splits as bin files
"""

class NameProcessor:
    """ processes bavarian names dataset for GPT training """

    def __init__(self):
        self.raw_data = self.read_data()
    
    def read_data(self):
        with open("./assets/names.txt", mode="r", encoding="utf-8") as file:
        names = file.readlines()

   





# import text & shuffle set

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

