import torch
from torch import nn
from torch.nn import functional as F
with open("nanogpt/chat-cleaned.md") as handle:
    text = handle.read()

print("dataset length in characters: ", len(text))

print(text[:1000])

chars = sorted(set(text))

if (chars[0], chars[1]) == ('\t', '\n'):
    chars[0], chars[1] = chars[1], chars[0]

vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = dict(enumerate(chars))

def encoder(s):
    return [stoi[c] for c in s]


def decoder(s):
    return "".join(itos[c] for c in s)


print(encoder("please please hurry"))
print(decoder(encoder("please please hurry")))

data = torch.tensor(encoder(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = data.size(0) * 9 // 10
train_data = data[:n]
valid_data = data[n:]

block_size = 8
train_data[:block_size + 1]
decoder(train_data[:block_size + 1].tolist())

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]

    print(f"when input is {context} the target is: {target}")

torch.manual_seed(5)

batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else valid_data
    ix = torch.randint(high=len(data) - block_size, size=(batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print("inputs:", tuple(xb.shape))
print(xb)
print("targets:", tuple(yb.shape))
print(yb)
print('-' * 24)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print("when input is {} the target is: {}".format(context.tolist(), target))


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(indices)
        #      (batch, seq_length, embedding_dim)
        # also (batch, time, channel)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        # logits = logits.view(B*T, C)
        # targets = targets.view(B*T)
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, indices: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, loss = self(indices) # (batch, time, channel)
            logits = logits[:, -1, :] # (batch, channel)
            probs = F.softmax(logits, dim=-1) # softmax all channel values
            next_ix = torch.multinomial(probs, num_samples=1)
            assert next_ix.shape == (indices.size(0), 1)
            indices = torch.cat((indices, next_ix), dim=1)
        return indices
        

m = BigramLanguageModel(vocab_size)
out, loss = m(xb, yb)
print(out.shape)
print(yb.shape)
print(loss)


idx = torch.zeros((1, 1), dtype=torch.long)
decoder(m.generate(idx, 300)[0].tolist())

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for step in range(10000):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(loss.item())

print(decoder(m.generate(idx, 300)[0].tolist()))
# torch.log(torch.tensor([logits.size(1)]))
