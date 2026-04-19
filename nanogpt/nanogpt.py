import torch
from torch import nn
from torch.nn import functional as F

with open("nanogpt/chat-cleaned.md") as handle:
    text = handle.read()

print("dataset length in characters: ", len(text))

print(text[:1000])

chars = sorted(set(text))

if (chars[0], chars[1]) == ("\t", "\n"):
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
train_data[: block_size + 1]
decoder(train_data[: block_size + 1].tolist())

x = train_data[:block_size]
y = train_data[1 : block_size + 1]

for t in range(block_size):
    context = x[: t + 1]
    target = y[t]

    print(f"when input is {context} the target is: {target}")

torch.manual_seed(5)

batch_size = 4
block_size = 8


def get_batch(split):
    data = train_data if split == "train" else valid_data
    ix = torch.randint(high=len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print("inputs:", tuple(xb.shape))
print(xb)
print("targets:", tuple(yb.shape))
print(yb)
print("-" * 24)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
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
            logits, loss = self(indices)  # (batch, time, channel)
            logits = logits[:, -1, :]  # (batch, channel)
            probs = F.softmax(logits, dim=-1)  # softmax all channel values
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


# # The mathematical trick in self-attention
#
# with a toy example

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.arange(B * T * C).float().view(B, T, C)
x.shape

xbow = torch.zeros_like(x)
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # start through this timestamp
        xbow[b, t] = xprev.mean(dim=0)

x[0]

xbow[0]


torch.manual_seed(42)
a = torch.ones((3, 3))
a = torch.tril(a)
a /= a.sum(dim=1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print("a =")
print(a)
print("b =")
print(b)
print("c =")
print(c)

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.arange(B * T * C).float().view(B, T, C)
x.shape

# wei = torch.tril(torch.ones(T, T))
# wei /= wei.sum(dim=1, keepdim=True)

wei = F.softmax(torch.tril(torch.ones(T, T)).log(), dim=1)

wei.shape

xbow = wei @ x  # {B, T, T} @ (B, T, C)

# wei (T, T) broadcasts to (B, T, T)
# x[batch] @ wei[batch] for batch in range(B)
# (T, T)   @ (T, C)


torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)  # (B, T, head_size), same shape for q and v
q = query(x)
v = value(x)
wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T)
# q[batch]       @ k[batch] for batch in range(B)
# (T, head_size) @ (head_size, T)

wei = wei + torch.tril(torch.ones((T, T))).log()
wei = F.softmax(wei, dim=-1)

out = wei @ v
out.shape
out[0]

if "attention head: my experiment":
    torch.manual_seed(1337)
    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)

    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    # (B, T, head_size) for q, k, v
    k = key(x)
    q = query(x)
    v = value(x)
    out = (
        F.softmax(
            q @ k.transpose(-2, -1) # scale
            + torch.tril(torch.ones((T, T))).log(), dim=-1
        )
        @ v
    )
    print(out.shape)
    print(out[0])


k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)

wei = q @ k.transpose(-2, -1) * head_size ** .5
k.var()
q.var()
wei.var() # of the order of the head_size


torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)

torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]) * 10, dim=-1)

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

torch.manual_seed(1337)
module = BatchNorm1d(100)
x = torch.randn(32, 100)
x = module(x)
x.shape

first_neuron = x[:, 0]
print(first_neuron.mean())
print(first_neuron.std())

first_row = x[0]

print(first_row.mean())
print(first_row.std())

class LayerNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True, unbiased=True)

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100)
x = module(x)
x.shape

first_neuron = x[:, 0]
print(first_neuron.mean())
print(first_neuron.std())

first_row = x[0]

print(first_row.mean())
print(first_row.std())

