import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from types import SimpleNamespace
from dataclasses import dataclass

# %matplotlib kitcat

# read in words
with open("./makemore/names-brazil-top100000.txt") as handle:
    words = handle.read().splitlines()

words[:8]
len(words)


# gather characters in our vocabulary
chars = set()
for word in words:
    chars |= set(word)
chars = sorted(chars)
chars[:5]


# map characters to indices and vice-versa
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)
vocab_size


block_size = 3


def build_dataset(words, debug=False) -> SimpleNamespace:
    x, y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + ".":
            ix = stoi[ch]
            x.append(context)
            y.append(ix)

            if debug:
                print(f"{context=} maps to {ix}={ch}")

            context = context[1:] + [ix]

    x = torch.tensor(x)
    y = torch.tensor(y)
    if debug:
        print(x.shape, y.shape)
    return SimpleNamespace(x=x, y=y)


build_dataset(words[:2], debug=True)


n = len(words)
n1 = int(0.8 * n)
n2 = int(0.9 * n)


random.seed(5)
random.shuffle(words)
dataset = SimpleNamespace(
    train=build_dataset(words[:n1]),
    valid=build_dataset(words[n1:n2]),
    test=build_dataset(words[n2:]),
)


class Linear:
    def __init__(self, fan_in: int, fan_out: int, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros((fan_out,)) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]


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


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, n_embeddings, embedding_dim):
        self.weight = torch.randn((n_embeddings, embedding_dim))

    def __call__(self, ix):
        self.out = self.weight[ix]
        return self.out

    def parameters(self):
        return [self.weight]

class Flatten:
    def __call__(self, x):
        self.out = x.flatten(start_dim=1)
        return self.out

    def parameters(self):
        return []

@dataclass
class Sequential:
    layers: list

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


n_embed = 10
n_hidden = 200

# gen = torch.Generator().manual_seed(5)
torch.manual_seed(42)

# set training hyperparameters
max_steps = 100_000
batch_size = 32


C = torch.randn((vocab_size, n_embed))
layers = [
    Linear(n_embed * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size)
]

with torch.no_grad():
    # less confident last layer
    layers[-1].weight *= 0.1


parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True



log_losses = []
for run in range(max_steps):
    ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,))
    x = dataset.train.x[ix]
    y = dataset.train.y[ix]

    emb = C[x]
    emb = emb.view(emb.shape[0], -1)

    x = emb
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)
    loss_scalar = loss.item()

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.001 if run > max_steps // 2 else 0.03

    for p in parameters:
        p.data -= p.grad * lr

    if run % 10_000 == 0:
        print(f"{run:07d}/{max_steps:07d}: {loss_scalar:0.04f}")

    log_losses.append(loss.log10().item())




plt.close("all")
plt.plot(log_losses)
plt.show()


torch.arange(10).view(2, -1)


smooth_log_losses = torch.tensor(log_losses).view(-1, 1000).mean(dim=1)
plt.close("all")
plt.plot(smooth_log_losses)
plt.show()



for layer in layers:
    layer.training = False



def split_loss(split):
    x, y = split.x, split.y
    emb = C[x]
    emb = emb.view(emb.shape[0], -1)
    for layer in layers:
        emb = layer(emb)
    loss = F.cross_entropy(emb, y)
    return loss.item()


print("train", split_loss(dataset.train))
print("valid", split_loss(dataset.valid))

## Using consolidated `Embedding` and `Flatten`

layers = [
    Embedding(vocab_size, n_embed),
    Flatten(),
    Linear(n_embed * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size)
]

with torch.no_grad():
    # less confident last layer
    layers[-1].weight *= 0.1


parameters = [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True



log_losses = []
for run in range(max_steps):
    ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,))
    x = dataset.train.x[ix]
    y = dataset.train.y[ix]

    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)
    loss_scalar = loss.item()

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.001 if run > max_steps // 2 else 0.03

    for p in parameters:
        p.data -= p.grad * lr

    if run % 10_000 == 0:
        print(f"{run:07d}/{max_steps:07d}: {loss_scalar:0.04f}")

    log_losses.append(loss.log10().item())
    break

## Using consolidated `Sequential`

model = Sequential([
    Embedding(vocab_size, n_embed),
    Flatten(),
    Linear(n_embed * block_size, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1


parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True



log_losses = []
for run in range(max_steps):
    ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,))
    x = dataset.train.x[ix]
    y = dataset.train.y[ix]

    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss_scalar = loss.item()

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.001 if run > max_steps // 2 else 0.03

    for p in parameters:
        p.data -= p.grad * lr

    if run % 10_000 == 0:
        print(f"{run:07d}/{max_steps:07d}: {loss_scalar:0.04f}")

    log_losses.append(loss.log10().item())
    break




for layer in model.layers:
    layer.training = False

def split_loss(split):
    x, y = split.x, split.y
    loss = F.cross_entropy(model(x), y)
    return loss.item()


print(f"{split_loss(dataset.train) = }")
print(f"{split_loss(dataset.valid) = }")

for _ in range(20):
    out = []
    context = [0] * block_size # all START tokens
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))


torch.tensor([context]).shape
model.layers[0].weight.shape
model.layers[0](torch.tensor([context])).shape
