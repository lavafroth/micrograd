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


block_size = 8


def build_dataset(words, debug=False) -> SimpleNamespace:
    x, y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + ".":
            ix = stoi[ch]
            x.append(context)
            y.append(ix)

            if debug:
                print(f"{context=}\tmaps to\t{ix} = {ch}")

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

    @staticmethod
    def norm_across_dim(x: torch.Tensor) -> tuple[int] | int:
        if x.ndim == 3:
            return (0, 1)
        if x.ndim == 2:
            return 0
        raise NotImplementedError("normalization across dimensions higher than 3 are not implemented")
        

    def __call__(self, x):
        if self.training:
            norm_across_dim = BatchNorm1d.norm_across_dim(x)
            xmean = x.mean(norm_across_dim, keepdim=True)
            xvar = x.var(norm_across_dim, keepdim=True, unbiased=True)
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

class FlattenGroupBy:
    def __init__(self, group_size: int):
        self.n = group_size

    def __call__(self, x: torch.Tensor):
        b, t, c = x.shape
        x = x.view(b, t//self.n, self.n * c)
        if x.size(1) == 1:
            x = x.squeeze(dim=1)
        self.out = x
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


n_embed = 24
# n_hidden = 200
n_hidden = 128

torch.manual_seed(42)

# set training hyperparameters
max_steps = 100_000
batch_size = 32

group_by = 2

model = Sequential([
    Embedding(vocab_size, n_embed),
    ##
    FlattenGroupBy(group_by),
    Linear(n_embed * group_by, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    ##
    FlattenGroupBy(group_by),
    Linear(n_hidden * group_by, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    ##
    FlattenGroupBy(group_by),
    Linear(n_hidden * group_by, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),
    ##
    Linear(n_hidden, vocab_size)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1


parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True


ix = torch.randint(high=dataset.train.x.size(0), size=(4, ))
x = dataset.train.x[ix]
y = dataset.train.y[ix]
print(x.shape)

x
logits = model(x)

for layer in model.layers:
    print('{:15s}'.format(layer.__class__.__name__), tuple(layer.out.shape))

(torch.randn(4, 80) @ torch.randn(80, 200) + torch.randn(200)).shape
(torch.randn(4, 10, 80) @ torch.randn(80, 200) + torch.randn(200)).shape

# introduce a batch dimension to multiply weight matrix per bi-gram
(torch.randn(4, 4, 20) @ torch.randn(20, 200) + torch.randn(200)).shape
example = torch.arange(4 * 8 * 10).view(4, 8, 10)
example


example_manual_concat = torch.cat([example[:, ::2, :], example[:, 1::2, :]], dim=2)
example_manual_concat.shape

example_auto_view = example.view(4, 4, 20)
example_auto_view.shape

example = torch.randn((32, 4, 68))
example_mean = example.mean(dim=0, keepdim=True)
example_std = example.var(dim=0, keepdim=True)**.5
example_normalized = (example - example_mean) / example_std
print(f"{example_std.shape = }")
print(f"{example_mean.shape = }")
print(f"{example_normalized.shape = }")

example_mean = example.mean(dim=(0, 1), keepdim=True)
example_std = example.var(dim=(0, 1), keepdim=True)**.5
example_normalized = (example - example_mean) / example_std
print(f"{example_std.shape = }")
print(f"{example_mean.shape = }")
print(f"{example_normalized.shape = }")

if "Training Block":
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
        # break


smooth_log_losses = torch.tensor(log_losses).view(-1, 1000).mean(dim=1)
plt.close("all")
plt.plot(smooth_log_losses)
plt.show()


model.layers[3].running_mean.shape




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


for x, y in zip(dataset.train.x[:8], dataset.train.y[:8]):
    print(''.join(itos[ix.item()] for ix in x), '\t', itos[y.item()])

dataset.train.x[[0]].shape
logits = model(dataset.train.x[[0]])
logits
logits.shape

logits = torch.zeros((8, 27))
for i in range(8):
    logits[i] = model(dataset.train.x[[i]])

logits

