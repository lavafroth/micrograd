import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def build_dataset(words, debug=False):
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
    return x, y


build_dataset(words[:2], debug=True)


import random

random.seed(5)


n = len(words)
n1 = int(0.8 * n)
n2 = int(0.9 * n)


from types import SimpleNamespace as ns

dataset = ns(train=ns(), valid=ns(), test=ns())


random.shuffle(words)


dataset.train.x, dataset.train.y = build_dataset(words[:n1])
dataset.valid.x, dataset.valid.y = build_dataset(words[n1:n2])
dataset.test.x, dataset.test.y = build_dataset(words[n2:])


n_embed = 10
n_hidden = 200

gen = torch.Generator().manual_seed(5)

# cell-choice-0: initialize model-0
g = dict(generator=gen, requires_grad=True)
C = torch.randn((vocab_size, n_embed), **g)
w1 = torch.randn((n_embed * block_size, n_hidden), **g)
b1 = torch.randn((n_hidden,), **g)
w2 = torch.randn((n_hidden, vocab_size), **g)
b2 = torch.randn((vocab_size,), **g)
parameters = [C, w1, b1, w2, b2]
sum(p.nelement() for p in parameters)


#  cell-choice-0: initialize model-1
C = torch.randn((vocab_size, n_embed), generator=gen)
w1 = torch.randn((n_embed * block_size, n_hidden), generator=gen) * 0.2
b1 = torch.randn((n_hidden,), generator=gen) * 0.01
w2 = torch.randn((n_hidden, vocab_size), generator=gen) * 0.01
b2 = torch.zeros((vocab_size,))
parameters = [C, w1, b1, w2, b2]
for p in parameters:
    p.requires_grad = True
sum(p.nelement() for p in parameters)


# set training hyperparameters
max_steps = 100_000
batch_size = 32


# double check
ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,), generator=gen)
x = dataset.train.x[ix]
C[x].shape


# toggle to inspect logits
inspect_logits = True


# training loop
log_losses = []
for run in range(max_steps):
    ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,), generator=gen)
    x = dataset.train.x[ix]
    y = dataset.train.y[ix]

    emb = C[x]
    emb = emb.view(emb.shape[0], -1)

    h_pre_activation = emb @ w1 + b1
    h_pre_activation -= h_pre_activation.mean(dim=0, keepdim=True)
    h_pre_activation /= h_pre_activation.std(dim=0, keepdim=True)
    h = torch.tanh(h_pre_activation)

    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)
    loss_scalar = loss.item()

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.001 if run > max_steps // 2 else 0.01

    for p in parameters:
        p.data -= p.grad * lr

    if run % 10_000 == 0:
        print(f"{run:07d}/{max_steps:07d}: {loss_scalar:0.04f}")

    log_losses.append(loss.log10().item())

    if inspect_logits:
        break


if inspect_logits:
    logits[0]
    h.shape
    h.view(-1).shape
    plt.close("all")
    plt.hist(h.view(-1).tolist(), 50)
    plt.show()
    plt.close("all")
    plt.hist(h_pre_activation.view(-1).tolist(), 50)
    plt.show()

    plt.close("all")
    plt.imshow(h.abs() > 0.99, cmap="Blues", interpolation="nearest")
    plt.show()

    h_pre_activation.shape
    h_pre_activation.mean(dim=0, keepdim=True).shape
    h_pre_activation.std(dim=0, keepdim=True).shape
else:
    # plot losses
    plt.close("all")
    plt.plot(torch.arange(0, max_steps, 1), log_losses)
    plt.show()


@torch.no_grad
def split_loss(split):
    x, y = split.x, split.y
    emb = C[x]
    emb = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb @ w1 + b1)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)
    return loss.item()


print("train", split_loss(dataset.train))
print("valid", split_loss(dataset.valid))


gen = torch.Generator().manual_seed(12)
for _ in range(10):
    word = ""
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])].view((1, -1))
        h = torch.tanh(emb @ w1 + b1)
        logits = h @ w2 + b2
        probs = torch.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=gen).item()
        if ix == 0:
            break
        context = context[1:] + [ix]
        word += itos[ix]
    print(word)


# dissecting the giant initial loss
initial_prob_of_any_char = 1 / vocab_size
-torch.tensor([initial_prob_of_any_char]).log()


# 4-d example of the issue
def four_d_loss(logits, true_class=2):
    probs = torch.softmax(logits, dim=0)
    return -probs[true_class].log()


four_d_loss(torch.tensor([0.0, 0.0, 0.0, 0.0]))  # uniform (what we want)
four_d_loss(torch.tensor([0.0, 0.0, 5.0, 0.0]))  # if we are lucky
four_d_loss(torch.randn(4))  # realistic
four_d_loss(torch.randn(4) * 8)  # very realistic
four_d_loss(torch.randn(4) * 100)  # insane


# addressing the weight initialization automatically
fan_in = block_size * n_embed
w_scale = fan_in**-0.5  # try 5, 1, 0.2

x = torch.randn(1000, fan_in)
w = torch.randn(fan_in, 200) * w_scale
y = x @ w
print(x.mean().item(), x.std().item())
print(y.mean().item(), y.std().item())

plt.close("all")
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(x.view(-1), 50, density=True)
axs[1].hist(y.view(-1), 50, density=True)
plt.show()

# copying kaiming initialization
tanh_gain = 5 / 3
w_scale = tanh_gain / (fan_in**0.5)

C = torch.randn((vocab_size, n_embed), generator=gen)
w1 = torch.randn((n_embed * block_size, n_hidden), generator=gen) * w_scale
# b1 = torch.randn((n_hidden,), generator=gen) * 0.01
w2 = torch.randn((n_hidden, vocab_size), generator=gen) * 0.01
b2 = torch.zeros((vocab_size,))

bnorm_gain = torch.ones((1, n_hidden))
bnorm_bias = torch.zeros((1, n_hidden))
parameters = [
    C,
    w1,
    # b1,
    w2,
    b2,
    bnorm_bias,
    bnorm_gain,
]
for p in parameters:
    p.requires_grad = True
sum(p.nelement() for p in parameters)

# training loop
log_losses = []
running_batch_norm_std = torch.ones((1, n_hidden))
running_batch_norm_mean = torch.zeros((1, n_hidden))
batch_norm_momentum = 0.001

max_steps = 100_000

for run in range(max_steps):
    ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,), generator=gen)
    x = dataset.train.x[ix]
    y = dataset.train.y[ix]

    emb = C[x]
    emb = emb.view(emb.shape[0], -1)

    h_pre_activation = emb @ w1  # + b1
    current_batch_norm_mean = h_pre_activation.mean(dim=0, keepdim=True)
    current_batch_norm_std = h_pre_activation.std(dim=0, keepdim=True)
    h_normalized = (h_pre_activation - current_batch_norm_std) / current_batch_norm_std
    h_shifted = h_normalized * bnorm_gain + bnorm_bias

    with torch.no_grad():
        running_batch_norm_mean = (
            running_batch_norm_mean * (1 - batch_norm_momentum)
            + current_batch_norm_mean * batch_norm_momentum
        )
        running_batch_norm_std = (
            running_batch_norm_std * (1 - batch_norm_momentum)
            + current_batch_norm_std * batch_norm_momentum
        )

    h = torch.tanh(h_shifted)

    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)
    loss_scalar = loss.item()

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.001 if run > max_steps // 2 else 0.01

    for p in parameters:
        p.data -= p.grad * lr

    if run % 10_000 == 0:
        print(f"{run:07d}/{max_steps:07d}: {loss_scalar:0.04f}")

    log_losses.append(loss.log10().item())


with torch.no_grad():
    emb = C[dataset.train.x]
    embcat = emb.view(emb.shape[0], -1)

    h_pre_activation = embcat @ w1  # + b1
    batch_norm_mean = h_pre_activation.mean(dim=0, keepdim=True)
    batch_norm_std = h_pre_activation.std(dim=0, keepdim=True)

(running_batch_norm_mean - batch_norm_mean).abs().mean()
(running_batch_norm_std - batch_norm_std).abs().mean()


@torch.no_grad()
def split_loss(split):
    x, y = split.x, split.y
    emb = C[x]
    emb = emb.view(emb.shape[0], -1)
    h_pre_activation = emb @ w1  # + b1
    h_normalized = (h_pre_activation - running_batch_norm_mean) / running_batch_norm_std
    h_shifted = h_normalized * bnorm_gain + bnorm_bias

    h = torch.tanh(h_shifted)

    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)
    return loss.item()


print("train", split_loss(dataset.train))
print("valid", split_loss(dataset.valid))


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=gen) / fan_in**0.5
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


n_embed = 10
n_hidden = 100
gen = torch.Generator().manual_seed(5)
C = torch.randn((vocab_size, n_embed), generator=gen)
layers = [
    Linear(n_embed * block_size, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
    BatchNorm1d(vocab_size),
]


#! fight the squashing

# linear_boost = 0.5
# linear_boost = 3
linear_boost = 5 / 3
with torch.no_grad():
    layers[-1].gamma *= 0.1
    # layers[-1].weight *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= linear_boost

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True


max_steps = 100_000
batch_size = 32
log_losses = []
update_to_data = []

# C.shape

for run in range(max_steps):
    ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,), generator=gen)
    x = dataset.train.x[ix]
    y = dataset.train.y[ix]

    emb = C[x]
    emb = emb.view(emb.shape[0], -1)

    x = emb
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)
    loss_scalar = loss.item()

    for layer in layers:
        layer.out.retain_grad()  # why am I doing this?

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.001 if run > max_steps // 2 else 0.03

    for p in parameters:
        p.data -= p.grad * lr

    if run % 10_000 == 0:
        print(f"{run:07d}/{max_steps:07d}: {loss_scalar:0.04f}")

    log_losses.append(loss.log10().item())
    with torch.no_grad():
        update_to_data.append(
            [(lr * p.grad.std() / p.data.std()).log10().item() for p in parameters]
        )

    if run > 1_000:
        break  # take out # take out

plt.close("all")
plt.figure(figsize=(10, 2))
legend = []
for i, layer in enumerate(layers[:-1]):
    if not isinstance(layer, Tanh):
        continue
    t = layer.out
    print(
        "layer {} {:10s}: mean {:0.02f} std {:0.02f} saturated {:0.02f}%".format(
            i,
            layer.__class__.__name__,
            t.mean(),
            t.std(),
            (t.abs() > 0.97).float().mean() * 100,
        )
    )
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legend.append(f"layer {i} ({layer.__class__.__name__})")
plt.legend(legend)
plt.title("activation distributions")
plt.show()


plt.close("all")
plt.figure(figsize=(10, 2))
legend = []
for i, layer in enumerate(layers[:-1]):
    if not isinstance(layer, Tanh):
        continue
    t = layer.out.grad
    print(
        "layer {} {:10s}: mean {:0.02f} std {:0.02f} saturated {:0.02f}%".format(
            i,
            layer.__class__.__name__,
            t.mean(),
            t.std(),
            (t.abs() > 0.97).float().mean() * 100,
        )
    )
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legend.append(f"layer {i} ({layer.__class__.__name__})")
plt.legend(legend)
plt.title("gradient distributions")
plt.show()


plt.close("all")
plt.figure(figsize=(10, 2))
legend = []
for i, p in enumerate(parameters):
    if p.ndim != 2:
        continue
    t = p.grad
    print(
        "weight {} mean {:0.02f} std {} grad:data ratio {}".format(
            tuple(p.shape), t.mean(), t.std(), t.std() / p.std()
        )
    )
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legend.append(f"layer {i} ({tuple(p.shape)})")
plt.legend(legend)
plt.title("weight gradient distributions")
plt.show()

plt.close("all")
plt.figure(figsize=(10, 2))
legend = []
for i, p in enumerate(parameters):
    if p.ndim != 2:
        continue
    plt.plot([update_to_data[j][i] for j in range(len(update_to_data))])
    legend.append(f'param {i}')
    plt.plot([0, len(update_to_data)], [-3, -3], 'k')
plt.legend(legend)
plt.show()



