import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from types import SimpleNamespace

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


random.seed(5)


n = len(words)
n1 = int(0.8 * n)
n2 = int(0.9 * n)


random.shuffle(words)
dataset = SimpleNamespace(
    train=build_dataset(words[:n1]),
    valid=build_dataset(words[n1:n2]),
    test=build_dataset(words[n2:]),
)


def cmp(attribute_name, d_tensor: torch.Tensor, tensor: torch.Tensor):
    ex = torch.all(d_tensor == tensor.grad).item()
    app = torch.allclose(d_tensor, tensor.grad)
    maxdiff = (d_tensor - tensor.grad).abs().max().item()
    closeness = "exact" if ex else "approx" if app else "differ"
    print(f"{attribute_name:37s} | {closeness:6s} | maxdiff: {maxdiff}")


n_embed = 10
n_hidden = 64

gen = torch.Generator().manual_seed(5)

# set training hyperparameters
max_steps = 100_000
batch_size = 32


# addressing the weight initialization automatically
fan_in = block_size * n_embed
w_scale = fan_in**-0.5  # try 5, 1, 0.2
# copying kaiming initialization
tanh_gain = 5 / 3
w_scale = tanh_gain / (fan_in**0.5)

C = torch.randn((vocab_size, n_embed), generator=gen)

# layer 1
w1 = torch.randn((n_embed * block_size, n_hidden), generator=gen) * w_scale
b1 = torch.randn((n_hidden,), generator=gen) * 0.01

# layer 2
w2 = torch.randn((n_hidden, vocab_size), generator=gen) * 0.01
b2 = torch.zeros((vocab_size,))

# batch norm
batch_norm_gain = torch.ones((1, n_hidden))
batch_norm_bias = torch.zeros((1, n_hidden))

parameters = [
    C,
    w1,
    b1,
    w2,
    b2,
    batch_norm_bias,
    batch_norm_gain,
]

for p in parameters:
    p.requires_grad = True

sum(p.nelement() for p in parameters)


ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,), generator=gen)
x = dataset.train.x[ix]
y = dataset.train.y[ix]

emb = C[x]
embcat = emb.view(emb.shape[0], -1)
h_pre_batch_norm = embcat @ w1 + b1

current_batch_norm_mean = 1 / batch_size * h_pre_batch_norm.sum(dim=0, keepdim=True)
current_batch_norm_difference = h_pre_batch_norm - current_batch_norm_mean
current_batch_norm_difference_squared = current_batch_norm_difference**2
current_batch_norm_variance = (
    1 / (batch_size - 1)
) * current_batch_norm_difference_squared.sum(dim=0, keepdim=True)

current_batch_norm_std_inverse = (current_batch_norm_variance + 1e-5) ** -0.5
batch_norm_raw = current_batch_norm_difference * current_batch_norm_std_inverse
h_pre_activation = batch_norm_gain * batch_norm_raw + batch_norm_bias
h = torch.tanh(h_pre_activation)

logits = h @ w2 + b2

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(dim=1, keepdim=True)
counts_sum_inverse = counts_sum**-1
probs = counts * counts_sum_inverse
logprobs = probs.log()
loss = -logprobs[range(batch_size), y].mean()

for p in parameters:
    p.grad = None
for t in [
    emb,
    embcat,
    h_pre_batch_norm,
    current_batch_norm_mean,
    current_batch_norm_difference,
    current_batch_norm_difference_squared,
    current_batch_norm_variance,
    current_batch_norm_std_inverse,
    batch_norm_raw,
    h_pre_activation,
    h,
    logits,
    logit_maxes,
    norm_logits,
    counts,
    counts_sum,
    counts_sum_inverse,
    probs,
    logprobs,
]:
    t.retain_grad()
loss.backward()
loss

with torch.no_grad():
    d_logprobs = torch.zeros_like(logprobs)
    d_logprobs[range(batch_size), y] = -1 / batch_size
    d_probs = probs**-1 * d_logprobs
    d_counts_sum_inverse = (counts * d_probs).sum(dim=1, keepdim=True)
    d_counts_sum = -(counts_sum**-2) * d_counts_sum_inverse
    d_counts = torch.ones_like(counts) * d_counts_sum + counts_sum_inverse * d_probs
    d_norm_logits = counts * d_counts
    d_logits = d_norm_logits.clone()
    d_logit_maxes = (-d_norm_logits).sum(dim=1, keepdim=True)
    # d_logits += F.one_hot(logits.max(dim=1).indices, num_classes=logits.shape[1]) * d_logit_maxes
    d_logits[range(d_logits.shape[0]), logits.max(dim=1).indices] += (
        d_logit_maxes.squeeze(1)
    )
    d_h = d_logits @ w2.T
    d_w2 = h.T @ d_logits
    d_b2 = d_logits.sum(dim=0)
    d_h_pre_activation = (1.0 - h**2.0) * d_h
    d_batch_norm_raw = batch_norm_gain * d_h_pre_activation
    d_batch_norm_gain = (batch_norm_raw * d_h_pre_activation).sum(dim=0, keepdim=True)
    d_batch_norm_bias = d_h_pre_activation.sum(dim=0, keepdim=True)
    d_current_batch_norm_std_inverse = (
        current_batch_norm_difference * d_batch_norm_raw
    ).sum(0, keepdim=True)
    d_current_batch_norm_variance = (
        -0.5
        * (current_batch_norm_variance + 1e-5) ** -1.5
        * d_current_batch_norm_std_inverse
    )
    # TODO: read about bessel's correction
    d_current_batch_norm_difference_squared = (
        1
        / (batch_size - 1)
        * d_current_batch_norm_variance.broadcast_to(
            current_batch_norm_difference_squared.shape
        )
    )
    d_current_batch_norm_difference = (
        2 * current_batch_norm_difference * d_current_batch_norm_difference_squared
        + current_batch_norm_std_inverse * d_batch_norm_raw
    )
    d_current_batch_norm_mean = -d_current_batch_norm_difference.sum(
        dim=0, keepdim=True
    )
    d_h_pre_batch_norm = (
        d_current_batch_norm_mean.broadcast_to(h_pre_batch_norm.shape) / batch_size
        + d_current_batch_norm_difference
    )
    d_embcat = d_h_pre_batch_norm @ w1.T
    d_emb = d_embcat.view(emb.shape)
    d_w1 = embcat.T @ d_h_pre_batch_norm
    d_b1 = d_h_pre_batch_norm.sum(dim=0)
    d_C = torch.zeros_like(C)
    d_C.index_add_(dim=0, index=x.view(-1), source=d_emb.view(-1, d_emb.shape[-1]))
    cmp("logprobs", d_logprobs, logprobs)
    cmp("probs", d_probs, probs)
    cmp("counts_sum_inverse", d_counts_sum_inverse, counts_sum_inverse)
    cmp("counts_sum", d_counts_sum, counts_sum)
    cmp("counts", d_counts, counts)
    cmp("norm_logits", d_norm_logits, norm_logits)
    cmp("logit_maxes", d_logit_maxes, logit_maxes)
    cmp("logits", d_logits, logits)
    cmp("h", d_h, h)
    cmp("w2", d_w2, w2)
    cmp("b2", d_b2, b2)
    cmp("h_pre_activation", d_h_pre_activation, h_pre_activation)
    cmp("batch_norm_gain", d_batch_norm_gain, batch_norm_gain)
    cmp("batch_norm_raw", d_batch_norm_raw, batch_norm_raw)
    cmp("batch_norm_bias", d_batch_norm_bias, batch_norm_bias)
    cmp(
        "current_batch_norm_std_inverse",
        d_current_batch_norm_std_inverse,
        current_batch_norm_std_inverse,
    )
    cmp(
        "current_batch_norm_variance",
        d_current_batch_norm_variance,
        current_batch_norm_variance,
    )
    cmp(
        "current_batch_norm_difference_squared",
        d_current_batch_norm_difference_squared,
        current_batch_norm_difference_squared,
    )
    cmp(
        "current_batch_norm_difference",
        d_current_batch_norm_difference,
        current_batch_norm_difference,
    )
    cmp("current_batch_norm_mean", d_current_batch_norm_mean, current_batch_norm_mean)
    cmp("h_pre_batch_norm", d_h_pre_batch_norm, h_pre_batch_norm)
    cmp("w1", d_w1, w1)
    cmp("b1", d_b1, b1)
    cmp("embcat", d_embcat, embcat)
    cmp("emb", d_emb, emb)
    cmp("C", d_C, C)


### Backward pass through the coalesced loss provided by torch

loss_coalesced = F.cross_entropy(logits, y)

with torch.no_grad():
    counts = logits.exp()
    d_logits = (
        (
            counts / counts.sum(dim=1, keepdim=True)
            - F.one_hot(y, num_classes=logits.shape[1])
        ) / y.shape[0]
    )
    print(d_logits.shape)
    cmp("logits", d_logits, logits)


F.softmax(logits, dim=1)[0]
d_logits[0] * batch_size

plt.close("all")
plt.figure(figsize=(8, 8))
plt.imshow(d_logits.detach(), cmap="gray")
plt.show()

### Backward pass through the batchnorm in one go

h_pre_activation_coalesced = (
    batch_norm_gain
    * (h_pre_batch_norm - h_pre_batch_norm.mean(dim=0, keepdim=True))
    / torch.sqrt(h_pre_batch_norm.var(dim=0, keepdim=True, unbiased=True))
    + batch_norm_bias
)


print("max diff:", (h_pre_activation - h_pre_activation_coalesced).abs().mean())

# calculate d_h_pre_batch_norm given d_h_pre_activation

d_h_pre_batch_norm = batch_norm_gain * current_batch_norm_std_inverse * (
    d_h_pre_activation
    - batch_norm_raw / (batch_size - 1) * (d_h_pre_activation * batch_norm_raw).sum(0)
    - d_h_pre_activation.sum(dim=0) / batch_size
)
cmp("h_pre_batch_norm", d_h_pre_batch_norm, h_pre_batch_norm)

###


n_embed = 10
n_hidden = 200

gen = torch.Generator().manual_seed(5)

# set training hyperparameters
max_steps = 100_000
batch_size = 32


fan_in = block_size * n_embed
w_scale = fan_in**-0.5
tanh_gain = 5 / 3
w_scale = tanh_gain / (fan_in**0.5)

C = torch.randn((vocab_size, n_embed), generator=gen)

# layer 1
w1 = torch.randn((n_embed * block_size, n_hidden), generator=gen) * w_scale
b1 = torch.randn((n_hidden,), generator=gen) * 0.01

# layer 2
w2 = torch.randn((n_hidden, vocab_size), generator=gen) * 0.01
b2 = torch.randn((vocab_size,)) * 0.1

# batch norm
batch_norm_gain = torch.randn((1, n_hidden)) * 0.1 + 1.0
batch_norm_bias = torch.randn((1, n_hidden)) * 0.1

parameters = [
    C,
    w1,
    b1,
    w2,
    b2,
    batch_norm_bias,
    batch_norm_gain,
]

for p in parameters:
    p.requires_grad = True

sum(p.nelement() for p in parameters)

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

# training loop
debug_backprop_ninja = False # change to True to compare manual backprop and torch autograd

log_losses = []
running_batch_norm_std = torch.ones((1, n_hidden))
running_batch_norm_mean = torch.zeros((1, n_hidden))
batch_norm_momentum = 0.001

max_steps = 100_000

for run in range(max_steps):
    ix = torch.randint(high=dataset.train.x.shape[0], size=(batch_size,), generator=gen)
    x = dataset.train.x[ix]
    y = dataset.train.y[ix]

    with torch.no_grad():
        emb = C[x]
        embcat = emb.view(emb.shape[0], -1)
        h_pre_batch_norm = embcat @ w1 + b1

        current_batch_norm_mean = 1 / batch_size * h_pre_batch_norm.sum(dim=0, keepdim=True)
        current_batch_norm_difference = h_pre_batch_norm - current_batch_norm_mean
        current_batch_norm_difference_squared = current_batch_norm_difference**2
        current_batch_norm_variance = (
            1 / (batch_size - 1)
        ) * current_batch_norm_difference_squared.sum(dim=0, keepdim=True)

        current_batch_norm_std_inverse = (current_batch_norm_variance + 1e-5) ** -0.5
        batch_norm_raw = current_batch_norm_difference * current_batch_norm_std_inverse
        h_pre_activation = batch_norm_gain * batch_norm_raw + batch_norm_bias
        h = torch.tanh(h_pre_activation)

        running_batch_norm_mean = (
            running_batch_norm_mean * (1 - batch_norm_momentum)
            + current_batch_norm_mean * batch_norm_momentum
        )
        running_batch_norm_std = (
            running_batch_norm_std * (1 - batch_norm_momentum)
            + current_batch_norm_variance**.5 * batch_norm_momentum
        )


        logits = h @ w2 + b2
        loss = F.cross_entropy(logits, y)
        loss_scalar = loss.item()

        for p in parameters:
            p.grad = None

        counts = logits.exp()
        d_logits = (
            (
                counts / counts.sum(dim=1, keepdim=True)
                - F.one_hot(y, num_classes=logits.shape[1])
            ) / y.shape[0]
        )
        d_h = d_logits @ w2.T
        d_w2 = h.T @ d_logits
        d_b2 = d_logits.sum(dim=0)
        d_h_pre_activation = (1.0 - h**2.0) * d_h

        d_batch_norm_gain = (batch_norm_raw * d_h_pre_activation).sum(dim=0, keepdim=True)
        d_batch_norm_bias = d_h_pre_activation.sum(dim=0, keepdim=True)
        d_h_pre_batch_norm = batch_norm_gain * current_batch_norm_std_inverse * (
            d_h_pre_activation
            - batch_norm_raw / (batch_size - 1) * (d_h_pre_activation * batch_norm_raw).sum(0)
            - d_h_pre_activation.sum(dim=0) / batch_size
        )

        d_embcat = d_h_pre_batch_norm @ w1.T
        d_emb = d_embcat.view(emb.shape)
        d_w1 = embcat.T @ d_h_pre_batch_norm
        d_b1 = d_h_pre_batch_norm.sum(dim=0)
        d_C = torch.zeros_like(C)
        d_C.index_add_(dim=0, index=x.view(-1), source=d_emb.view(-1, d_emb.shape[-1]))
        batch_norm_bias.grad = d_batch_norm_bias
        batch_norm_gain.grad = d_batch_norm_gain
        grads = [d_C, d_w1, d_b1, d_w2, d_b2, d_batch_norm_bias, d_batch_norm_gain]

    # loss.backward()

        lr = 0.001 if run > max_steps // 2 else 0.01

        if debug_backprop_ninja and run > 100:
            for p, grad in zip(parameters, grads):
                cmp(str(tuple(p.shape)), grad, p)
            break

        for p, grad in zip(parameters, grads):
            p.data -= grad * lr

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
    h_shifted = h_normalized * batch_norm_gain + batch_norm_bias

    h = torch.tanh(h_shifted)

    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)
    return loss.item()


print("train", split_loss(dataset.train))
print("valid", split_loss(dataset.valid))
