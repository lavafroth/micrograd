import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    return F, plt, torch


@app.cell
def _():
    with open('./makemore/names-brazil-top100000.txt') as handle:
        words = handle.read().splitlines()

    print(len(words), words[:8])
    return (words,)


@app.cell
def _(words):
    chars = set()
    for word in words:
        chars |= set(word)
    chars = sorted(chars)
    print(chars[:5])
    return (chars,)


@app.cell
def _(chars):
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    print(len(itos))
    print(len(stoi))
    return itos, stoi


@app.cell
def _():
    from types import SimpleNamespace

    return (SimpleNamespace,)


@app.cell
def _(SimpleNamespace):
    exploration = SimpleNamespace()
    return (exploration,)


@app.cell
def _(exploration, itos, stoi, torch, words):
    def generate_dataset(block_size=3, debug=True):
        x, y = [], []
        for w in words[:5]:
            print(w)
            context = [0] * block_size
            for ch in w + ".":
                ix = stoi[ch]
                x.append(context)
                y.append(ix)
                print("".join(itos[i] for i in context), "maps to", itos[ix])
                context = context[1:] + [ix]

        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y

    exploration.x, exploration.y = generate_dataset()
    print(exploration.x.shape)
    print(exploration.y.shape)
    return


@app.cell
def _(exploration, torch):
    exploration.C = torch.randn((27, 2))
    print(exploration.C)
    return


@app.cell
def _(F, exploration, torch):
    exploration.one_hot_5 = F.one_hot(torch.tensor([5]), num_classes=27).float()
    print(exploration.one_hot_5, exploration.one_hot_5.dtype)
    print(exploration.one_hot_5 @ exploration.C)
    print('using x', exploration.x.shape, 'to index into embedding matrix')
    print(exploration.C[exploration.x].shape)
    return


@app.cell
def _(exploration):
    exploration.emb = exploration.C[exploration.x]
    return


@app.cell
def _(exploration, torch):
    def _():
        emb = exploration.emb
        print(torch.cat((emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]), 1).shape)
        print(torch.cat(torch.unbind(emb, dim=1), dim=1).shape)
    _()
    return


@app.cell
def _(torch):
    def _():
        a = torch.arange(0, 10, 1)
        a = a.view((5, 2))
        print(a.storage())
        print(a.shape)
    _()
    return


@app.cell
def _(exploration):
    exploration.emb_flat = exploration.emb.view((exploration.emb.shape[0], -1))
    return


@app.cell
def _(exploration, torch):
    exploration.w1 = torch.randn((exploration.emb_flat.shape[1], 100))
    exploration.b1 = torch.randn((100, ))
    return


@app.cell
def _(exploration):
    exploration.matmul = exploration.emb_flat @ exploration.w1
    print(exploration.matmul.shape, "+", exploration.b1.shape)
    exploration.activations = exploration.matmul + exploration.b1
    print(exploration.activations.shape)
    return


@app.cell
def _(exploration, torch):
    exploration.hidden_state = torch.tanh(exploration.activations)
    return


@app.cell
def _(exploration):
    exploration.hidden_state.shape
    return


@app.cell
def _(exploration):
    exploration.hidden_state
    return


@app.cell
def _(exploration, torch):
    exploration.w2 = torch.randn((100, 27))
    exploration.b2 = torch.randn((27,))
    exploration.logits = exploration.hidden_state @ exploration.w2 + exploration.b2
    return


@app.cell
def _(exploration):
    print(exploration.logits.shape)
    exploration.logits[:2, :5]
    return


@app.cell
def _(exploration):
    exploration.counts = exploration.logits.exp()
    exploration.probs = exploration.counts / exploration.counts.sum(dim=1, keepdim=True)
    return


@app.cell
def _(exploration):
    exploration.probs.shape
    return


@app.cell
def _(exploration):
    exploration.probs[0].sum()
    return


@app.cell
def _(exploration):
    exploration.y
    return


@app.cell
def _(exploration):
    exploration.gathered = exploration.probs.gather(dim=1, index=exploration.y.unsqueeze(1)).squeeze()
    return


@app.cell
def _(exploration):
    loss = -exploration.gathered.log().mean()
    loss
    return


@app.cell
def _(F, exploration):
    F.cross_entropy(exploration.logits, exploration.y)
    return


@app.cell
def _(torch):
    def _(subtract_max=False):
        logits = torch.tensor([-5, 3, 100, 2])
        if subtract_max:
            logits -= logits.max()
        counts = logits.exp()
        probs = counts / counts.sum()
        print(probs)

    _(subtract_max=True)
    return


@app.cell
def _(SimpleNamespace, chars, torch):
    train_0 = SimpleNamespace()
    train_0.g = torch.Generator().manual_seed(5)
    train_0.embedding_size = 2
    train_0.context_size = 3
    train_0.vocab_size = len(chars) + 1
    train_0.C = torch.randn((train_0.vocab_size, train_0.embedding_size), generator=train_0.g)
    train_0.w1 = torch.randn((train_0.embedding_size * train_0.context_size, 100), generator=train_0.g)
    train_0.b1 = torch.randn((100, ), generator=train_0.g)

    train_0.w2 = torch.randn((100, train_0.vocab_size), generator=train_0.g)
    train_0.b2 = torch.randn((train_0.vocab_size,), generator=train_0.g)

    train_0.parameters = [train_0.C, train_0.w1, train_0.b1, train_0.w2,  train_0.b2]
    return (train_0,)


@app.cell
def _(train_0):
    sum(p.nelement() for p in train_0.parameters)
    return


@app.cell
def _(train_0):
    for p in train_0.parameters:
        p.requires_grad = True
    return


@app.cell
def _(F, exploration, torch, train_0):
    def _(lr=1e-3):
        for _ in range(20):
            emb = train_0.C[exploration.x]
            h = torch.tanh(emb.view(emb.shape[0], -1) @ train_0.w1 + train_0.b1)
            logits = h @ train_0.w2 + train_0.b2
    
            loss = F.cross_entropy(logits, exploration.y)
    
            print(loss.item())
        
            for p in train_0.parameters:
                p.grad = None
            loss.backward()
        
            for p in train_0.parameters:
                p.data -= p.grad * lr

        print(logits.max(dim=1))
        print(exploration.y)
        return


    _(lr=.1)
    return


@app.cell
def _(SimpleNamespace):
    train_1 = SimpleNamespace()
    return (train_1,)


@app.cell
def _(chars, torch, train_1):
    train_1.g = torch.Generator().manual_seed(5)
    train_1.embedding_size = 2
    train_1.context_size = 3
    train_1.vocab_size = len(chars) + 1
    kwargs = dict(generator=train_1.g, requires_grad=True)
    train_1.C = torch.randn((train_1.vocab_size, train_1.embedding_size), **kwargs)

    train_1.w1 = torch.randn((train_1.embedding_size * train_1.context_size, 100), **kwargs)
    train_1.b1 = torch.randn((100, ), **kwargs)

    train_1.w2 = torch.randn((100, train_1.vocab_size), **kwargs)
    train_1.b2 = torch.randn((train_1.vocab_size,), **kwargs)

    train_1.parameters = [train_1.C, train_1.w1, train_1.b1, train_1.w2,  train_1.b2]
    return


@app.cell
def _(stoi, torch, train_1, words):
    def generate_dataset_1(words, block_size=3):
        x, y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + ".":
                ix = stoi[ch]
                x.append(context)
                y.append(ix)
                context = context[1:] + [ix]
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y

    train_1.x, train_1.y = generate_dataset_1(words=words)
    return (generate_dataset_1,)


@app.cell
def _(train_1):
    train_1.x.shape
    return


@app.cell
def _(train_1):
    train_1.x.dtype, train_1.y.dtype
    return


@app.cell
def _(torch, train_1):
    torch.randint(0, train_1.x.shape[0], (32, ))
    return


@app.cell
def _(plt, torch):
    lr_exponent = torch.linspace(-3, -1, 1000)
    lrs = 5 * 10**lr_exponent
    plt.plot(lr_exponent, lrs)
    return lr_exponent, lrs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning rate experimentation
    - good = 1
    - broken = 0.5
    - fast = 0.01
    - slow = 0.005
    """)
    return


@app.cell
def _(F, lr_exponent, lrs, plt, torch, train_1):
    def _(lr=0.01):
        lrsi = []
        lossi = []
        for run in range(1000):
            minibatch_indices = torch.randint(high=train_1.x.shape[0], size=(32, ))
            x = train_1.x[minibatch_indices]
            y = train_1.y[minibatch_indices]

            if run == 0:
                assert x.shape == (32, 3)
                assert y.shape == (32, )
            
            emb = train_1.C[x]

            if run == 0:
                assert emb.shape == (32, 3, 2)

            emb = emb.view(emb.shape[0], -1)

            h = torch.tanh(emb @ train_1.w1 + train_1.b1)
            logits = h @ train_1.w2 + train_1.b2

            loss = F.cross_entropy(logits, y)
            # print(loss.item())
        
            for p in train_1.parameters:
                p.grad = None

            loss.backward()

            for p in train_1.parameters:
                p.data -= p.grad * lrs[run]

            lrsi.append(lr_exponent[run])
            lossi.append(loss.item())

        print(loss.item())
        return lrsi, lossi


    plt.plot(*_(0.005))
    return


@app.cell
def _():
    5*10**-1.7
    return


@app.cell
def _(F, torch, train_1):
    def _(lr=0.01):
        for run in range(10000):
            minibatch_indices = torch.randint(high=train_1.x.shape[0], size=(32, ))
            x = train_1.x[minibatch_indices]
            y = train_1.y[minibatch_indices]
            
            emb = train_1.C[x]
            emb = emb.view(emb.shape[0], -1)

            h = torch.tanh(emb @ train_1.w1 + train_1.b1)
            logits = h @ train_1.w2 + train_1.b2

            loss = F.cross_entropy(logits, y)
            # print(loss.item())
        
            for p in train_1.parameters:
                p.grad = None

            loss.backward()

            for p in train_1.parameters:
                p.data -= p.grad * lr

        print(loss.item())


    # _(5*10**-1.7)
    _(5*10**-2.7)
    return


@app.cell
def _(F, torch, train_1):
    def _():
        emb = train_1.C[train_1.x]
        emb = emb.view((emb.shape[0], -1))
        h = torch.tanh(emb @ train_1.w1 + train_1.b1)
        logits = h @ train_1.w2 + train_1.b2
        loss = F.cross_entropy(logits, train_1.y)
        print(loss.item())
        return loss

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let's split the dataset
    """)
    return


@app.cell
def _(SimpleNamespace, generate_dataset_1, words):
    import random

    words_copy = words.copy()
    random.seed(0)
    random.shuffle(words_copy)

    n = len(words_copy)
    n1 = int(0.8 * n)
    n2 = int(0.9 * n)

    dataset = SimpleNamespace(
        train=SimpleNamespace(),
        dev=SimpleNamespace(),
        test=SimpleNamespace(),
    )

    dataset.train.x, dataset.train.y = generate_dataset_1(words=words_copy[:n1])
    dataset.dev.x, dataset.dev.y = generate_dataset_1(words=words_copy[n1:n2])
    dataset.test.x, dataset.test.y = generate_dataset_1(words=words_copy[n2:])
    return (dataset,)


@app.cell
def _(F, dataset, torch, train_1):
    def _(lr=0.01):
        for run in range(10000):
            minibatch_indices = torch.randint(high=dataset.train.x.shape[0], size=(32, ))
            x = dataset.train.x[minibatch_indices]
            y = dataset.train.y[minibatch_indices]
            
            emb = train_1.C[x]
            emb = emb.view(emb.shape[0], -1)

            h = torch.tanh(emb @ train_1.w1 + train_1.b1)
            logits = h @ train_1.w2 + train_1.b2

            loss = F.cross_entropy(logits, y)
            # print(loss.item())
        
            for p in train_1.parameters:
                p.grad = None

            loss.backward()

            for p in train_1.parameters:
                p.data -= p.grad * lr

        print(loss.item())


    # _(5*10**-1.7)
    _(5*10**-2.7)
    return


@app.cell
def _(F, dataset, torch, train_1):
    def _():
        emb = train_1.C[dataset.dev.x]
        emb = emb.view((emb.shape[0], -1))
        h = torch.tanh(emb @ train_1.w1 + train_1.b1)
        logits = h @ train_1.w2 + train_1.b2
        loss = F.cross_entropy(logits, dataset.dev.y)
        print(loss.item())
        return loss

    _()
    return


@app.cell
def _(F, dataset, torch, train_1):
    def _():
        emb = train_1.C[dataset.train.x]
        emb = emb.view((emb.shape[0], -1))
        h = torch.tanh(emb @ train_1.w1 + train_1.b1)
        logits = h @ train_1.w2 + train_1.b2
        loss = F.cross_entropy(logits, dataset.train.y)
        print(loss.item())

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let's train a bigger model
    """)
    return


@app.cell
def _(SimpleNamespace, chars, torch):
    train_2 = SimpleNamespace(
        g=torch.Generator().manual_seed(5),
        embedding_size=10,
        context_size=3,
        vocab_size=len(chars) + 1,
        hidden_size=200,
    )
    kwargs_2 = dict(generator=train_2.g, requires_grad=True)
    train_2.C = torch.randn(
        (train_2.vocab_size, train_2.embedding_size), **kwargs_2
    )


    train_2.w1 = torch.randn(
        (train_2.embedding_size * train_2.context_size, train_2.hidden_size),
        **kwargs_2,
    )
    train_2.b1 = torch.randn((train_2.hidden_size,), **kwargs_2)

    train_2.w2 = torch.randn((train_2.hidden_size, train_2.vocab_size), **kwargs_2)
    train_2.b2 = torch.randn((train_2.vocab_size,), **kwargs_2)

    train_2.parameters = [
        train_2.C,
        train_2.w1,
        train_2.b1,
        train_2.w2,
        train_2.b2,
    ]
    return (train_2,)


@app.cell
def _(SimpleNamespace):
    tracking_2 = SimpleNamespace(
        lossi=[],
        stepi=[],
    )
    # _(5*10**-2.7)
    return (tracking_2,)


@app.cell
def _(F, dataset, torch, tracking_2, train_2):
    def _(lossi, stepi):
        for run in range(10000):
            minibatch_indices = torch.randint(
                high=dataset.train.x.shape[0], size=(64,)
            )
            x = dataset.train.x[minibatch_indices]
            y = dataset.train.y[minibatch_indices]

            emb = train_2.C[x]
            emb = emb.view(emb.shape[0], -1)

            h = torch.tanh(emb @ train_2.w1 + train_2.b1)
            logits = h @ train_2.w2 + train_2.b2

            loss = F.cross_entropy(logits, y)
            # print(loss.item())

            for p in train_2.parameters:
                p.grad = None

            loss.backward()

            if run > 5000:
                lr = 5 * 10**-2.7
            else:
                lr = 5 * 10**-1.7
            for p in train_2.parameters:
                p.data -= p.grad * lr

            lossi.append(loss.log10().item())
            stepi.append(run)
        print(loss.item())

    _(lossi=tracking_2.lossi, stepi=tracking_2.stepi)
    return


@app.cell
def _(plt, tracking_2):
    plt.plot(tracking_2.stepi, tracking_2.lossi)
    return


@app.cell
def _(F, dataset, torch, train_2):
    def _():
        emb = train_2.C[dataset.train.x]
        emb = emb.view((emb.shape[0], -1))
        h = torch.tanh(emb @ train_2.w1 + train_2.b1)
        logits = h @ train_2.w2 + train_2.b2
        loss = F.cross_entropy(logits, dataset.train.y)
        print(loss.item())

    _()
    return


@app.cell
def _(itos, torch, train_2):
    def _():
        g = torch.Generator().manual_seed(12)

        for _ in range(10):
            word = ''
            context = [0] * train_2.context_size
            while True:
                emb = train_2.C[torch.tensor([context])].view((1, -1))
                h = torch.tanh(emb @ train_2.w1 + train_2.b1)
                logits = h@train_2.w2 + train_2.b2
                probs = torch.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                if ix == 0:
                    break
                context = context[1:] + [ix]
                word += itos[ix]
            print(word)
    _()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
