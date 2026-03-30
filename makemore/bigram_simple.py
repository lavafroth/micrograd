import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Bigram Model
    """)
    return


@app.cell
def _():
    words = None
    with open('names-brazil-top100000.txt') as handle:
        words = handle.read().splitlines()
    return (words,)


@app.cell
def _(words):
    words[:6]
    return


@app.cell
def _(words):
    len(words)
    return


@app.cell
def _(words):
    min(len(w) for w in words)
    return


@app.cell
def _(words):
    max(len(w) for w in words)
    return


@app.cell
def _(words):
    def _():
        b = {}
        for w in words[:3]:
            w = ['<start>', *w, '<end>']
            for bigram in zip(w, w[1:]):
                b[bigram] = b.get(bigram, 0) + 1
                print(bigram)

        return b

    _()
    return


@app.cell
def _(words):
    def _():
        b = {}
        for w in words:
            w = ['<start>', *w, '<end>']
            for bigram in zip(w, w[1:]):
                b[bigram] = b.get(bigram, 0) + 1

        return sorted(b.items(), key=lambda kv: -kv[1])
    _()
    return


@app.cell
def _(words):
    charset = set()
    for word in words:
        for char in word:
            charset.add(char)
    charset = sorted(charset)
    stoi = {s: i+1 for i, s in enumerate(charset)}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    return charset, itos, stoi


@app.cell
def _(charset):
    '.' in charset
    return


@app.cell
def _(stoi):
    list(stoi.items())[-2:]
    return


@app.cell
def _(charset):
    vocab_size = len(charset) + 1
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    return N, vocab_size


@app.cell
def _(N, stoi, words):
    for w in words:
        w = ['.', *w, '.']
        for i0, i1 in zip(w, w[1:]):
            i0 = stoi[i0]
            i1 = stoi[i1]
            N[i0, i1] += 1
    return


@app.cell
def _(N):
    plt.imshow(N)
    return


@app.cell
def _(N, itos, vocab_size):
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    for i in range(vocab_size):
        for j in range(vocab_size):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")
    plt.gca()
    return (i,)


@app.cell
def _(N):
    N[0]
    return


@app.cell
def _(N):
    p = N[0].float()
    p /= p.sum()
    return (p,)


@app.cell
def _(p):
    p
    return


@app.cell
def _():
    g = torch.Generator().manual_seed(5)
    p_1 = torch.rand(3, generator=g)
    p_1 /= p_1.sum()
    return g, p_1


@app.cell
def _(p_1):
    p_1
    return


@app.cell
def _(g, p_1):
    torch.multinomial(p_1, generator=g, replacement=True, num_samples=20)
    return


@app.cell
def _(g, itos, p):
    ix = torch.multinomial(p, num_samples=1, generator=g, replacement=True).item()

    itos[ix]
    return


@app.cell
def _(N):
    P = N.float() + 1
    P /= P.sum(dim=1, keepdim=True)
    return (P,)


@app.cell
def _(P):
    P[0].sum()
    return


@app.cell
def _(P, itos):
    def generate_name(g):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, generator=g, replacement=True).item()
            if ix == 0:
                if len(out) < 4 or len(out) > 10:
                    out = []
                    continue
                break
            out.append(itos[ix])
        return ''.join(out)

    def _():
        g = torch.Generator().manual_seed(5)
        for _ in range(10):
            print(generate_name(g))

    _()
    return


@app.cell
def _(P, stoi, words):
    def print_nll_for_words(words):
        log_likelihood = 0.0
        n = 0
        for w in words:
            w = ['.', *w, '.']
            for c0, c1 in zip(w, w[1:]):
                i0 = stoi[c0]
                i1 = stoi[c1]
                p = P[i0, i1]
                log_prob = p.log()
                log_likelihood += log_prob
                n += 1
                # print(f'{c0}{c1} {p:0.4f} {log_prob:0.4f}')
        nll_loss = -log_likelihood
        average_nll_loss = nll_loss / n

        print(f"{nll_loss = :0.04f}") 
        print(f"{average_nll_loss = :0.04f}") 

    print_nll_for_words(["himadrizzq"])
    print_nll_for_words(words)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Create the training set
    """)
    return


@app.cell
def _(stoi, words):
    def _():
        xs, ys = [], []
        for w in words[:1]:
            w = ['.', *w, '.']
            for i0, i1 in zip(w, w[1:]):
                i0 = stoi[i0]
                i1 = stoi[i1]

                xs.append(i0)
                ys.append(i1)

            xs = torch.tensor(xs, dtype=torch.long)
            ys = torch.tensor(ys, dtype=torch.long)

        return xs, ys

    xs, ys = _()
    return xs, ys


@app.cell
def _(vocab_size, xs):
    xenc = F.one_hot(xs, num_classes=vocab_size).float()
    print(xenc.dtype, xenc.shape)
    plt.imshow(xenc)
    return (xenc,)


@app.cell
def _(vocab_size, xenc):
    W_0 = torch.randn((vocab_size, 1))
    print("W", W_0.shape, W_0[:5].tolist(), "...")
    activation_0 = xenc @ W_0
    print("activation", activation_0.shape, activation_0.tolist())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $(7, 27) \times (27, 1) = (7, 1)$
    """)
    return


@app.cell
def _(vocab_size, xenc):
    W_27_neurons = torch.randn((vocab_size, vocab_size))
    activation_1 = xenc @ W_27_neurons
    print('activation', activation_1.shape)

    print('sanity check')
    print(f'{activation_1[3, 13] = :0.4f}')
    manual_dot_product = (xenc[3] * W_27_neurons[:, 13]).sum().item()
    print(f'{manual_dot_product = :0.4f}')
    return (activation_1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $(7, 27) \times (27, 27) = (7, 27)$
    """)
    return


@app.cell
def _(activation_1):
    logits = activation_1
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    return (probs,)


@app.cell
def _(probs):
    probs.shape
    return


@app.cell
def _(i, itos, probs, xs, ys):
    nlls = torch.zeros(7)
    md = []
    for _i in range(7):
        x = xs[_i].item()
        y = ys[_i].item()
        y_true_prob = probs[_i, y]
        logp = y_true_prob.log()
        nll = -logp

        lines = (
            f"bigram example {i + 1}: `{itos[x]}{itos[y]}` indices ({x}, {y})",
            f"input to the neural net {x}",
            f"output probabilities {probs[_i]}",
            f"true label prediction {y_true_prob.item()}",
            f"log likelihood {logp}",
            f"nll {nll}",
        )
        nlls[_i] = nll
        md.append("  \n".join(lines))

    mo.md(
        f"Average negative log likelihood, nll loss = {nlls.mean().item():0.04f}\n---\n"
        + "\n---\n".join(md)
    )
    return


@app.cell
def _(ys):
    ys.unsqueeze(1)
    return


@app.cell
def _(probs, ys):
    for _i in range(len(ys)):
        print(probs[_i, ys[_i]].item())
    return


@app.cell
def _(probs, ys):
    plucked = torch.gather(probs, 1, ys.unsqueeze(1))
    plucked
    return (plucked,)


@app.cell
def _(plucked):
    loss = -plucked.squeeze().log().mean()
    loss
    return


@app.cell
def _(vocab_size, xenc, ys):
    def _():
        g = torch.Generator().manual_seed(5)
        W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

        def gimme_loss():
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)
            loss = -torch.gather(probs, 1, ys.unsqueeze(1)).squeeze().log().mean()
            return loss

        loss = gimme_loss()
        print(loss.item())
    
        W.grad = None
        loss.backward()

        W.data += -0.1 * W.grad

        loss = gimme_loss()
        print(loss.item())
        return
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Putting it all together
    """)
    return


@app.cell
def _(stoi, words):
    def _():
        xs, ys = [], []
        for w in words:
            w = ['.', *w, '.']
            for i0, i1 in zip(w, w[1:]):
                i0 = stoi[i0]
                i1 = stoi[i1]

                xs.append(i0)
                ys.append(i1)

        xs = torch.tensor(xs, dtype=torch.long)
        ys = torch.tensor(ys, dtype=torch.long)

        print('number of examples: ', xs.nelement())

        return xs, ys

    xs_1, ys_1 = _()
    return xs_1, ys_1


@app.cell
def _(xs_1):
    xs_1
    return


@app.cell
def _(vocab_size, xs_1, ys_1):
    def train_loop():
        g = torch.Generator().manual_seed(5)
        W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    
        for k in range(100):
            xenc = F.one_hot(xs_1, num_classes=vocab_size).float()
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)
            likelihood = torch.gather(probs, 1, ys_1.unsqueeze(dim=1)).squeeze()
            log_likelihood = likelihood.log()
            nll = -log_likelihood
            loss = nll.mean()

            print(loss.item())
            W.grad = None
            loss.backward()
            W.data += -20 * W.grad

    return (train_loop,)


@app.cell
def _(train_loop):
    train_loop()
    return


@app.cell
def _(vocab_size, xs_1, ys_1):
    def train_loop_regularized():
        g = torch.Generator().manual_seed(5)
        W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    
        for k in range(100):
            xenc = F.one_hot(xs_1, num_classes=vocab_size).float()
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)
            likelihood = torch.gather(probs, 1, ys_1.unsqueeze(dim=1)).squeeze()
            log_likelihood = likelihood.log()
            nll = -log_likelihood + 0.01 * (W**2).mean()
            loss = nll.mean()

            print(loss.item(), end="\r")
            W.grad = None
            loss.backward()
            W.data += -20 * W.grad

        return W

    return (train_loop_regularized,)


@app.cell
def _(train_loop_regularized):
    trained_w = train_loop_regularized()
    return (trained_w,)


@app.cell
def _(itos, trained_w, vocab_size):
    def _():
        g = torch.Generator().manual_seed(5)
        for i in range(7):
            out = []
            ix = 0
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
                logits = xenc @ trained_w
                counts = logits.exp()
                probs = counts / counts.sum(dim=1, keepdim=True)

                ix = torch.multinomial(probs, generator=g, num_samples=1, replacement=True).item()
                if ix == 0:
                    if len(out) < 4 or len(out) > 10:
                        out = []
                        continue
                    break
                out.append(itos[ix])
            print(''.join(out))


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
