import torch
from torch import nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 30000
eval_interval = 3000
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32

torch.manual_seed(5)

with open("nanogpt/chat-cleaned.md") as handle:
    text = handle.read()

chars = sorted(set(text))

try:
    newline_index = chars.index("\n")
except ValueError:
    newline_index = 0
    pass

vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = dict(enumerate(chars))


def encoder(s):
    return [stoi[c] for c in s]


def decoder(s):
    return "".join(itos[c] for c in s)


data = torch.tensor(encoder(text), dtype=torch.long)

n = data.size(0) * 9 // 10
train_data = data[:n]
valid_data = data[n:]

block_size = 8
train_data[: block_size + 1]
decoder(train_data[: block_size + 1].tolist())


def get_batch(split):
    data = train_data if split == "train" else valid_data
    ix = torch.randint(high=len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "valid"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()

    return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    @staticmethod
    def linear(out_features: int):
        return nn.Linear(in_features=n_embed, out_features=out_features, bias=False)

    def __init__(self, head_size: int):
        super().__init__()
        self.key = Head.linear(head_size)
        self.query = Head.linear(head_size)
        self.value = Head.linear(head_size)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)).log()
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5 + self.tril[:T, :T]  # ty:ignore[not-subscriptable]
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        n_embed = head_size * num_heads
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,  # along channel dimension
        )
        out = self.proj(out)

        return out


class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        assert n_embed % n_heads == 0
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(head_size, n_heads)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, 4),
            Block(n_embed, 4),
            Block(n_embed, 4),
            nn.LayerNorm(n_embed)
        )
        # self.self_attention_head = Head(n_embed)
        # num_heads = 4
        # head_size = n_embed // num_heads
        # self.self_attention_heads = MultiHeadAttention(head_size, num_heads)

        # self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, indices: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = indices.shape
        token_embeddings = self.token_embedding_table(indices)
        # also (batch, token, n_embed)
        positional_embeddings = self.position_embedding_table(torch.arange(T))
        x = token_embeddings + positional_embeddings
        # x = self.self_attention_heads(x)
        # x = self.self_attention_heads(x)  # still (batch, token, n_embed)
        # x = self.ffwd(x)

        x = self.blocks(x)
        logits = self.lm_head(x)
        # (batch, time, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        # logits = logits.view(B*T, C)
        # targets = targets.view(B*T)
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, indices: torch.Tensor, max_new_tokens: int):
        # indices: (B, T)
        for _ in range(max_new_tokens):
            indices_conditional = indices[:, -block_size:]
            logits, loss = self(indices_conditional)  # (batch, time, channel)
            logits = logits[:, -1, :]  # (batch, channel)
            probs = F.softmax(logits, dim=-1)  # softmax all channel values
            next_ix = torch.multinomial(probs, num_samples=1)
            assert next_ix.shape == (indices.size(0), 1)
            indices = torch.cat((indices, next_ix), dim=1)
        return indices


m = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for run in range(max_iters):
    if not run % eval_interval:
        losses = estimate_loss(m)
        print(f"step {run:07d}: {losses=}")

    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()

print(loss.item())

prompt = ""
while prompt != "stop":
    context = torch.ones((1, 1), dtype=torch.long, device=device) * newline_index
    print(decoder(m.generate(context, 300)[0].tolist()))
    prompt = input("?!> ")
# torch.log(torch.tensor([logits.size(1)]))
