import torch
from torch import nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 30000
eval_interval = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(5)

with open("nanogpt/chat-cleaned.md") as handle:
    text = handle.read()

chars = sorted(set(text))

try:
    newline_index = chars.index('\n')
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
train_data[:block_size + 1]
decoder(train_data[:block_size + 1].tolist())

def get_batch(split):
    data = train_data if split == "train" else valid_data
    ix = torch.randint(high=len(data) - block_size, size=(batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()

    return out


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
        

m = BigramLanguageModel(vocab_size).to(device)
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

context = torch.ones((1, 1), dtype=torch.long, device=device) * newline_index
print(decoder(m.generate(context, 300)[0].tolist()))
# torch.log(torch.tensor([logits.size(1)]))
