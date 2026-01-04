import torch
import torch.nn as nn
from torch.nn import functional as F

# Dataset Training Function
def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# LLM Functions and Class
def _generate_square_subsequent_mask(sz, device):
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

class QuoteGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=256, n_head=8, n_layer=8):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        try:
            self.head.weight = self.token_emb.weight
        except Exception:
            pass

        # weights
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds model block_size {self.block_size}")

        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device)[None, :]
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        mask = _generate_square_subsequent_mask(T, device=idx.device)
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
# Text Generation Section
@torch.no_grad()
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)


@torch.no_grad()
def generate(model, stoi, decode, start="Butterworth: ", length=50, temperature=1.0, top_k=0):
    model.eval()
    device = next(model.parameters()).device
    context = torch.tensor([stoi.get(c, 0) for c in start], dtype=torch.long, device=device)[None, :]
    for _ in range(length):
        ctx = context if context.size(1) <= model.block_size else context[:, -model.block_size:]
        logits = model(ctx)  # (B, T, V)
        logits = logits[:, -1, :] / max(1e-8, temperature)
        if top_k > 0:
            logits = top_k_logits(logits, k=top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        if next_id.item() == stoi['<']:
            break
        context = torch.cat((context, next_id), dim=1)
    return decode(context[0].cpu().tolist())