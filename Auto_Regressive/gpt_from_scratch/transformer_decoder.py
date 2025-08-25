import torch
import tiktoken
import time

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"[INFO] Using {device} device")

# Acceleration
torch.set_float32_matmul_precision('high')

class DataCollator:
    def __init__(self, path_to_data, ratio_train):
        with open(path_to_data, mode='r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        self.content = content
        
        # (1) using Home-made solution
        # self.vocab = sorted(list(set(content)))
        # self.n_vocab = len(self.vocab)
        # dict_ctoi = { char:idx for idx, char in enumerate(self.vocab) }
        # dict_itoc = { idx:char for idx, char in enumerate(self.vocab) }
        # self.fn_encode = lambda s: [dict_ctoi[c] for c in s]
        # self.fn_decode = lambda s: ''.join([dict_itoc[i] for i in s])

        # (2) using tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        self.n_vocab = encoding.n_vocab
        self.fn_encode = encoding.encode
        self.fn_decode = encoding.decode

        data = torch.tensor(self.fn_encode(content), dtype=torch.long)
        n = int(len(data) * ratio_train)
        self.train_data = data[:n]
        self.eval_data = data[n:]
        self.position = 0

    def collate_data(self, category, batch_size, context_size):
        data = self.train_data if category == 'train' else self.eval_data
        batch_start_idx = torch.randint(len(data) - context_size - 1, (batch_size,))
        x = torch.stack([data[idx:idx+context_size] for idx in batch_start_idx])
        y = torch.stack([data[idx+1:idx+context_size+1] for idx in batch_start_idx])
        x, y = x.to(device), y.to(device)
        return x, y

    def next(self, batch_size, context_size):
        if self.position + batch_size * context_size + 1 > len(self.train_data):
            self.position = 0
        batch = self.train_data[self.position: self.position + batch_size * context_size + 1]
        x = (batch[:-1]).view(batch_size, context_size)
        y = (batch[1:]).view(batch_size, context_size)
        self.position += batch_size * context_size
        x, y = x.to(device), y.to(device)
        return x, y

class MaskedSingleHeadAttention(torch.nn.Module):
    def __init__(self, head_size, context_size, n_embedding, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.key = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.value = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        # x: (b, c, f)
        batch, ctx, features = x.shape
        # q or k: (b, c, f) @ (f, h) = (b, c, h) where h(head_size) = f / n_head
        q = self.query(x)
        k = self.key(x)
        # calc attention score, w: (b, c, c)
        w = q @ k.transpose(-2, -1) * q.shape[-1]**-0.5
        w = w.masked_fill(self.tril[:ctx, :ctx] == 0, float('-inf'))
        w = torch.nn.functional.softmax(w, dim=-1)
        w = self.dropout(w)
        # cal weighted value, v: (b, c, h)
        v = self.value(x)
        # (b, c, c) @ (b, c, h) = (b, c ,h)
        rslt = w @ v
        return rslt

# params: 4 * n_embedding ^ 2 (Q, K, V, projection)
class MaskedMultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, context_size, n_embedding, dropout_p):
        super().__init__()
        head_size = n_embedding // n_head
        self.heads = torch.nn.ModuleList([MaskedSingleHeadAttention(head_size, context_size, n_embedding, dropout_p) for _ in range(n_head)])
        self.projection = torch.nn.Linear(n_embedding, n_embedding)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        # (b, c ,h) --cat--> (b, c, f)
        rslt = torch.cat([head(x) for head in self.heads], dim=-1)
        rslt = self.dropout(self.projection(rslt))
        return rslt

# params: 2 * 4 * n_embedding ^ 2
class FeedFoward(torch.nn.Module):
    def __init__(self, n_embedding, dropout_p):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(n_embedding, n_embedding * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(n_embedding * 4, n_embedding),
            torch.nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.seq(x)

class TransformerUnit(torch.nn.Module):
    def __init__(self, n_head, context_size, n_embedding, dropout_p):
        super().__init__()
        self.mha = MaskedMultiHeadAttention(n_head, context_size, n_embedding, dropout_p)
        self.ff = FeedFoward(n_embedding, dropout_p)
        self.mha_ln = torch.nn.LayerNorm(n_embedding)
        self.ff_ln = torch.nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.mha(self.mha_ln(x))
        x = x + self.ff(self.ff_ln(x))
        return x

# params: vocab_size * n_embedding * 2 + context_size * n_embedding
class NaiveLangModel(torch.nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, context_size, n_embedding, dropout_p, use_tie):
        super().__init__()
        # params: vocab_size * n_embedding
        self.token_embed = torch.nn.Embedding(vocab_size, n_embedding)
        # params: context_size * n_embedding
        self.position_embed = torch.nn.Embedding(context_size, n_embedding)
        self.units = torch.nn.Sequential(*[TransformerUnit(n_head, context_size, n_embedding, dropout_p) for _ in range(n_layer)])
        self.ln = torch.nn.LayerNorm(n_embedding)
        if use_tie:
            self.lm_head = torch.nn.Linear(n_embedding, vocab_size, bias=False)
            self.lm_head.weight = self.token_embed.weight
        else:
            # params: vocab_size * n_embedding
            self.lm_head = torch.nn.Linear(n_embedding, vocab_size)
        
        self.context_size = context_size
        self.use_tie = use_tie

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, labels=None):
        batch, ctx = inputs.shape
        # t_embed: (b, c, f); p_embed: (c,f)
        t_embed = self.token_embed(inputs)
        p_embed = self.position_embed(torch.arange(ctx, device=device))
        # x: (b, c, f)
        x = t_embed + p_embed
        x = self.units(x)
        x = self.ln(x)
        # logits: (b, c, v) 
        logits = self.lm_head(x)

        if labels is None:
            return logits, None

        batch, ctx, features = logits.shape
        predicts = logits.view(batch*ctx, features)
        targets = labels.view(batch*ctx)
        return logits, torch.nn.functional.cross_entropy(predicts, targets)

    def generate(self, inputs, max_gen):
        for _ in range(max_gen):
            inputs_last_window = inputs[:, -self.context_size:]
            logits, loss = self(inputs_last_window)
            logits = logits[:, -1, :]
            pred_next = torch.multinomial(torch.nn.functional.softmax(logits, dim=1), num_samples=1)
            inputs = torch.cat((inputs, pred_next), dim=1)
        return inputs

def train_model(learning_rate, batch_size, steps, eval_interval, n_eval, weight_decay, need_compile):
    @torch.no_grad()
    def calc_loss(n_eval, batch_size):
        rslt = {}
        model.eval()
        for c in ['train', 'eval']:
            losses = torch.zeros(n_eval)
            for i in range(n_eval):
                x, y = dc.collate_data(c, batch_size, model.context_size)
                # Acceleration
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                #_, loss = model(x, y)
                losses[i] = loss.item()
            rslt[c] = losses.mean()
        model.train()
        return rslt

    def prepare_optimizer(learning_rate, weight_decay, need_compile):
        decay = set()
        no_decay = set()
        
        for name, module in model.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{pname}" if name else pname
                if pname.endswith("bias") or "ln" in name or "ln" in pname or "embed" in name or "lm_head" in name:
                    no_decay.add(full_name)
                else:
                    decay.add(full_name)

        # print("[decay]: ", decay)
        # print("[no_decay]: ", no_decay)

        if model.use_tie:
            if need_compile:
                no_decay.remove("_orig_mod.lm_head.weight")
            else:
                no_decay.remove("lm_head.weight")

        param_dict = {pn: p for pn, p in model.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        fused_exist = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        fused = fused_exist and device_type == "cuda"
        print(f"[INFO] config AdamW with using fused : {fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
        return optimizer

    def update_lr(step):
        max_lr = 6e-4
        min_lr = max_lr * 0.1
        warmup_steps = 100
        max_steps = 1000

        if step < warmup_steps:
            return max_lr * (step+1) / warmup_steps
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 1 -> 0
        return min_lr + coeff * (max_lr - min_lr)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Apply weight decay
    optimizer = prepare_optimizer(learning_rate=learning_rate, weight_decay=weight_decay, need_compile=need_compile)

    for step in range(steps):
        if step % eval_interval == 0 or step == steps - 1:
            losses = calc_loss(n_eval, batch_size)
            print(f"[step {step:4d}] train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}")

        t0 = time.time()
        x, y = dc.collate_data('train', batch_size, model.context_size)
        optimizer.zero_grad(set_to_none=True)
        # Acceleration
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        #_, loss = model(x, y)
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = update_lr(step)
        for param_grp in optimizer.param_groups:
            param_grp['lr'] = lr

        optimizer.step()
        if "cuda" in device:
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        throughput = (batch_size * model.context_size) / dt
        if step % eval_interval == 0 or step == steps - 1:
            print(f"    elapsed {dt*1000:.2f}ms {throughput:.2f} tok/s norm: {norm:.4f} lr: {lr:.4e}")

dc = DataCollator('./Tolkien.txt', 0.9)

print("[INFO] Read in corpora:", len(dc.content))
print("[INFO] Training set tokens:", len(dc.train_data))
print("[INFO] Vocab size:", dc.n_vocab)

# Hyperparam
#vocab_size=dc.n_vocab
# Acceleration
vocab_size=50304
n_layer = 4
#n_head = 4
n_head = 6
# n_embedding = 256
n_embedding = 384
dropout_p = 0.2
context_size=256 # context length for prediction

steps = 2000
eval_interval = 100 # evaluate every N steps
# batch_size = 128
batch_size = 64
n_eval = 100       # evaluate n_eval times then calculate the mean
lr = 3e-4
#lr = 1e-3
weight_decay = 0.01
need_compile = True  # sudo apt install python3.9-dev

print(f"[INFO] 1 epoch == {len(dc.train_data) // (batch_size * context_size)} batches")

# params: vocab_size * n_embedding * 2 + context_size * n_embedding + 12 * n_embedding ^ 2
model = NaiveLangModel(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, context_size=context_size, n_embedding=n_embedding, dropout_p=dropout_p, use_tie=True)
model = model.to(device)
# Acceleration
if need_compile:
    model = torch.compile(model)
print("[INFO] Model params:", sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# model.load_state_dict(torch.load("model.pth", weights_only=True))

train_model(learning_rate=lr, batch_size=batch_size, steps=steps, eval_interval=eval_interval, n_eval=n_eval, weight_decay=weight_decay, need_compile=need_compile)

prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
print(dc.fn_decode(model.generate(prompt, max_gen=2000)[0].tolist()))

torch.save(model.state_dict(), "model.pth")
print("[INFO] Model saved.")
