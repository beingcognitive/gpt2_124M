import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

### 아래와 같은 약 100줄의 코드로 기존에 약 2천줄에 달하는 코드를 간소화시켰다.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
class CausalSelfAttention(nn.Module): ### Andrej Karpathy의 다른 강의에서 보였던 Head를 Multi-Head Attention으로 구현한 것을, pytorch에서의 연산 효율성을 위해 하나의 모듈로 재구성한 것일 뿐임. 본질적으로 같다.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the transformer
        qkv = self.c_attn(x)
        q, k, v =qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) ### 최적화 #4
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) ### q,k간에 상호 얼마나 참조하는지 파악
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) ### autoregressive mask를 적용하여, 과거의 데이터만 보게 만들고,
        att = F.softmax(att, dim=-1) ### SoftMax로 1로 normalize시키고
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) ### Weighted Sum 을 계산하고,
        

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side ##Concatenation 연산에 해당
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh') ### 초창기 tensorflow에서 erf속도가 너무 느려서 tanh로 GELU를 approximation하는 것을 만들었음. 지금은 gelu자체에 속도 이슈가 없어서 approximation을 사용할 이유가 없으나, 최대한 GPT-2에 근접하게 만드는 것이 오늘 강의의 목표이므로, approximation을 사용했음. ### 그리고, relu대신 gelu를 사용하면 0 근처에서의 학습 데이터를 계속해서 사용할 수 있는 강점이 있기 때문. 최근에는 gelu 이후 다른 함수들을 많이 쓰고 있지만, 같은 이유로 오늘은 gelu를 사용한다.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x): # Residual한 Path를 깔끔하게 내려주는 것이 0.기본문서보다 더 동작을 잘 하게 만들 수 있다.
        x = x + self.attn(self.ln_1(x)) ## 어텐션은 커뮤니케이션 연산에 해당. 1024개 토큰끼리 상호 작용이 활발하게 일어나기 때문에, aggregation,pooling,weighted sum,reduce라고 볼 수 있고,
        x = x + self.mlp(self.ln_2(x)) ## MLP는 모든 토큰이 개별적으로 연산되고, 토큰 간의 연산은 없음. 따라서, 윗 줄의 attn은 REDUCE, 이 줄의 mlp는 MAP에 해당한다고도 볼 수 있다. 즉, Transformer는 Map Reduce의 반복이라고도 볼 수 있음.
        return x

@dataclass
class GPTConfig: ### huggingface에 올라온 gpt2 124M모델과 hyperparameter를 맞췄음.
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Figure 1.에서는 Output Embedding이라고 적혀있지만, 그것이 여기서는 Token Embedding (wte)에 해당
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd), #Layer normalization 은 각 블록의 뒤로 옮겨졌다. #Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a pre-activation residual network
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #an additional layer normalization was added after the final self attention block.

        ## weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 ##
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): ## 인풋은 항상 인덱스인데, 'token'의 인덱스들임. BxT사이즈
        # idx is of shape (B, T) ## T는 타임, T개의 token이 존재함. idx는 항상 BxT이다!
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) ## Cross_Entropy는 2차원을 인풋으로 받지 못하기 때문에 BxT를 flatten시키는 작업이 필요했다. (B*T, voab_size)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no. 
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

#####-----------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') # GPT-2 토크나이저 사용
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0 # 데이터를 다 사용했으면, 다시 0으로 돌아와서 다음 에폭을 시작하자.
        return x, y

#------------------
# attempt to autodetect the decvice
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

### REMOVE THIS
# device = "cpu"### REMOVE THIS
### REMOVE THIS

print(f"using device: {device}")

##### Code Reproducibility를 위해서 시드 고정
torch.manual_seed(1337) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
#####

total_batch_size = 524288 # 2**19 ~0.5M, in number of tokens
B = 4
T = 32
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)
### 최적화 #1. 
torch.set_float32_matmul_precision('high') ### highest 에서 fp32를 사용하는 것 대신, TF32를 사용함으로써, Precision을 아주 살짝 포기하고, 전체 연산 속도를 높인다.

# get logits
# model = GPT(GPTConfig())
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) ### 이 한줄로 추가 최적화 #3. gcc처럼 컴파일하여 사용하는 셈.
# logits,loss = model(x, y)
# print(logits.shape) #torch.Size([4, 32, 50257]) 아웃풋 로짓의 크기는 4x32에 대한 50257 토큰개수만큼-. 각 위치 다음에 무엇이 오는가에 대한 로짓값이 됨.
# print(loss)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # Adam에 있는 버그를 수정한 AdamW를 사용한다. SGD보다 최적화 속도가 더 빠름
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() ## 항상 제로그레디언트로 시작해야 함
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()    

    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    #     logits, loss = model(x, y)
    #     # import code; code.interact(local=locals()) ### >>> logits.dtype ==> torch.bfloat16
    # import code; code.interact(local=locals()) ### >>> logits.dtype ==> torch.float32 자료형이 fp32임. 이를 TF32로 고쳐서 아주 약간만 precision을 희생시킨다면, 8배의 TFLOPS를 얻어낼 수 있다. 이건 공짜이기 때문에, Andrej가 가장 좋아하는 최적화 방법론 중 하나
    
    # logits, loss = model(x, y)
    # loss.backward()


    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) ## model이 아주 가끔씩 shock을 당하는 일을 막기 위함.
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
    optimizer.step() # 파라미터 업데이트
    # torch.cuda.synchronize() ### CUDA가 있을 때에 GPU와 CPU가 별도로 실행되는 것을 막기 위해, CPU가 GPU의 실행을 기다리는 역할.
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.0f} | tokens_processed: {tokens_processed}")

import sys; sys.exit(0) # 이 아래줄은 모두 실행하지 않음.

num_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B=5, T=8
torch.manual_seed(42)
# torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad(): ## Backward로 가지 않는다고 torch에게 알려주는 것. 그러면 캐싱 등 고생하지 않게 됨.
        logits = model(x)
        # take the logits at the last position  #약간의 낭비는 있는 셈
        logits = logits[:, -1, :]
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        #topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    # print(x[i])
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)