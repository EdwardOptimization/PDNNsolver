"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_name='adamw', schedule_args=None):
        """
        Modified to support switching between 'adamw' and 'pid_prodigy'.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
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
        
        optimizer = None
        scheduler = None  # 初始化 scheduler 为 None
        # Create Optimizer based on optimizer_name
        import inspect

        if optimizer_name == 'adamw':
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            
            print(f"using fused AdamW: {use_fused}")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            
        # === 分支 A: 行业标准 (AdamW + Cosine) ===
        elif optimizer_name == 'adamw_cosine':
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
            
            # [修正] 从 schedule_args 字典中获取参数，而不是 self.config
            if schedule_args is None:
                raise ValueError("optimizer 'adamw_cosine' requires schedule_args to be passed")
                
            warmup_iters = schedule_args['warmup_iters']
            lr_decay_iters = schedule_args['lr_decay_iters']
            min_lr = schedule_args['min_lr']
            
            def get_lr(it):
                if it < warmup_iters:
                    return it / warmup_iters
                if it > lr_decay_iters:
                    return min_lr / learning_rate
                decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                return min_lr / learning_rate + coeff * (1 - min_lr / learning_rate)
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        elif optimizer_name == 'pid_prodigy':
            # Import our custom optimizer
            try:
                from pid_prodigy import PID_Prodigy
            except ImportError:
                raise ImportError("Could not import PID_Prodigy. Make sure pid_prodigy.py is in the same directory.")
            
            print(f"using custom PID-Prodigy optimizer (initial lr={learning_rate})")
            # Note: PID_Prodigy does not support 'fused' argument, and handles LR/Decay internally
            optimizer = PID_Prodigy(optim_groups, lr=learning_rate, betas=betas, weight_decay=weight_decay)
            
        elif optimizer_name == "pid_prodigy_restart":
            from pid_prodigy_restart import PID_Prodigy_Restart
            print(">>> Using PID-Prodigy-Restart (V41 Hybrid Ultimate) inside model.py")

            # [标准分组] 区分 Weights (需要 Decay) 和 Biases (不需要 Decay)
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            weight_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            bias_params = [p for n, p in param_dict.items() if p.dim() < 2]

            optim_groups = [
                {
                    'params': weight_params,
                    'weight_decay': weight_decay, # 只有 Weights 需要 Weight Decay
                },
                {
                    'params': bias_params,
                    'weight_decay': 0.0,          # Biases 绝对不 Weight Decay
                }
            ]

            # 实例化优化器
            optimizer = PID_Prodigy_Restart(
                optim_groups,
                lr=learning_rate,           # 必须是 1.0
                betas=betas,
                weight_decay=weight_decay,  # 默认值
                d0=1e-5,                    # 标准起步距离
                pid_beta=0.9,
                # pid_gain=0.02,              # 标准增益
                base_brake_ratio=1.2,
                
                # [V35 特性] 动态阈值重启
                restart_threshold_init=0.5, # 起步宽容度
                min_restart_threshold=0.01, # 最终严厉度
                restart_patience=20,        # 给足耐心
                
                # [关键修改 1] 增大阻力系数
                # 原来是 2e-4 -> 改为 5e-4
                # 即使 SNR 较高，也会有轻微的自然衰减，防止 LR 一直顶天花板
                # base_d_decay=5e-4, 

                # [关键修改 2] 提高 SNR 目标 (让优化器更"挑剔")
                # 原来是 2.0 -> 改为 6.0
                # 这样 calculated_snr 会变小，decay term (1-snr) 会变大
                # 迫使 LR 在后期自然下降
                snr_target=6.0,    
                
                d_max=20.0,                 # [放宽] 让物理阻力接管上限，而不是硬卡
                max_lr=0.01,                # [熔断] 最后的安全防线
                
                # 高级参数
                cos_var_weight=10.0
            )

            # [防呆检查]
            if learning_rate != 1.0:
                print(f"\n{'='*80}")
                print(f"WARNING: Learning rate is {learning_rate}, but Prodigy expects 1.0!")
                print(f"Please run with --learning_rate=1.0")
                print(f"{'='*80}\n")
            
        elif optimizer_name == 'pid_prodigy_restart_old':
            from pid_prodigy_restart_0 import PID_Prodigy_Restart
            print(f"using HPR-LP style Restart Optimizer")
            # restart_threshold: -0.8 表示夹角大于 143 度时认为在剧烈掉头
            # restart_patience: 连续 20 次检测到掉头才重置，防止误杀
            optimizer = PID_Prodigy_Restart(optim_groups, lr=learning_rate, 
                                            restart_threshold=-0.8, 
                                            restart_patience=20)
            
        elif optimizer_name == 'muon':
            from muon import Muon
            print("using Original Muon optimizer")
            # Muon 对 weight_decay 不敏感 (2D 部分)，主要是 1D 部分用 AdamW 需要
            # 注意：Muon 的 LR 必须设置得很大 (0.02 - 0.05)
            optimizer = Muon(optim_groups, lr=learning_rate, momentum=0.95)

        elif optimizer_name == 'pid_muon':
            from pid_muon import PID_Muon
            print("using PID-Muon optimizer")
            # Muon 需要特殊的 weight_decay 策略，通常 0.01 即可
            optimizer = PID_Muon(
                optim_groups, 
                lr=0.05,               # Base LR: 保持大
                weight_decay=0.01,
                
                # === PID 参数调整 ===
                d0=5e-1,               # 1. 拒绝起步肉：初始值加大 100 倍
                pid_beta=0.8,          # 2. 提高响应速度：减少 PID 自身的惯性
                pid_gain=0.01,         # 3. 稳油门：防止 D 暴涨
                brake_ratio=5.0,       # 4. 强刹车：一旦方向不对，立即重踩
                
                # === Restart 参数 ===
                restart_threshold=-0.8,
                restart_patience=5     # 容忍度降低，发现不对劲赶紧重开
            )
            
        elif optimizer_name == 'muon_fixed_restart':
            from muon_restart import Muon_Fixed_Restart
            print("using Muon + Restart (Fixed LR)")
            # Muon 需要较大的 LR，推荐 0.05
            # restart_threshold: -0.5 意味着夹角超过 120 度就预警
            optimizer = Muon_Fixed_Restart(optim_groups, lr=0.05, restart_threshold=-0.5)
        
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_name}")

        return optimizer, scheduler

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
