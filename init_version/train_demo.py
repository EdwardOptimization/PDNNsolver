"""
PD-AdamW Demo for NanoGPT
-------------------------
这是一个单文件演示脚本，展示如何将 PDHG (数值优化) 中的 Lipschitz 自适应步长机制
引入到 AdamW 优化器中，从而实现无需 Scheduler 的自动学习率调整。

运行环境要求:
pip install torch numpy requests
"""

import os
import math
import time
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# 1. 配置参数 (Configuration)
# ==============================================================================
# 为了演示方便，参数设置得很小，确保在 CPU 或低端显卡上也能秒级运行
batch_size = 64        # 每次训练多少个序列
block_size = 128       # 上下文窗口大小
max_iters = 1000       # 总训练步数
eval_interval = 100    # 每隔多少步评估一次 Loss
learning_rate = 1e-3   # 初始学习率 (会被 PD-AdamW 自动调整)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1          # 正则化

print(f"正在使用设备: {device}")

# ==============================================================================
# 2. 核心算法: PD-AdamW (Primal-Dual Inspired Adaptive AdamW)
# ==============================================================================
class PD_AdamW(torch.optim.Optimizer):
    """
    PD-AdamW v2: 加入了震荡抑制机制 (Oscillation Damping)。
    灵感来源: PDHG 的 Backtracking 思想。
    如果发现梯度方向反转 (Zig-Zag)，强制认为地形陡峭 (放大 L_k)，从而降低 LR。
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, alpha=0.5, oscillation_penalty=2.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, alpha=alpha,
                        oscillation_penalty=oscillation_penalty)
        super(PD_AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # --- A. 全局统计量计算 ---
        total_grad_diff_sq = 0.0
        total_param_diff_sq = 0.0
        total_dot_product = 0.0 # 用于检测梯度方向一致性
        has_prev_state = False
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                
                if 'prev_grad' in state and 'prev_param' in state:
                    has_prev_state = True
                    grad_diff = p.grad - state['prev_grad']
                    param_diff = p - state['prev_param']
                    
                    total_grad_diff_sq += grad_diff.norm().item()**2
                    total_param_diff_sq += param_diff.norm().item()**2
                    
                    # 计算 <g_t, g_{t-1}>
                    # 如果结果为负，说明大部分参数的梯度反向了
                    total_dot_product += torch.sum(p.grad * state['prev_grad']).item()

        # --- B. 估算 Lipschitz 并应用震荡惩罚 ---
        suggested_lr = None
        
        if has_prev_state and total_param_diff_sq > 1e-12:
            L_k = math.sqrt(total_grad_diff_sq / total_param_diff_sq)
            
            # [新增] 震荡检测
            # 如果点积为负，说明我们在震荡。
            # 此时我们人为地“放大”估算的曲率 L_k，迫使 LR 减小。
            penalty = 1.0
            if total_dot_product < 0:
                penalty = self.param_groups[0]['oscillation_penalty']
            
            # 应用惩罚后的 L_k
            adjusted_L_k = L_k * penalty
            
            alpha = self.param_groups[0]['alpha']
            suggested_lr = alpha / (adjusted_L_k + 1e-6)

        # --- C. 更新参数 ---
        active_lr = 0.0

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)
                    state['prev_param'] = torch.zeros_like(p)
                    state['adaptive_lr'] = group['lr']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                p.mul_(1 - group['lr'] * group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # --- D. 动态步长更新 ---
                current_lr = state['adaptive_lr']
                
                if suggested_lr is not None:
                    # 如果检测到震荡 (suggested_lr 变小了)，我们会快速下降
                    if suggested_lr < current_lr:
                        new_lr = 0.9 * current_lr + 0.1 * suggested_lr
                    else:
                        # 正常加速
                        new_lr = current_lr * 1.05
                    
                    # 仍然保留硬截断，但在震荡惩罚下，应该很难一直触顶了
                    new_lr = max(min(new_lr, 0.05), 1e-6)
                    state['adaptive_lr'] = new_lr
                
                step_size = state['adaptive_lr']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = step_size * (math.sqrt(bias_correction2) / bias_correction1)
                
                # 保存状态
                current_param_val = p.clone()
                state['prev_grad'].copy_(grad)
                state['prev_param'].copy_(current_param_val)

                # Update
                p.addcdiv_(exp_avg, denom, value=-step_size)

                active_lr = state['adaptive_lr']
                
        return active_lr

# ==============================================================================
# 3. 数据处理 (Data Loading)
# ==============================================================================
def prepare_data():
    file_path = 'input.txt'
    if not os.path.exists(file_path):
        print("未找到数据集，正在下载 tiny_shakespeare.txt ...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        r = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(r.text)
        print("下载完成。")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# 准备数据
text = prepare_data()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#以此切分训练集和验证集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ==============================================================================
# 4. 模型定义 (NanoGPT Architecture)
# ==============================================================================
class Head(nn.Module):
    """ 单个 Self-Attention Head """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ 多个 Heads 并行 """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 简单的 MLP """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: Communication followed by Computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# ==============================================================================
# 5. 训练循环 (Training Loop)
# ==============================================================================

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    model = GPTLanguageModel().to(device)
    
    # 打印模型参数量
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    
    # 使用自定义的 PD-AdamW 优化器
    # 注意：我们这里给一个初始 lr，但它会随即被算法覆盖和调整
    optimizer = PD_AdamW(model.parameters(), lr=1e-3, alpha=0.01) 
    
    start_time = time.time()
    
    print("-" * 65)
    print(f"{'Iter':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Auto LR':<12} | {'Time (s)':<8}")
    print("-" * 65)
    
    for iter in range(max_iters):
        
        # 每隔一段时间评估 loss
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            # 获取当前优化器的 adaptive_lr 用于显示 (如果还没开始跑，可能取不到，给个默认值)
            curr_lr = 0.0
            if hasattr(optimizer, 'param_groups'):
                 # 获取第一个参数的状态里的 lr
                for p in optimizer.param_groups[0]['params']:
                    if p in optimizer.state and 'adaptive_lr' in optimizer.state[p]:
                        curr_lr = optimizer.state[p]['adaptive_lr']
                        break
            if curr_lr == 0.0: curr_lr = learning_rate

            print(f"{iter:<8} | {losses['train']:.4f}       | {losses['val']:.4f}       | {curr_lr:.6f}       | {time.time()-start_time:.2f}")
    
        # 采样数据
        xb, yb = get_batch('train')
    
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Optimizer Step (这里包含了自动步长调整逻辑)
        optimizer.step()
    
    print("-" * 65)
    print("训练结束。生成一段文本测试效果:\n")
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=200)[0].tolist())
    print(generated_text)