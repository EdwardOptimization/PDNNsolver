"""
PD-AdamW vs. Standard AdamW Benchmark
-------------------------------------
对比实验：
1. Baseline: PyTorch 原生 AdamW + Cosine Annealing LR (工业标准)
2. PD-AdamW: 自适应步长优化器 (无预设 Schedule)

观察重点：
在不告诉 PD-AdamW 总步数的情况下，它能否自动拟合出类似 Cosine 的下降曲线？
"""

import math
import time
import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# ================= 配置参数 =================
batch_size = 64
block_size = 128
max_iters = 1000       # 两个选手都跑 1000 步
learning_rate = 1e-3   # 初始 LR
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

print(f"Benchmark running on: {device}")

# ================= 核心算法: PD-AdamW v2 =================
class PD_AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, alpha=0.005, oscillation_penalty=2.0):
        # 注意: alpha 设为 0.005 (比 0.01 保守，比 0.002 激进)
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, alpha=alpha,
                        oscillation_penalty=oscillation_penalty)
        super(PD_AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        # --- A. 全局统计量 ---
        total_grad_diff_sq = 0.0
        total_param_diff_sq = 0.0
        total_dot_product = 0.0
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
                    total_dot_product += torch.sum(p.grad * state['prev_grad']).item()

        # --- B. 估算 Lipschitz & 震荡检测 ---
        suggested_lr = None
        if has_prev_state and total_param_diff_sq > 1e-12:
            L_k = math.sqrt(total_grad_diff_sq / total_param_diff_sq)
            
            # 震荡惩罚: 如果梯度反向，强制放大曲率估计
            penalty = 1.0
            if total_dot_product < 0:
                penalty = group['oscillation_penalty']
            
            # 计算建议步长
            alpha = group['alpha']
            suggested_lr = alpha / (L_k * penalty + 1e-8)

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

                # 动态 LR 更新
                current_lr = state['adaptive_lr']
                if suggested_lr is not None:
                    if suggested_lr > current_lr:
                        new_lr = current_lr * 1.05 # 加速
                    else:
                        new_lr = 0.9 * current_lr + 0.1 * suggested_lr # 减速/刹车
                    new_lr = max(min(new_lr, 0.05), 1e-6)
                    state['adaptive_lr'] = new_lr
                
                step_size = state['adaptive_lr']
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = step_size * (math.sqrt(bias_correction2) / bias_correction1)
                
                # 更新前保存状态
                current_param_val = p.clone()
                state['prev_grad'].copy_(grad)
                state['prev_param'].copy_(current_param_val)

                p.addcdiv_(exp_avg, denom, value=-step_size)
                active_lr = state['adaptive_lr']
                
        return active_lr

# ================= 辅助代码: 数据与模型 =================
# (保持不变，省略部分细节以节省篇幅，功能与之前相同)
def get_data():
    if not os.path.exists('input.txt'):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open('input.txt', 'w', encoding='utf-8') as f: f.write(requests.get(url).text)
    with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    return data[:n], data[n:], len(chars)

train_data, val_data, vocab_size = get_data()

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 简化的模型类
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd, 
                                       dropout=dropout, batch_first=True, norm_first=True) 
            for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# ================= 训练函数 =================
def run_training(optimizer_name):
    # 每次重新初始化模型，保证公平
    torch.manual_seed(1337)
    model = GPT().to(device)
    
    if optimizer_name == "Baseline (AdamW+Cosine)":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        # 这是一个强力的辅助：余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    else:
        # PD-AdamW 自带导航，不需要 Scheduler
        optimizer = PD_AdamW(model.parameters(), lr=learning_rate, alpha=0.005, oscillation_penalty=2.0)
        scheduler = None

    print(f"\n🚀 开始训练: {optimizer_name}")
    history = []
    start_t = time.time()
    
    for iter in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        curr_lr = 0.0
        if optimizer_name == "PD-AdamW (AutoLR)":
            curr_lr = optimizer.step() # PD-AdamW 返回 LR
        else:
            optimizer.step()
            scheduler.step() # 手动更新 LR
            curr_lr = scheduler.get_last_lr()[0]

        if iter % 100 == 0 or iter == max_iters - 1:
            # 快速验证一下 Val Loss
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch('val')
                _, vloss = model(vx, vy)
            model.train()
            
            print(f"Iter {iter:4d} | Train: {loss.item():.4f} | Val: {vloss.item():.4f} | LR: {curr_lr:.6f}")
            history.append((iter, loss.item(), vloss.item(), curr_lr))
            
    print(f"耗时: {time.time()-start_time:.2f}s")
    return history

# ================= 主程序 =================
if __name__ == '__main__':
    start_time = time.time()
    
    # 1. 运行基准 (Baseline)
    hist_base = run_training("Baseline (AdamW+Cosine)")
    
    # 2. 运行实验组 (PD-AdamW)
    hist_pd = run_training("PD-AdamW (AutoLR)")

    # 3. 打印对比结果
    print("\n" + "="*50)
    print("FINAL RESULT COMPARISON (Val Loss)")
    print("="*50)
    print(f"{'Iter':<6} | {'Baseline':<10} | {'PD-AdamW':<10} | {'Gap':<10}")
    print("-" * 50)
    
    for i in range(len(hist_base)):
        step, _, v_base, lr_base = hist_base[i]
        _, _, v_pd, lr_pd = hist_pd[i]
        
        diff = v_base - v_pd
        marker = "🏆 PD" if v_pd < v_base else "  Base"
        
        print(f"{step:<6} | {v_base:.4f}     | {v_pd:.4f}     | {marker}")

    print("="*50)
    print("分析:")
    print("1. Baseline 使用了完美的 Cosine Schedule (先热身再衰减)。")
    print("2. PD-AdamW 全程自适应 (不知道总步数)。")
    print("如果 PD-AdamW 的 Loss 能接近甚至低于 Baseline，说明自适应机制成功了。")