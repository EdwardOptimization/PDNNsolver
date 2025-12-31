import os
import math
import time
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# ... (配置参数保持不变) ...
BATCH_SIZE = 64
BLOCK_SIZE = 128
MAX_ITERS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

class PID_Prodigy(torch.optim.Optimizer):
    """
    PID-Prodigy: 引入 PID 控制理论的自适应优化器。
    
    核心思想:
    不再做简单的"涨/缩"开关，而是维护一个 D 的'速度' (Velocity)。
    这个速度由梯度方向的历史累积决定 (Integral/Momentum)。
    
    效果:
    1. 抗噪性: 偶尔的一个坏梯度不会打断整体的增长趋势 (利用惯性)。
    2. 智能刹车: 如果连续出现震荡，速度会平滑减小直至变为负值，实现软着陆。
    """
    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), 
                 weight_decay=0.0, d0=1e-3, 
                 pid_beta=0.9,    # 速度的动量 (I项平滑度)
                 pid_gain=0.02,   # D 的变化幅度 (每次最多变 2%)
                 brake_ratio=2.0  # 刹车灵敏度 (刹车比油门重多少倍)
                 ): 
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, 
                        d=d0, d0=d0, 
                        pid_beta=pid_beta, pid_gain=pid_gain, brake_ratio=brake_ratio)
        super(PID_Prodigy, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        group0 = self.param_groups[0]
        if 'step' not in group0: group0['step'] = 0
        group0['step'] += 1
        k = group0['step']
        
        d_current = group0['d']
        d_max_observed = d_current
        
        moving_away_signal = 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]

                if len(state) == 0:
                    state['p0'] = p.clone().detach()
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # 初始化 PID 速度状态
                    group0['d_velocity'] = 0.0 

                disp = p - state['p0']
                dist = disp.norm().item()
                if dist > d_max_observed:
                    d_max_observed = dist
                
                # 信号计算：<grad, disp>
                moving_away_signal += torch.sum(p.grad * disp).item()

        # ====================================================
        # [PID Control Logic] 智能动态收缩
        # ====================================================
        
        # 1. 计算即时推力 (Force)
        # 如果我们在远离起点 (signal < 0)，推力为 +1 (油门)
        # 如果我们在往回走 (signal > 0)，推力为 -brake_ratio (刹车)
        current_force = 1.0
        if moving_away_signal > 0:
            current_force = -group0['brake_ratio']
        
        # 2. 更新 PID 速度 (I-Term / Momentum)
        # velocity = beta * velocity + (1-beta) * force
        # 这相当于对 Force 做了一个低通滤波，过滤掉瞬时噪声
        if 'd_velocity' not in group0: group0['d_velocity'] = 0.0
        
        beta = group0['pid_beta']
        group0['d_velocity'] = beta * group0['d_velocity'] + (1 - beta) * current_force
        
        # 3. 应用变化
        # D_new = D_old * (1 + gain * velocity)
        # 限制 velocity 的生效范围，防止 D 变化太剧烈
        gain = group0['pid_gain']
        change_rate = gain * group0['d_velocity']
        
        # 如果在观测范围内 (d_max_observed > d_current)，我们倾向于增长
        # 但完全由 velocity 说了算。如果 velocity 是负的，即使观测到了更远距离，我们也不涨反降（防止虚假繁荣）
        
        # 更新 D
        group0['d'] = group0['d'] * (1.0 + change_rate)
        
        # 兜底与同步
        # 如果观测到的实际距离实在太大了，我们可以适度快进（Trust Region Expansion）
        # 但要服从 velocity 的指挥，如果 velocity < 0，禁止快进
        if group0['d_velocity'] > 0 and d_max_observed > group0['d']:
             group0['d'] = min(d_max_observed, group0['d'])

        group0['d'] = max(group0['d'], group0['d0'])

        # --- Phase 2: Update (Standard AdamW Step) ---
        step_size_factor = 1.0 / math.sqrt(k + 100)

        active_lr = 0.0
        for group in self.param_groups:
            lr = group['lr']
            d = group0['d']
            beta1, beta2 = group['betas']
            
            adaptive_lr = lr * d * step_size_factor
            active_lr = adaptive_lr

            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if group['weight_decay'] > 0:
                    p.mul_(1 - adaptive_lr * group['weight_decay'])
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(1e-8)
                
                p.addcdiv_(exp_avg, denom, value=-adaptive_lr)

        return active_lr

# ==============================================================================
# 辅助代码 (保持不变)
# ==============================================================================
def prepare_data():
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

train_data, val_data, vocab_size = prepare_data()

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=N_EMBD, nhead=N_HEAD, dim_feedforward=4*N_EMBD, 
                                       dropout=DROPOUT, batch_first=True, norm_first=True) 
            for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

def run_training(opt_mode):
    torch.manual_seed(1337)
    model = GPT().to(DEVICE)
    
    if opt_mode == "baseline":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS)
        name = "Baseline (AdamW + Cosine)"
    else:
        # 启用 PID 逻辑
        optimizer = PID_Prodigy(model.parameters(), lr=1.0, weight_decay=1e-2, 
                                pid_beta=0.9, brake_ratio=2.0)
        scheduler = None
        name = "PID-Prodigy (Smart)"

    print(f"\n[{name}] 开始训练...")
    start_time = time.time()
    history = []

    for iter in range(MAX_ITERS):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        curr_lr = 0.0
        if scheduler:
            optimizer.step()
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
        else:
            curr_lr = optimizer.step()

        if iter % 100 == 0 or iter == MAX_ITERS - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch('val')
                _, vloss = model(vx, vy)
            model.train()
            print(f"Iter {iter:4d} | Train: {loss.item():.4f} | Val: {vloss.item():.4f} | LR: {curr_lr:.6f}")
            history.append((iter, vloss.item(), curr_lr))
    
    print(f"[{name}] 耗时: {time.time()-start_time:.2f}s")
    return history

if __name__ == '__main__':
    print("="*60)
    print(f"Benchmark: Baseline vs. PID-Prodigy")
    print("="*60)
    hist_base = run_training("baseline")
    hist_pid = run_training("pid")

    print("\n" + "="*60)
    print("FINAL RESULTS (Validation Loss)")
    print("="*60)
    print(f"{'Iter':<6} | {'Baseline':<22} | {'PID-Prodigy':<18}")
    print("-" * 60)
    for i in range(len(hist_base)):
        v_pid = hist_pid[i][1]
        v_base = hist_base[i][1]
        marker = " <--" if v_pid <= v_base else ""
        print(f"{hist_base[i][0]:<6} | {v_base:.4f} (LR={hist_base[i][2]:.5f})   | {v_pid:.4f} (LR={hist_pid[i][2]:.5f}){marker}")