import math
import torch
import torch.optim

def newton_schulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz 迭代: 将矩阵 G 正交化 (Approximate Inverse Square Root).
    Muon 的核心魔法。
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # 归一化光谱范数 (Spectral Norm)
    X /= (X.norm(p=2, dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

class PID_Muon(torch.optim.Optimizer):
    """
    PID-Muon: Parameter-Free Momentum Orthogonal Optimizer.
    
    Composition:
    - Direction: Muon (Newton-Schulz Orthogonalization) for 2D tensors.
    - Direction: AdamW for 1D tensors (embeddings/biases).
    - Step Size: PID-Prodigy (Adaptive D-Estimation) for ALL tensors.
    - Safety: HPR-LP Restart Strategy.
    """
    def __init__(self, params, lr=0.02, # Muon 的 base lr 通常较大，0.02 是经验值
                 weight_decay=0.01,
                 betas=(0.95, 0.95),    # Muon 通常喜欢 0.95 的 momentum
                 ns_steps=5,            # Newton-Schulz 迭代次数
                 d0=1e-5, pid_beta=0.9, pid_gain=0.02, brake_ratio=2.0,
                 restart_threshold=-0.8, restart_patience=10):
        
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas,
                        ns_steps=ns_steps,
                        d=d0, d0=d0, pid_beta=pid_beta, pid_gain=pid_gain, 
                        brake_ratio=brake_ratio,
                        restart_threshold=restart_threshold,
                        restart_patience=restart_patience)
        super(PID_Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        for group in self.param_groups:
            if 'step' not in group: group['step'] = 0
            group['step'] += 1
            k = group['step']

            # ==========================================
            # 1. PID-Prodigy: 全局几何计算 & 步长估计
            # ==========================================
            # 我们使用“原始梯度”来计算距离 D，因为这代表了 Loss Landscape 的真实几何
            
            dot_product_sum = 0.0
            grad_norm_sq_sum = 0.0
            disp_norm_sq_sum = 0.0
            d_max_observed = group['d']
            
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                
                # Lazy Init
                if len(state) == 0:
                    state['p0'] = p.clone().detach() # Anchor point
                    state['momentum'] = torch.zeros_like(p) # For Muon/Adam
                    # Adam second moment (only for 1D params)
                    if p.ndim < 2:
                        state['v'] = torch.zeros_like(p)
                
                if 'd_velocity' not in group: group['d_velocity'] = 0.0
                if 'stagnation_counter' not in group: group['stagnation_counter'] = 0

                disp = p.sub(state['p0'])
                grad = p.grad
                
                dot_product_sum += torch.sum(grad * disp).item()
                grad_norm_sq_sum += torch.sum(grad * grad).item()
                disp_norm_sq_sum += torch.sum(disp * disp).item()
                
                dist = disp.norm().item()
                if dist > d_max_observed: d_max_observed = dist

            # --- Restart Logic ---
            grad_norm = math.sqrt(grad_norm_sq_sum) + 1e-8
            disp_norm = math.sqrt(disp_norm_sq_sum) + 1e-8
            cosine_sim = dot_product_sum / (grad_norm * disp_norm)
            
            do_restart = False
            if cosine_sim < group['restart_threshold']:
                group['stagnation_counter'] += 1
            else:
                group['stagnation_counter'] = max(0, group['stagnation_counter'] - 1)
                
            if group['stagnation_counter'] > group['restart_patience']:
                do_restart = True
                print(f"[PID-Muon] Restart triggered at step {k} (Cos: {cosine_sim:.4f})")
                group['d'] = disp_norm
                group['d_velocity'] = 0.0
                group['stagnation_counter'] = 0
                
                # Reset Momentum for all params
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['momentum'].zero_()
                    if p.ndim < 2: self.state[p]['v'].zero_()
                return 0.0 # Skip update

            # --- PID Control ---
            current_force = 1.0 if dot_product_sum < 0 else -group['brake_ratio']
            beta = group['pid_beta']
            group['d_velocity'] = beta * group['d_velocity'] + (1 - beta) * current_force
            change_rate = group['pid_gain'] * group['d_velocity']
            group['d'] = max(group['d'] * (1.0 + change_rate), group['d0'])
            if group['d_velocity'] > 0 and d_max_observed > group['d']:
                 group['d'] = min(d_max_observed, group['d'])

            # ==========================================
            # 2. Parameter Update (Muon for 2D, AdamW for 1D)
            # ==========================================
            
            # 计算 Prodigy 自适应缩放因子
            # 注意：Muon 的 update 通常已经是归一化的 (RMS=1 或 Spectral=1)
            # 所以这里的 step_size 直接代表了“在这个正交方向上走多远”
            adaptive_step_size = group['lr'] * group['d'] / (k**0.5)
            
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                # --- A. Weight Decay ---
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - adaptive_step_size * group['weight_decay'])
                
                # --- B. Momentum Update ---
                buf = state['momentum']
                buf.mul_(beta1).add_(grad, alpha=1) # Muon usually uses simple momentum
                
                # --- C. Update Direction ---
                if p.ndim >= 2:
                    # === Muon Path (2D+ Tensors) ===
                    # 正交化动量 buffer
                    ortho_update = newton_schulz5(buf, steps=group['ns_steps'])
                    
                    # 尺度缩放 (Muon scaling trick)
                    # Muon 论文建议将 update 缩放到 RMS=1 或者类似
                    # 这里 newton_schulz5 已经做了 Spectral Norm 归一化
                    # 我们直接应用 Prodigy 算出来的 step_size
                    
                    # 为了数值稳定性，最好转回 p 的精度
                    update = ortho_update.to(p.dtype)
                    
                    # Update
                    p.data.add_(update, alpha=-adaptive_step_size)
                    
                else:
                    # === AdamW Path (1D Tensors) ===
                    # 对于 vector 类型的参数，Muon 没法做正交化，退化为 AdamW
                    v = state['v']
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = v.sqrt().add_(1e-8)
                    
                    # Bias correction
                    bc2 = 1 - beta2 ** k
                    step_size = adaptive_step_size * math.sqrt(bc2) # approximate bc1 with momentum
                    
                    p.data.addcdiv_(buf, denom, value=-step_size)

        return adaptive_step_size