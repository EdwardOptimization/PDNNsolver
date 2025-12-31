import math
import torch
import torch.optim

class PID_Prodigy(torch.optim.Optimizer):
    """
    PID-Prodigy (High Performance Version for GPT-2 Benchmarking)
    
    特点:
    1. Parameter-Free: 初始 LR 设为 1.0 即可。
    2. PID Controlled: 利用 PID 控制 D 的动态伸缩，防止震荡。
    3. Memory Optimized: 优化了状态存储。
    """
    def __init__(self, params, lr=1.0, 
                 betas=(0.9, 0.999), weight_decay=0.0, 
                 d0=1e-5,            # 初始距离，稍微给一点点，防止除零
                 pid_beta=0.9,       # I-term 惯性
                 pid_gain=0.01,      # 每次调节幅度 (保守一点: 1%)
                 brake_ratio=2.0,    # 刹车力度
                 eps=1e-8):
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        d=d0, d0=d0, 
                        pid_beta=pid_beta, pid_gain=pid_gain, 
                        brake_ratio=brake_ratio, eps=eps)
        super(PID_Prodigy, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        # 全局共享状态 (假设只有一个 param_group 或者大家步调一致)
        group0 = self.param_groups[0]
        if 'step' not in group0: group0['step'] = 0
        group0['step'] += 1
        k = group0['step']

        # --- Phase 1: 计算全局几何信号 ---
        d_current = group0['d']
        d_max_observed = d_current
        moving_away_signal = 0.0
        
        # 为了性能，尽量减少 Python 层面的开销
        # 如果是标准 PyTorch，这里很难完全消除循环，除非用 C++ 扩展
        # 但我们可以尽量精简循环体
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                
                state = self.state[p]
                
                # Lazy Initialization
                if len(state) == 0:
                    # 必须保存 p0 (Anchor Point) 用于计算位移
                    # 这是一个显存开销点：相当于模型参数量翻倍
                    # 对于 124M 模型，额外占用约 500MB 显存，完全可接受
                    state['p0'] = p.clone().detach()
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if 'd_velocity' not in group0: group0['d_velocity'] = 0.0

                # 计算 disp = p - p0
                # 注意：这里不创建新 Tensor，直接用临时计算
                disp = p.sub(state['p0']) 
                
                # 更新最大观测距离
                dist = disp.norm().item()
                if dist > d_max_observed: d_max_observed = dist
                
                # 累积信号 <grad, disp>
                # dot product
                moving_away_signal += torch.sum(p.grad * disp).item()

        # --- Phase 2: PID 控制 ---
        # 1. Force Calculation
        current_force = 1.0 if moving_away_signal < 0 else -group0['brake_ratio']
        
        # 2. Velocity Update (Momentum)
        beta = group0['pid_beta']
        group0['d_velocity'] = beta * group0['d_velocity'] + (1 - beta) * current_force
        
        # 3. Apply Change
        # 只有当观测到更大的距离时，且 velocity 允许时，才增长
        # 否则主要听从 velocity 的收缩建议
        
        # 基础变化
        change_rate = group0['pid_gain'] * group0['d_velocity']
        group0['d'] = group0['d'] * (1.0 + change_rate)
        
        # 修正：如果观测到了明确的远方，且我们正在加速中，允许快进
        if group0['d_velocity'] > 0 and d_max_observed > group0['d']:
             group0['d'] = min(d_max_observed, group0['d'])

        # 兜底
        group0['d'] = max(group0['d'], group0['d0'])
        
        # --- Phase 3: 参数更新 (Fused AdamW Style) ---
        # 预计算通用因子
        # 引入 bias correction 和 sqrt(k) 衰减
        bias_correction1 = 1 - group0['betas'][0] ** k
        bias_correction2 = 1 - group0['betas'][1] ** k
        
        # 这里的 step_size 包含了 LR, D, 和 sqrt(k) 衰减
        # Step = 1.0 * D / sqrt(k)
        step_size_base = group0['lr'] * group0['d'] / (k**0.5)
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # 1. Weight Decay (Decoupled)
                if weight_decay > 0:
                    p.data.mul_(1 - step_size_base * weight_decay)
                
                # 2. Adam Moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 3. Parameter Update
                denom = exp_avg_sq.sqrt().add_(eps)
                
                # 组合最终步长
                step_size = step_size_base * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return step_size_base # 返回当前的基础学习率供记录