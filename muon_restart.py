import math
import torch
import torch.optim as optim

def newton_schulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz Orthogonalization
    Standard implementation for Muon.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm(p=2, dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

class Muon_Fixed_Restart(optim.Optimizer):
    """
    Muon + HPR-LP Restart (Fixed LR version).
    
    Mechanism:
    1. Update is Standard Muon (Orthogonalized Momentum).
    2. Step size is fixed (controlled by external LR schedule).
    3. Restart: If trajectory makes a U-Turn (cos(g, x-x0) < threshold),
       momentum is cleared and the 'anchor point' x0 is reset.
    """
    def __init__(self, params, lr=0.05, 
                 momentum=0.95, 
                 weight_decay=0.01,
                 ns_steps=5,
                 adamw_params=None,
                 restart_threshold=-0.5, # 稍微宽松一点的阈值
                 restart_patience=20):   # 给Muon一点调整惯性的时间
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps,
                        restart_threshold=restart_threshold,
                        restart_patience=restart_patience)
        
        super().__init__(params, defaults)
        
        # Internal AdamW defaults for 1D params (biases/norms)
        self.adamw_defaults = dict(lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        if adamw_params:
            self.adamw_defaults.update(adamw_params)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        for group in self.param_groups:
            # === [新增] 初始化并更新计数器 k ===
            if 'step' not in group: group['step'] = 0
            group['step'] += 1
            k = group['step']  # <--- 这就是报错缺少的 k
            # Common params
            lr = group['lr'] # Expecting Cosine Decay from outside
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            
            # Restart counters
            if 'stagnation_counter' not in group: group['stagnation_counter'] = 0

            # --- Phase 1: Check Global Restart Condition ---
            # HPR-LP Logic: Check alignment between Gradient and Displacement (x - x0)
            dot_product_sum = 0.0
            grad_norm_sq_sum = 0.0
            disp_norm_sq_sum = 0.0
            
            # We need a second loop to gather stats before updating
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                
                # Init State
                if len(state) == 0:
                    state['p0'] = p.clone().detach() # Anchor point x0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    # For AdamW parts
                    if p.ndim < 2:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                # Calc stats
                disp = p.sub(state['p0'])
                grad = p.grad
                
                dot_product_sum += torch.sum(grad * disp).item()
                grad_norm_sq_sum += torch.sum(grad * grad).item()
                disp_norm_sq_sum += torch.sum(disp * disp).item()

            # Analyze Trajectory
            grad_norm = math.sqrt(grad_norm_sq_sum) + 1e-8
            disp_norm = math.sqrt(disp_norm_sq_sum) + 1e-8
            cosine_sim = dot_product_sum / (grad_norm * disp_norm)

            if k % 100 == 0:
                print(f"[Step {k}] Cosine Sim: {cosine_sim:.4f} (Threshold: {group['restart_threshold']})")
            
            # Restart Trigger Logic
            if cosine_sim < group['restart_threshold']:
                group['stagnation_counter'] += 1
            else:
                group['stagnation_counter'] = max(0, group['stagnation_counter'] - 1)
            
            do_restart = group['stagnation_counter'] > group['restart_patience']
            
            if do_restart:
                print(f"[Muon-Restart] Triggered! (Cos: {cosine_sim:.4f}). Clearing Momentum.")
                group['stagnation_counter'] = 0
                
                # === EXECUTE RESTART ===
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    
                    # 1. Reset Momentum (Kill inertia)
                    state['momentum_buffer'].zero_()
                    if p.ndim < 2:
                        state['exp_avg'].zero_()
                        state['exp_avg_sq'].zero_()
                    
                    # 2. Reset Anchor (Treat current pos as new start)
                    state['p0'].copy_(p)
                    
                # Skip update this step to let gradients stabilize
                return loss

            # --- Phase 2: Apply Updates (Standard Muon / AdamW) ---
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]

                # === Muon Path (2D+) ===
                if p.ndim >= 2:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    
                    # Nesterov
                    g_update = g.add(buf, alpha=momentum)
                    
                    # Orthogonalize
                    g_ortho = newton_schulz5(g_update, steps=ns_steps)
                    
                    # Fixed LR Update
                    # Weight decay is handled implicitly or via AdamW style if added here
                    # Standard Muon usually does raw update
                    p.data.add_(g_ortho.to(p.dtype), alpha=-lr)

                # === AdamW Path (1D) ===
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    beta1, beta2 = self.adamw_defaults['betas']
                    wd = self.adamw_defaults['weight_decay']
                    
                    p.data.mul_(1 - lr * wd)
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(1e-8)
                    p.data.addcdiv_(exp_avg, denom, value=-lr)

        return loss