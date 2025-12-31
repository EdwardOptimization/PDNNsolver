import math
import torch
import torch.optim
import torch.distributed as distributed

class PID_Prodigy_Restart(torch.optim.Optimizer):
    """
    PID-Prodigy with Restart Strategy (HPR-LP Style).
    
    A Parameter-Free Optimizer for Deep Learning.
    
    Core Concepts:
    1. Distance Estimation (Prodigy): 
       Estimates the distance to the optimal solution (D) to set step sizes.
       
    2. PID Control (Smoothness): 
       Uses a PID controller to adjust D smoothly based on gradient alignment,
       preventing the oscillation issues seen in raw Prodigy.
       
    3. Restart Strategy (HPR-LP): 
       Detects when the optimization trajectory is making a "U-Turn" (negative cosine similarity).
       If stagnation is detected, it performs a hard reset (Restart) to clear momentum
       and collapse D, allowing the optimizer to instantly change direction.
    """
    
    def __init__(self, params, lr=1.0, 
                 betas=(0.9, 0.999), weight_decay=0.0, 
                 d0=1e-5,            # Initial distance estimate (small value)
                 pid_beta=0.9,       # PID Integral/Velocity smoothing factor
                 pid_gain=0.02,      # PID Proportional gain (how fast D changes)
                 brake_ratio=2.0,    # How hard to brake when moving backwards
                 restart_threshold=-0.8, # Cosine similarity threshold to trigger stagnation count
                 restart_patience=10,    # Number of bad steps before triggering a restart
                 eps=1e-8):
        
        if distributed.is_initialized() and distributed.get_rank() == 0:
            print(f">>> [PID-Prodigy] Initialized (HPR-LP Original Style) <<<")
            print(f"    Monitoring Enabled. Restart Threshold: {restart_threshold}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        d=d0, d0=d0, 
                        pid_beta=pid_beta, pid_gain=pid_gain, 
                        brake_ratio=brake_ratio,
                        restart_threshold=restart_threshold,
                        restart_patience=restart_patience,
                        eps=eps)
        super(PID_Prodigy_Restart, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate over each parameter group
        for group in self.param_groups:
            if 'step' not in group: group['step'] = 0
            group['step'] += 1
            k = group['step']

            d_current = group['d']
            d_max_observed = d_current
            
            # Global statistics for this group
            dot_product_sum = 0.0 # <g, x-x0>
            grad_norm_sq_sum = 0.0
            disp_norm_sq_sum = 0.0
            
            # --- Phase 1: Calculate Global Geometry Stats ---
            # We must iterate all params first to get the global cosine similarity
            
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                
                # Lazy State Initialization
                if len(state) == 0:
                    # p0 is the "Anchor Point" (Initial value)
                    state['p0'] = p.clone().detach()
                    # Exponential moving averages for Adam
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    # Init group-level state if not present
                    if 'd_velocity' not in group: group['d_velocity'] = 0.0
                    if 'stagnation_counter' not in group: group['stagnation_counter'] = 0

                # Calculate displacement from anchor
                disp = p.sub(state['p0'])
                grad = p.grad
                
                # Accumulate scalars for cosine calculation
                # Using item() here is a bit slow on CPU but necessary for logic.
                # On GPU, the overhead is amortized if model is large enough.
                dot_product_sum += torch.sum(grad * disp).item()
                grad_norm_sq_sum += torch.sum(grad * grad).item()
                disp_norm_sq_sum += torch.sum(disp * disp).item()
                
                # Update Max Distance Observed
                dist = disp.norm().item()
                if dist > d_max_observed: d_max_observed = dist

            # --- Phase 2: Restart Logic (HPR-LP Style) ---
            
            grad_norm = math.sqrt(grad_norm_sq_sum) + 1e-8
            disp_norm = math.sqrt(disp_norm_sq_sum) + 1e-8
            
            # Cosine Similarity: -1 means exact opposite direction (U-Turn)
            cosine_sim = dot_product_sum / (grad_norm * disp_norm)
            
            # Check for bad direction
            is_bad_direction = cosine_sim < group['restart_threshold']
            
            if is_bad_direction:
                group['stagnation_counter'] += 1
            else:
                # Decay counter (Hysteresis) rather than instant reset
                group['stagnation_counter'] = max(0, group['stagnation_counter'] - 1)
                
            do_restart = group['stagnation_counter'] > group['restart_patience']
            
            if do_restart:
                # === EXECUTE RESTART ===
                # This corresponds to "resetting the method" in PDLP/HPR-LP
                if not distributed.is_initialized() or distributed.get_rank() == 0:
                    g_name = "WEIGHTS" if group['weight_decay'] > 0 else "BIASES "
                    print(f">>> [Restart Triggered] Group: {g_name} | Step {k} | Cos: {cosine_sim:.4f} | Resetting D & Momentum.")
                
                # 1. Reset D to current actual displacement (remove "bubble")
                group['d'] = disp_norm
                
                # 2. Kill PID Velocity (stop inertia)
                group['d_velocity'] = 0.0
                
                # 3. Reset Counter
                group['stagnation_counter'] = 0
                
                # 4. Kill Momentum (Adam States)
                # This allows the optimizer to turn immediately without fighting old history
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    state['exp_avg'].zero_()
                    state['exp_avg_sq'].zero_()
                
                # === CRITICAL ===
                # Skip parameter update for this step.
                # We just reset the state; using the current "bad" gradient to update 
                # immediately would pollute the fresh state.
                continue 

            # --- Phase 3: PID Control ---
            
            # Force Calculation: 
            # If dot_product > 0 (moving back/oscillating), Apply Brake.
            # Else (moving away), Apply Throttle.
            current_force = 1.0
            if dot_product_sum > 0: 
                current_force = -group['brake_ratio']

            # Update PID Velocity (I-term)
            beta = group['pid_beta']
            group['d_velocity'] = beta * group['d_velocity'] + (1 - beta) * current_force
            
            # Apply gain to update D
            change_rate = group['pid_gain'] * group['d_velocity']
            group['d'] = group['d'] * (1.0 + change_rate)
            
            # Trust Region Expansion (Fast-forward)
            # Only allow expansion if we are accelerating (velocity > 0)
            if group['d_velocity'] > 0 and d_max_observed > group['d']:
                 group['d'] = min(d_max_observed, group['d'])

            # Lower bound protection
            group['d'] = max(group['d'], group['d0'])
            
            # --- Phase 4: Parameter Update (Fused AdamW Style) ---
            
            # Prodigy Step Size Formula: LR * D / sqrt(k)
            step_size_base = group['lr'] * group['d'] / (k**0.5)
            
            # [Added] Monitoring Logic
            if k % 10 == 0 and (not distributed.is_initialized() or distributed.get_rank() == 0):
                g_name = "WEIGHTS" if group['weight_decay'] > 0 else "BIASES "
                # Print real LR to see if PID is crushing it
                print(f"  [Monitor] {g_name} | Step {k} | D: {group['d']:.6f} | Cos: {cosine_sim:.4f} | Vel: {group['d_velocity']:.4f} | LR: {step_size_base:.8f}")
            
            bias_correction1 = 1 - group['betas'][0] ** k
            bias_correction2 = 1 - group['betas'][1] ** k
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # Decoupled Weight Decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - step_size_base * group['weight_decay'])
                
                # Update Moments
                exp_avg.mul_(group['betas'][0]).add_(grad, alpha=1 - group['betas'][0])
                exp_avg_sq.mul_(group['betas'][1]).addcmul_(grad, grad, value=1 - group['betas'][1])
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Final Step Size with Adam Bias Correction
                step_size = step_size_base * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss