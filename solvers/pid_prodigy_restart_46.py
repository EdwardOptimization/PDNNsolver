import math
import torch
import torch.optim
import torch.distributed as distributed

class PID_Prodigy_Restart(torch.optim.Optimizer):
    """
    PID-Prodigy-Restart (V46 - Fuzzy Kalman PID)
    
    [架构]: Fuzzy Controller -> PID Execution -> Kalman Filter -> Parameter Update
    
    1. [Fuzzy Brain]: 根据 Cosine(方向) 和 SNR(路况) 动态决策 PID 参数。
       - 好的时候：High Gain (加速快), Low Decay (阻力小)。
       - 差的时候：Low Gain (微操), High Decay (稳住)。
    2. [PID Body]: 执行加速/减速指令，产生 d_raw。
    3. [Kalman Skin]: 对 d_raw 进行相对去噪，滤除震荡，输出丝滑的 d_final。
    """

    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), weight_decay=0.0,
                 d0=1e-5, pid_beta=0.9, 
                 
                 # [Fuzzy 参数范围]
                 # 增益动态范围: 0.005 (稳) ~ 0.05 (快)
                 # V44 固定是 0.02
                 min_gain=0.005, max_gain=0.05,
                 
                 # 阻力动态范围: 1e-5 (滑行) ~ 8e-4 (急刹)
                 # V44 固定是 5e-4
                 min_decay=1e-5, max_decay=8e-4,
                 
                 # [Kalman/SNR 参数]
                 snr_target=6.0,
                 base_brake_ratio=1.2,
                 
                 # [Restart 参数]
                 restart_threshold_init=0.5, min_restart_threshold=0.01, restart_patience=15,
                 
                 d_max=30.0, max_lr=0.01,
                 cos_var_weight=10.0, eps=1e-8):

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        d=d0, d0=d0,
                        pid_beta=pid_beta,
                        
                        # Fuzzy Ranges
                        min_gain=min_gain, max_gain=max_gain,
                        min_decay=min_decay, max_decay=max_decay,
                        
                        snr_target=snr_target,
                        base_brake_ratio=base_brake_ratio,
                        restart_threshold_init=restart_threshold_init,
                        min_restart_threshold=min_restart_threshold,
                        restart_patience=restart_patience,
                        
                        d_max=d_max, max_lr=max_lr,
                        cos_var_weight=cos_var_weight, eps=eps)
        super().__init__(params, defaults)
        
        if not distributed.is_initialized() or distributed.get_rank() == 0:
            print(f">>> [PID-Prodigy] V46 Fuzzy-Kalman Loaded | Intelligent Gain & Smooth Filter")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # --- 初始化 ---
            if 'step' not in group:
                group['step'] = 0
                group['d_velocity'] = 0.0
                group['stagnation_counter'] = 0
                group['cos_mean'] = 0.0
                group['cos_var'] = 0.0
                
                # Kalman 状态初始化 (Best Practice)
                group['k_est'] = group['d']
                # 相对初始化: 初始误差设为 d 的 100%，确保起步时 Filter 即使在低 d 下也能工作
                group['k_err'] = group['d'] * 1.0 
            
            group['step'] += 1
            k = group['step']

            # --- 1. 统计量计算 (Stats) ---
            dot_product_sum = 0.0
            grad_norm_sq_sum = 0.0
            disp_norm_sq_sum = 0.0
            d_max_observed = group['d'] # 用于软启动保护

            exp_avg_norm_sum = 0.0
            exp_avg_sq_mean_sum = 0.0
            param_count = 0

            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if len(state) == 0:
                    state['p0'] = p.clone().detach()
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                disp = p - state['p0']
                grad = p.grad

                dot_product_sum += (grad * disp).sum().item()
                grad_norm_sq_sum += (grad * grad).sum().item()
                disp_norm_sq_sum += (disp * disp).sum().item()
                
                dist = disp.norm().item()
                if dist > d_max_observed: d_max_observed = dist

                exp_avg_norm_sum += state['exp_avg'].norm().item()
                exp_avg_sq_mean_sum += state['exp_avg_sq'].mean().item()
                param_count += 1

            if distributed.is_initialized() and p.device is not None:
                metrics = torch.tensor([dot_product_sum, grad_norm_sq_sum, disp_norm_sq_sum], device=p.device)
                distributed.all_reduce(metrics, op=distributed.ReduceOp.SUM)
                d_max_tensor = torch.tensor([d_max_observed], device=p.device)
                distributed.all_reduce(d_max_tensor, op=distributed.ReduceOp.MAX)
                metrics /= distributed.get_world_size()
                dot_product_sum, grad_norm_sq_sum, disp_norm_sq_sum = metrics.tolist()
                d_max_observed = d_max_tensor.item()

            grad_norm = math.sqrt(grad_norm_sq_sum) + 1e-8
            disp_norm = math.sqrt(disp_norm_sq_sum) + 1e-8
            cosine_sim = -1.0 if disp_norm < 1e-6 else dot_product_sum / (grad_norm * disp_norm)

            # SNR Calc
            if param_count > 0:
                avg_exp_avg_norm = exp_avg_norm_sum / param_count
                avg_exp_avg_sq_mean = exp_avg_sq_mean_sum / param_count
                snr = avg_exp_avg_norm / (math.sqrt(avg_exp_avg_sq_mean) + 1e-8)
                snr_factor = min(1.0, snr / group['snr_target'])
            else:
                snr_factor = 0.5

            # Cosine Variance Calc
            cos_beta = 0.9
            group['cos_mean'] = cos_beta * group['cos_mean'] + (1 - cos_beta) * cosine_sim
            group['cos_var'] = cos_beta * group['cos_var'] + (1 - cos_beta) * (cosine_sim - group['cos_mean']) ** 2
            cos_var_factor = 1.0 / (1.0 + group['cos_var_weight'] * group['cos_var'])

            # --- 2. 动态阈值重启 (Restart Logic) ---
            decay_factor = 1.0 / (k ** 0.5)
            current_restart_threshold = group['restart_threshold_init'] * decay_factor
            current_restart_threshold = max(group['min_restart_threshold'], current_restart_threshold)

            is_bad = cosine_sim > current_restart_threshold
            group['stagnation_counter'] = group['stagnation_counter'] + 1 if is_bad else max(0, group['stagnation_counter'] - 1)
            
            if group['stagnation_counter'] > group['restart_patience']:
                if not distributed.is_initialized() or distributed.get_rank() == 0:
                    g_name = "WEIGHTS" if group['weight_decay'] > 0 else "BIASES"
                    print(f">>> [RESTART] {g_name} | Step {k} | Cos: {cosine_sim:.4f} | Soft Reset")
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    state['p0'].copy_(p.data)
                    state['exp_avg'].zero_()
                    state['exp_avg_sq'].zero_()
                group['d'] = disp_norm
                group['d_velocity'] = 0.0
                group['stagnation_counter'] = 0
                continue

            # ==========================================
            # [Layer A] Fuzzy Logic Brain (智能决策)
            # ==========================================
            
            # 输入 1: 方向好不好? (Cos 越负越好, Sigmoid 映射到 0~1)
            is_good_dir = torch.sigmoid(torch.tensor(-5.0 * cosine_sim)).item()
            
            # 输入 2: 信号强不强?
            is_high_snr = snr_factor
            
            # 规则 1: 信心指数 (既方向对，又信号强) -> 决定 Gain
            confidence_score = is_good_dir * is_high_snr
            
            # 规则 2: 刹车指数 (信号差，或者方向乱) -> 决定 Decay
            landing_score = 1.0 - is_high_snr
            
            # 输出 A: 动态增益 (Gain)
            # 范围: min_gain(0.005) ~ max_gain(0.05)
            fuzzy_gain = group['min_gain'] + (group['max_gain'] - group['min_gain']) * confidence_score
            
            # 输出 B: 动态阻力 (Decay)
            # 范围: min_decay(1e-5) ~ max_decay(8e-4)
            # 注意: 这里计算的是"基础阻力系数"，后面还会乘以 (1-snr)
            fuzzy_decay_base = group['min_decay'] + (group['max_decay'] - group['min_decay']) * landing_score

            # ==========================================
            # [Layer B] PID Execution (执行)
            # ==========================================
            
            adaptive_brake = group['base_brake_ratio'] * snr_factor * cos_var_factor
            current_force = 1.0
            if cosine_sim > 0: 
                current_force = -adaptive_brake

            beta = group['pid_beta']
            group['d_velocity'] = beta * group['d_velocity'] + (1 - beta) * current_force
            
            # [关键] 使用 Fuzzy Gain 更新
            change_rate = fuzzy_gain * group['d_velocity']
            d_raw = group['d'] * (1.0 + change_rate)
            
            # [关键] 使用 Fuzzy Decay 施加阻力
            # 这里 logic 是: base_decay * (1 - snr)
            adaptive_decay = fuzzy_decay_base * (1.0 - snr_factor)
            d_raw *= (1.0 - adaptive_decay)

            # ==========================================
            # [Layer C] Kalman Step Denoiser (相对去噪)
            # ==========================================
            
            # 1. 预测
            pred_d = group['k_est']
            
            # [V44 FIX] 相对 Process Noise: 假设 d 每步有 5% 的内在变化
            process_std = pred_d * 0.05
            process_noise_Q = process_std ** 2 + 1e-12
            pred_err = group['k_err'] + process_noise_Q
            
            # 2. 测量噪声 R (相对)
            # High SNR -> 1% error; Low SNR -> 200% error
            relative_R_factor = (1.0 - snr_factor) * 2.0 + 0.01
            measure_std = d_raw * relative_R_factor
            measure_noise_R = measure_std ** 2 + 1e-12
            
            # 3. 更新
            kalman_gain = pred_err / (pred_err + measure_noise_R)
            new_est = pred_d + kalman_gain * (d_raw - pred_d)
            new_err = (1.0 - kalman_gain) * pred_err
            
            group['k_est'] = new_est
            group['k_err'] = new_err
            
            # 赋值给 d
            group['d'] = new_est

            # ==========================================
            # [Layer D] Safety Limits
            # ==========================================
            
            # [V44 FIX] 软启动保护 (前 100 步)
            # 防止 Fuzzy Gain 在初期过猛导致 Loss 爆炸
            if k < 100 and d_max_observed > 0:
                 group['d'] = min(group['d'], d_max_observed * 2.0)

            # 硬限制
            group['d'] = min(group['d'], group['d_max'])
            group['d'] = max(group['d'], group['d0'])

            # --- 参数更新 (AdamW) ---
            raw_step_size = group['lr'] * group['d'] / (k**0.5)
            step_size_base = min(raw_step_size, group['max_lr'])

            bias_correction1 = 1 - group['betas'][0] ** k
            bias_correction2 = 1 - group['betas'][1] ** k

            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if group['weight_decay'] > 0:
                    p.data.mul_(1 - step_size_base * group['weight_decay'])

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(group['betas'][0]).add_(grad, alpha=1 - group['betas'][0])
                exp_avg_sq.mul_(group['betas'][1]).addcmul_(grad, grad, value=1 - group['betas'][1])
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = step_size_base * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

            # --- Monitor (V46 增强版) ---
            if k % 500 == 0 and (not distributed.is_initialized() or distributed.get_rank() == 0):
                g_name = "WEIGHTS" if group['weight_decay'] > 0 else "BIASES"
                # 打印出 Fuzzy 计算的 Gain 和 Decay，方便观察"大脑"的运作
                print(f"  [V46] {g_name} | Step {k:5d} | D: {group['d']:.5f} | "
                      f"Gain: {fuzzy_gain:.4f} | Decay: {fuzzy_decay_base:.1e} | SNR: {snr_factor:.2f}")

        return loss