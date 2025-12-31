import math
import torch
import torch.optim
import torch.distributed as distributed

class PID_Prodigy_Restart(torch.optim.Optimizer):
    """
    PID-Prodigy-Restart (V44 - Step Denoising Edition)
    
    [用户核心思想]: 不用 Diffusion 生成权重，而是用 Diffusion/Denoising 思想去生成最优的 Step Size。
    
    1. [PID Controller]: 负责产生原始的、反应灵敏的步长 d_raw (Drift)。
    2. [Kalman Denoiser]: 负责对 d_raw 进行去噪。
       - 前期 (High SNR): 信任 PID，跟随变化。
       - 后期 (Low SNR): 不信任 PID 的震荡，执行平滑的去噪轨迹 (Auto-Landing)。
    3. [Result]: 一条完美的、丝滑的收敛曲线。
    """

    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), weight_decay=0.0,
                 d0=1e-5,
                 pid_beta=0.9,
                 pid_gain=0.02,
                 base_brake_ratio=1.2,
                 
                 # [特性 1] 动态阈值重启 (V35)
                 restart_threshold_init=0.5,    # 起步宽容
                 min_restart_threshold=0.01,    # 最终严厉
                 restart_patience=15,
                 
                 # [特性 2] SNR 自适应衰减 (V40)
                 # 基础阻力系数：当全是噪声时，每步衰减多少。
                 # 2e-4 意味着约 5000 步减半。
                 base_d_decay=5e-4,             
                 
                 d_max=20.0,
                 max_lr=0.01,                   # 熔断保护
                 snr_target=6.0,
                 cos_var_weight=10.0,
                 eps=1e-8,
                 # [V44 新特性] 去噪强度
                 # 值越小，去噪越强（越不信任瞬时 PID，越平滑）
                 # 建议 0.1 ~ 0.5
                 kalman_gain_init=0.5):

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        d=d0, d0=d0,
                        pid_beta=pid_beta, pid_gain=pid_gain,
                        base_brake_ratio=base_brake_ratio,
                        restart_threshold_init=restart_threshold_init,
                        min_restart_threshold=min_restart_threshold,
                        restart_patience=restart_patience,
                        base_d_decay=base_d_decay,  # 保存阻力参数
                        d_max=d_max,
                        max_lr=max_lr,
                        snr_target=snr_target,
                        kalman_gain_init=kalman_gain_init, # 保存去噪参数
                        # # Kalman 状态初始化
                        # p_estimate=d0,  # 估计值 (Posterior)
                        # p_error=1.0,    # 估计误差协方差
                        cos_var_weight=cos_var_weight,
                        eps=eps)
        super().__init__(params, defaults)

        if not distributed.is_initialized() or distributed.get_rank() == 0:
            print(f">>> [PID-Prodigy] V44 Step-Denoiser Loaded | Denoising noise via Kalman Filter")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                group['step'] = 0
                group['d_velocity'] = 0.0
                group['stagnation_counter'] = 0
                group['cos_mean'] = 0.0
                group['cos_var'] = 0.0
                # Kalman 状态
                group['k_est'] = group['d']  # 初始估计
                group['k_err'] = 1.0         # 初始不确定性
            
            group['step'] += 1
            k = group['step']

            # --- 统计量 ---
            dot_product_sum = 0.0
            grad_norm_sq_sum = 0.0
            disp_norm_sq_sum = 0.0
            d_max_observed = group['d']

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

            # --- 核心指标 ---
            grad_norm = math.sqrt(grad_norm_sq_sum) + 1e-8
            disp_norm = math.sqrt(disp_norm_sq_sum) + 1e-8
            cosine_sim = -1.0 if disp_norm < 1e-6 else dot_product_sum / (grad_norm * disp_norm)

            # --- SNR 计算 (用于刹车和衰减) ---
            if param_count > 0:
                avg_exp_avg_norm = exp_avg_norm_sum / param_count
                avg_exp_avg_sq_mean = exp_avg_sq_mean_sum / param_count
                snr = avg_exp_avg_norm / (math.sqrt(avg_exp_avg_sq_mean) + 1e-8)
                # SNR Factor: 1.0 = High Quality, 0.0 = Pure Noise
                snr_factor = min(1.0, snr / group['snr_target'])
            else:
                snr_factor = 0.5

            cos_beta = 0.9
            group['cos_mean'] = cos_beta * group['cos_mean'] + (1 - cos_beta) * cosine_sim
            group['cos_var'] = cos_beta * group['cos_var'] + (1 - cos_beta) * (cosine_sim - group['cos_mean']) ** 2
            cos_var_factor = 1.0 / (1.0 + group['cos_var_weight'] * group['cos_var'])
            
            adaptive_brake = group['base_brake_ratio'] * snr_factor * cos_var_factor

            # --- 动态阈值重启 (V35 逻辑) ---
            decay_factor = 1.0 / (k ** 0.5) 
            current_restart_threshold = group['restart_threshold_init'] * decay_factor
            current_restart_threshold = max(group['min_restart_threshold'], current_restart_threshold)

            is_bad = cosine_sim > current_restart_threshold
            group['stagnation_counter'] = group['stagnation_counter'] + 1 if is_bad else max(0, group['stagnation_counter'] - 1)
            do_restart = group['stagnation_counter'] > group['restart_patience']

            if do_restart:
                if not distributed.is_initialized() or distributed.get_rank() == 0:
                    g_name = "WEIGHTS" if group['weight_decay'] > 0 else "BIASES"
                    print(f">>> [RESTART] {g_name} | Step {k} | Cos: {cosine_sim:.4f} | "
                          f"Threshold: {current_restart_threshold:.4f} | Action: Soft Reset")

                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    state['p0'].copy_(p.data)
                    state['exp_avg'].zero_()
                    state['exp_avg_sq'].zero_()

                # Soft Restart: 继承速度
                group['d'] = disp_norm
                group['d_velocity'] = 0.0
                group['stagnation_counter'] = 0
                group['cos_mean'] = 0.0
                group['cos_var'] = 0.0
                continue

            # --- PID Bang-Bang Control ---
            current_force = 1.0
            if cosine_sim > 0: 
                current_force = -adaptive_brake

            beta = group['pid_beta']
            group['d_velocity'] = beta * group['d_velocity'] + (1 - beta) * current_force
            change_rate = group['pid_gain'] * group['d_velocity']
            
            d_raw = group['d'] * (1.0 + change_rate)
            
            # 2. Physics Decay (V42)
            # 注意：这里我们让 snr_target 更高，产生更强的阻力
            # 但这个阻力是作用在 d_raw 上的
            adaptive_decay = group['base_d_decay'] * (1.0 - snr_factor)
            d_raw *= (1.0 - adaptive_decay)

            # ==========================================
            # [V44 最终修正] 相对卡尔曼滤波 (Scale-Invariant)
            # ==========================================
            
            # 1. 预测 (Predict)
            pred_d = group['k_est'] 
            
            # [Process Noise Q]: 设定为当前 d 的 5% (相对变化)
            # 加上 1e-12 防止 d=0 时的数值错误
            process_std = pred_d * 0.05
            process_noise_Q = process_std ** 2 + 1e-12
            
            pred_err = group['k_err'] + process_noise_Q 
            
            # 2. 测量噪声 R (Measurement Noise) - 关键修改！
            # 我们根据 SNR 动态计算"相对噪声比例"
            # High SNR (1.0) -> relative_R_factor = 0.01 (1% 误差，信任 PID)
            # Low SNR (0.0)  -> relative_R_factor = 2.0  (200% 误差，极度不信)
            relative_R_factor = (1.0 - snr_factor) * 2.0 + 0.01
            
            # 将相对比例转换为绝对方差 R
            measure_std = d_raw * relative_R_factor
            measure_noise_R = measure_std ** 2 + 1e-12
            
            # 3. 更新 (Update)
            # 现在的 K 完全由 (Q/R) 的比例决定，与 d 的大小无关！
            kalman_gain = pred_err / (pred_err + measure_noise_R)
            
            new_est = pred_d + kalman_gain * (d_raw - pred_d)
            new_err = (1.0 - kalman_gain) * pred_err
            
            # 保存
            group['k_est'] = new_est
            group['k_err'] = new_err
            group['d'] = new_est

            # # 3. Trust Region
            # if group['d_velocity'] > 0 and d_max_observed > group['d']:
            #     group['d'] = min(d_max_observed, group['d'])

            # 4. Limits
            group['d'] = min(group['d'], group['d_max'])
            group['d'] = max(group['d'], group['d0'])

            # --- 参数更新 ---
            # 基础步长：lr * d / sqrt(k) (保留 V35 的 sqrt(k) 以配合动态阈值)
            # 再加上 max_lr 熔断保护
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

            # --- 监控 (完整版) ---
            if k % 500 == 0 and (not distributed.is_initialized() or distributed.get_rank() == 0):
                g_name = "WEIGHTS" if group['weight_decay'] > 0 else "BIASES"
                # 现在打印 SNR, Decay 和 Threshold，一切尽在掌握
                print(f"  [Monitor] {g_name} | Step {k:5d} | D: {group['d']:.6f} | "
                      f"Cos: {cosine_sim:+.3f} | SNR: {snr_factor:.2f} | "
                      f"Decay: {adaptive_decay:.1e} | LR: {step_size_base:.8f}")

        return loss