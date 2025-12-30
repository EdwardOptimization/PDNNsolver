import math
import torch
import torch.optim
import torch.distributed as distributed

class PID_Prodigy_Restart(torch.optim.Optimizer):
    """
    PID-Prodigy-Restart (数学自适应版)

    新增特性（噪声鲁棒 P 项调节）：
    1. SNR-based adaptive brake: 使用 exp_avg / sqrt(exp_avg_sq) 估计梯度信噪比
       - SNR 低（噪声主导） → 自动减弱刹车
       - SNR 高（信号可靠） → 恢复强刹车
    2. Cosine variance-based factor: 短期 cosine 波动大 → 进一步抑制刹车
    3. 保留原有优秀设计：真实 disp_norm 重置 d、清零动量、跳过更新等
    """

    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), weight_decay=0.0,
                 d0=1e-5,
                 pid_beta=0.9,
                 pid_gain=0.02,
                 base_brake_ratio=1.2,       # 基础刹车力度（最大值）
                 restart_threshold=-0.6,
                 restart_patience=15,
                 d_max=10.0,
                 snr_target=2.0,             # SNR 目标值，可调 1.5~3.0
                 cos_var_weight=10.0,        # cosine 方差影响权重，可调 5~20
                 eps=1e-8):

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        d=d0, d0=d0,
                        pid_beta=pid_beta, pid_gain=pid_gain,
                        base_brake_ratio=base_brake_ratio,
                        restart_threshold=restart_threshold,
                        restart_patience=restart_patience,
                        d_max=d_max,
                        snr_target=snr_target,
                        cos_var_weight=cos_var_weight,
                        eps=eps)
        super().__init__(params, defaults)

        if not distributed.is_initialized() or distributed.get_rank() == 0:
            print(f">>> [PID-Prodigy-Restart] Adaptive Edition Initialized")
            print(f"    base_brake={base_brake_ratio} | snr_target={snr_target} | "
                  f"cos_var_weight={cos_var_weight} | d_max={d_max}")

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
            group['step'] += 1
            k = group['step']

            dot_product_sum = 0.0
            grad_norm_sq_sum = 0.0
            disp_norm_sq_sum = 0.0
            d_max_observed = group['d']

            # 用于 SNR 计算的全局统计
            exp_avg_norm_sum = 0.0
            exp_avg_sq_mean_sum = 0.0
            param_count = 0

            for p in group['params']:
                if p.grad is None:
                    continue
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
                if dist > d_max_observed:
                    d_max_observed = dist

                # 累积用于 SNR 计算（只对有动量的参数）
                exp_avg_norm_sum += state['exp_avg'].norm().item()
                exp_avg_sq_mean_sum += state['exp_avg_sq'].mean().item()
                param_count += 1

            # Cosine similarity
            grad_norm = math.sqrt(grad_norm_sq_sum) + 1e-8
            disp_norm = math.sqrt(disp_norm_sq_sum) + 1e-8
            cosine_sim = 1.0 if disp_norm < 1e-6 else dot_product_sum / (grad_norm * disp_norm)

            # === 自适应刹车力度计算（方案1 + 方案2）===
            # 1. SNR-based factor
            if param_count > 0:
                avg_exp_avg_norm = exp_avg_norm_sum / param_count
                avg_exp_avg_sq_mean = exp_avg_sq_mean_sum / param_count
                snr = avg_exp_avg_norm / (math.sqrt(avg_exp_avg_sq_mean) + 1e-8)
                snr_factor = min(1.0, snr / group['snr_target'])
            else:
                snr_factor = 0.5  # 保守fallback

            # 2. Cosine variance factor（指数移动方差）
            cos_beta = 0.9
            group['cos_mean'] = cos_beta * group['cos_mean'] + (1 - cos_beta) * cosine_sim
            group['cos_var'] = cos_beta * group['cos_var'] + (1 - cos_beta) * (cosine_sim - group['cos_mean']) ** 2
            cos_var_factor = 1.0 / (1.0 + group['cos_var_weight'] * group['cos_var'])

            # 最终自适应刹车力度
            adaptive_brake = group['base_brake_ratio'] * snr_factor * cos_var_factor

            # === Restart detection ===
            is_bad = cosine_sim < group['restart_threshold']
            group['stagnation_counter'] = group['stagnation_counter'] + 1 if is_bad else max(0, group['stagnation_counter'] - 1)
            do_restart = group['stagnation_counter'] > group['restart_patience']

            if do_restart:
                if not distributed.is_initialized() or distributed.get_rank() == 0:
                    g_name = "WEIGHTS" if group['weight_decay'] > 0 else "BIASES"
                    print(f">>> [RESTART TRIGGERED] {g_name} | Step {k} | Cos: {cosine_sim:.4f} | "
                          f"SNR: {snr:.2f} | AdaptiveBrake: {adaptive_brake:.3f}")

                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state['p0'].copy_(p.data)
                    state['exp_avg'].zero_()
                    state['exp_avg_sq'].zero_()

                group['d'] = disp_norm
                group['d_velocity'] = 0.0
                group['stagnation_counter'] = 0
                group['cos_mean'] = 0.0
                group['cos_var'] = 0.0

                continue

            # === PID Control with adaptive brake ===
            current_force = 1.0
            if dot_product_sum > 0:  # 坏方向
                current_force = -adaptive_brake

            beta = group['pid_beta']
            group['d_velocity'] = beta * group['d_velocity'] + (1 - beta) * current_force

            change_rate = group['pid_gain'] * group['d_velocity']
            group['d'] = group['d'] * (1.0 + change_rate)

            if group['d_velocity'] > 0 and d_max_observed > group['d']:
                group['d'] = min(d_max_observed, group['d'])

            group['d'] = min(group['d'], group['d_max'])
            group['d'] = max(group['d'], group['d0'])

            # === Parameter update ===
            step_size_base = group['lr'] * group['d']

            bias_correction1 = 1 - group['betas'][0] ** k
            bias_correction2 = 1 - group['betas'][1] ** k

            for p in group['params']:
                if p.grad is None:
                    continue
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

            # === 监控（增加自适应信息）===
            if k % 500 == 0 and (not distributed.is_initialized() or distributed.get_rank() == 0):
                g_name = "WEIGHTS" if group['weight_decay'] > 0 else "BIASES"
                print(f"  [Monitor] {g_name} | Step {k:5d} | D: {group['d']:.6f} | "
                      f"Cos: {cosine_sim:+.3f} | Vel: {group['d_velocity']:+.3f} | "
                      f"SNR: {snr:.2f} | Brake: {adaptive_brake:.3f} | LR_eff: {step_size_base:.8f}")

        return loss