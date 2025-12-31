#!/bin/bash

# ================= 配置区域 =================
BATCH_SIZE=64
GRAD_ACCUM=8
MAX_ITERS=20000
WANDB_PROJECT="owt-benchmark-h100"
CONFIG="config/train_gpt2.py"

mkdir -p logs

echo "=================================================="
echo "🚀 开始 H100 GPT-2 优化器对决 (20000 步) - Adaptive Edition"
echo "开始时间: $(date)"
echo "Batch Size: $BATCH_SIZE | Grad Accum: $GRAD_ACCUM | Tokens/Iter: $((BATCH_SIZE * 1024 * GRAD_ACCUM))"
echo "=================================================="

# --------------------------------------------------
# 任务 1: PID-Prodigy-Restart (自适应数学版)
# --------------------------------------------------
echo "正在启动: 🟢 PID-Prodigy-Restart (Adaptive Edition)"
echo "日志文件: logs/pid_restart_adaptive.log"

python -u train.py $CONFIG \
    --optimizer=pid_prodigy_restart \
    --learning_rate=1.0 \
    --warmup_iters=0 \
    --max_iters=$MAX_ITERS \
    --batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --dtype=bfloat16 \
    --compile=True \
    --wandb_log=True \
    --wandb_project=$WANDB_PROJECT \
    --wandb_run_name='gpt2-pid-restart-adaptive-v46' \
    > logs/pid_restart_adaptive_v46.log 2>&1

echo "✅ PID-Prodigy-Restart (Adaptive) 启动完成！"
echo "休息 20 秒..."
sleep 20

# # --------------------------------------------------
# # 任务 2: AdamW Baseline
# # --------------------------------------------------
# echo "正在启动: 🔴 AdamW Baseline..."
# echo "日志文件: logs/adamw_baseline.log"

# python -u train.py $CONFIG \
#     --optimizer=adamw \
#     --learning_rate=6e-4 \
#     --weight_decay=0.1 \
#     --warmup_iters=2000 \
#     --max_iters=$MAX_ITERS \
#     --batch_size=$BATCH_SIZE \
#     --gradient_accumulation_steps=$GRAD_ACCUM \
#     --dtype=bfloat16 \
#     --compile=True \
#     --wandb_log=True \
#     --wandb_project=$WANDB_PROJECT \
#     --wandb_run_name='gpt2-adamw-baseline' \
#     > logs/adamw_baseline.log 2>&1

# echo "=================================================="
# echo "🎉 所有任务已启动！"
# echo "结束时间: $(date)"
# echo "战报链接: https://wandb.ai/edwardqu-individual/$WANDB_PROJECT"
# echo "自适应版准备好彻底碾压 AdamW 了吗？🚀"
# echo "=================================================="