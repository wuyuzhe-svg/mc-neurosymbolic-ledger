#!/bin/bash
# fire_v5.sh — main training launcher (batched paired forward + de-warmed place column + binarized action bits)
# keep WARM_PLACE=0
cd /root/autodl-tmp/vrisingwm || exit 1
source /root/venvs/wm/bin/activate

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export V3=1 LPROBE=1 LEDGER=1 W_PRESERVE=0 W_CF=0
export WARM_PLACE=0
export STEPS=20000 BS=2
export POS_FRAC=0.50 NEG_FRAC=0.10
export T_PROBE=800 T_HI_LO=500 T_HI_HI=1000   # 高噪逼迫
export RESUME=/root/autodl-tmp/vrisingwm/checkpoints/keep/ft_12000.pt
export OUT=/root/autodl-tmp/vrisingwm/checkpoints/mg2_v5

exec torchrun --nproc_per_node=4 --master_port=29617 \
  /root/autodl-tmp/vrisingwm/training/train_mg2.py \
  > /root/autodl-tmp/vrisingwm/logs/v5_train.log 2>&1
