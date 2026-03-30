#!/bin/bash
CONFIG_PATH='./configs/nuscenes/recondrive.yaml'
PRETRAINED_CHECKPOINT_PATH='./checkpoints/recondrive_stage1.ckpt'

python -m scripts.trainer \
    --cfg_path=${CONFIG_PATH} \
    --train_4d \
    --devices="${1:-1}" \
    --pretrained_ckpt=${PRETRAINED_CHECKPOINT_PATH}
