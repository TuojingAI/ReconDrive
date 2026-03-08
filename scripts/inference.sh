#!/bin/bash

CONFIG_PATH='./configs/nuscenes/recondrive.yaml'
CHECKPOINT_PATH='/data/yuntianbo/ReconDrive/recondrive_stage2.ckpt'
OUTPUT_DIR='./work_dirs/recondrive_stage2_eval_output'

DEVICE='0'
EVAL_RESOLUTION="280x518"

# SAVE_NOVEL_RENDERS for Left and Right Moving
NOVEL_VIEW_DISTANCES="1.0,2.0,3.0"
SAVE_NOVEL_RENDERS=false

python -m scripts.inference \
    --cfg_path="$CONFIG_PATH" \
    --restore_ckpt="$CHECKPOINT_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --device="$DEVICE" \
    --novel_distances="$NOVEL_VIEW_DISTANCES" \
    --eval_resolution="$EVAL_RESOLUTION" \
    $([ "$SAVE_NOVEL_RENDERS" = false ] && echo "--no_renders")