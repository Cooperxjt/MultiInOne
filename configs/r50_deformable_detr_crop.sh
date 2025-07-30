#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_sacd
PY_ARGS=${@:1}

python -u main_sacd.py \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --output_dir ${EXP_DIR} \
    --num_workers 4 \
    --batch_size 1 \
    --epochs 100 \
    --num_queries 4 \
    --topk_num 4 \
    --num_classes 1 \
    ${PY_ARGS}
