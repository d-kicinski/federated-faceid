#!/usr/bin/env bash

source ./scripts/common.sh

python3 src/cifar/train.py \
  --id ServerResnetBatchNorm \
  --num_global_batch 256 \
  --num_global_epochs 100 \
  --learning_rate 0.01 \
  --learning_rate_decay 0.1 \
  --model_class Resnet18 \
  --layer_norm_class BatchNorm
