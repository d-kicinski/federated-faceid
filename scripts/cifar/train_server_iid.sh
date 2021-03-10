#!/usr/bin/env bash

source ./scripts/common.sh

python3 src/cifar/train.py \
  --id Baseline \
  --num_global_batch 256 \
  --num_global_epochs 100 \
  --learning_rate 0.01 \
  --learning_rate_decay 0.1
