#!/usr/bin/env bash

source ./scripts/common.sh

python3 src/cifar/train.py \
  --id Baseline \
  --num_global_batch 100 \
  --num_global_epochs 400 \
  --learning_rate 0.15 \
  --learning_rate_decay 0.99
