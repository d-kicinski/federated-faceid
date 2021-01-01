#!/usr/bin/env bash

source ./scripts/common.sh

python3 src/cifar/train.py \
  --distributed \
  --non_iid \
  --num_local_epochs 5 \
  --num_local_batch 50 \
  --learning_rate 0.15 \
  --num_user 100 \
  --num_subsets_per_user 1
