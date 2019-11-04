#!/usr/bin/env bash

PYTHONPATH=./mnist python3 mnist/train_mnist.py \
  --distributed \
  --num_local_epochs 5 \
  --num_local_batch 50 \
  --learning_rate 0.15 \
  --num_user 100 \
  --num_subsets_per_user 1 \
