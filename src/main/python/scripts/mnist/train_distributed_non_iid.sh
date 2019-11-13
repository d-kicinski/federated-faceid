#!/usr/bin/env bash

PYTHONPATH=. python3 fedfaceid/mnist/train_mnist.py \
  --distributed \
  --non_iid \
  --num_local_epochs 5 \
  --num_local_batch 50 \
  --learning_rate 0.15 \
  --num_user 100 \
  --num_subsets_per_user 1
