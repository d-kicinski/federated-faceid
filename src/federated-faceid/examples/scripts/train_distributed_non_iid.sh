#!/usr/bin/env bash

PYTHONPATH=./mnist python3 mnist/train_mnist.py \
  --distributed \
  --non_iid \
  --num_local_epochs 5 \
  --num_local_batch 50 \
  --learning_rate 0.15
