#!/usr/bin/env bash

PYTHONPATH=./mnist python3 mnist/train_mnist.py \
  --id FedSGD \
  --skip_stopping \
  --distributed \
  --num_user 100 \
  --user_fraction 1.0 \
  --num_global_epochs 400 \
  --num_local_epochs 1 \
  --num_local_batch 500 \
  --learning_rate 0.6 \
  --learning_rate_decay 0.9934
