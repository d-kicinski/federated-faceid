#!/usr/bin/env bash

PYTHONPATH=. python3 fedfaceid/mnist/train_mnist.py \
  --id FedAvg \
  --skip_stopping \
  --distributed \
  --num_user 100 \
  --user_fraction 0.1 \
  --num_global_epochs 4000 \
  --num_local_epochs 5 \
  --num_local_batch 50 \
  --learning_rate 0.25 \
  --learning_rate_decay 0.99
