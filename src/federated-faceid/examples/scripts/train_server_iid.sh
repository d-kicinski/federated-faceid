#!/usr/bin/env bash

PYTHONPATH=./mnist python3 mnist/train_mnist.py \
    --num_global_batch 100 \
    --num_global_epochs 400 \
    --learning_rate 0.15 \
    --learning_rate_decay 0.99 \
    --skip_stopping
