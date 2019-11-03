#!/usr/bin/env bash

PYTHONPATH=./mnist python3 mnist/train_mnist.py \
  --distributed \
  --non_iid
