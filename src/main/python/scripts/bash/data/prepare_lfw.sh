#!/usr/bin/env bash


PYTHONPATH=. python ./scripts/python/data/align_faces.py \
     --data_dir "../../../data/lfw/raw" \
     --image_size 160 \
     --margin 32 \
     --num_workers 2

