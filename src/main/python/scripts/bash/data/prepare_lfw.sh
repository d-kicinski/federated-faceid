#!/usr/bin/env bash


PYTHONPATH=. python ./scripts/python/data/align_faces.py \
     --data_dir "../../../data/lfw/raw" \
     --output_dir "../../../data/lfw/data/test" \
     --image_size 160 \
     --margin 32 \
     --num_workers 2 \
     --cpu

PYTHONPATH=. python ./scripts/python/data/align_faces.py \
     --data_dir "../../../data/lfw/raw" \
     --output_dir "../../../data/lfw/data/train" \
     --image_size 182 \
     --margin 44 \
     --num_workers 2 \
     --cpu
