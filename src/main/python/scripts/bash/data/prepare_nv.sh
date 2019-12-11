#!/usr/bin/env bash


PYTHONPATH=. python ./scripts/python/data/align_faces.py \
     --data_dir "../../../data/nv-gen" \
     --image_size 182 \
     --margin 44 \
     --num_workers 2 \
     --cpu
