#!/usr/bin/env bash


PYTHONPATH=. python ./scripts/python/data/align_faces.py \
     --data_dir "../../../data/vggface2/train" \
     --image_size 336 \
     --margin 44 \
     --num_workers 2

PYTHONPATH=. python ./scripts/python/data/align_faces.py \
    --data_dir "../../../data/vggface2/test" \
    --image_size 182 \
    --margin 44
