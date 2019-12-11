#!/usr/bin/env bash


PYTHONPATH=. python ./scripts/python/data/align_faces.py \
     --data_dir "../../../data/nv/nv-gen" \
     --data_dir "../../../data/nv/nv-gen_cropped" \
     --image_size 182 \
     --margin 44 \
     --num_workers 2 \
     --cpu

PYTHONPATH=. python3 facenet/datasets/generate_csv_files.py \
    --dataroot  "../../../data/nv/nv-gen_cropped" \
    --csv_name "../../../data/nv/nv-gen_train_meta.csv"


