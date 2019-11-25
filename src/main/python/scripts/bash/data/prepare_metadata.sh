#!/usr/bin/env bash

python3 fedfaceid/facenet/datasets/generate_csv_files.py \
    --dataroot  "../../../data/vggface2/train_cropped" \
    --csv_name "../../../data/vggface2/train_cropped_meta.csv"

