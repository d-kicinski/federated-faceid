#!/usr/bin/env bash

source ./scripts/common.sh

DATA_DIR="data/lfw/raw"
OUTPUT_TEST_DIR="data/lfw/data/test"
OUTPUT_TRAIN_DIR="data/lfw/data/train"

python3 ./src/datautils/align_faces.py \
     --data_dir ${DATA_DIR}\
     --output_dir ${OUTPUT_TEST_DIR} \
     --image_size 160 \
     --margin 32 \
     --num_workers 2 \
     --cpu

python3 ./src/datautils/align_faces.py \
     --data_dir ${DATA_DIR}\
     --output_dir ${OUTPUT_TRAIN_DIR} \
     --image_size 182 \
     --margin 44 \
     --num_workers 2 \
     --cpu

python3 ./src/facenet/datasets/generate_csv_files.py \
    --dataroot  ${OUTPUT_TRAIN_DIR} \
    --csv_name "data/lfw/data/train_meta.csv"
