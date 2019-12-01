#!/usr/bin/env bash
set -e
export PYTHONPATH=./fedfaceid

python3 ./fedfaceid/facenet/train_triplet.py \
    --embedding_dim 512 \
    --batch_size 128 \
    --triplet_loss_margin 0.2 \
    --model_architecture "resnet34" \
    --optimizer "adam" \
    --learning_rate 0.001 \
    --output_dir "../../../outputs/resnet34_final" \
    --dataset_dir "../../../data/vggface2/train_cropped" \
    --lfw_dir "../../../data/lfw/data" \
    --dataset_csv_file "../../../data/vggface2/train_cropped_meta.csv" \
