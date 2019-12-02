#!/usr/bin/env bash
set -e
export PYTHONPATH=./fedfaceid

python3 ./fedfaceid/facenet/train_triplet.py \
    --num_triplets_train 1000000 \
    --embedding_dim 128 \
    --batch_size 96 \
    --num_workers 12 \
    --triplet_loss_margin 0.2 \
    --model_architecture "resnet50" \
    --optimizer "adam" \
    --learning_rate 0.003 \
    --output_dir "../../../outputs/resnet50_triplet_shuffled" \
    --dataset_dir "../../../data/vggface2/train_cropped" \
    --lfw_dir "../../../data/lfw/data" \
    --dataset_csv_file "../../../data/vggface2/train_cropped_meta.csv" \
