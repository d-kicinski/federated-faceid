#!/usr/bin/env bash
set -e
export PYTHONPATH=./fedfaceid

python3 ./fedfaceid/facenet/train_triplet.py \
    --num_triplets_train 100000 \
    --embedding_dim 512 \
    --model_architecture "inceptionresnetv1" \
    --optimizer "adam" \
    --learning_rate 0.001 \
    --output_dir "../../../output_dir_inception" \
    --dataset_dir "../../../data/vggface2/train_cropped" \
    --lfw_dir "../../../data/lfw/data" \
    --dataset_csv_file "../../../data/vggface2/train_cropped_meta.csv" \
    --training_triplets_path "../../../data/vggface2/train_triplets_100000.npy"
