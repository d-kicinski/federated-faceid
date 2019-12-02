#!/usr/bin/env bash
set -e
export PYTHONPATH=./fedfaceid

python3 ./fedfaceid/facenet/train_triplet.py \
    --num_triplets_train 1000000 \
    --embedding_dim 128 \
    --batch_size 96 \
    --triplet_loss_margin 0.2 \
    --model_architecture "resnet50" \
    --optimizer "adam" \
    --learning_rate 0.003 \
    --output_dir "../../../output_dir/inception_v1_static_triplets" \
    --dataset_dir "../../../data/vggface2/train_cropped" \
    --lfw_dir "../../../data/lfw/data" \
    --dataset_csv_file "../../../data/vggface2/train_cropped_meta.csv" \
    --training_triplets_path "../../../data/vggface2/train_triplets_1000000.npy"
