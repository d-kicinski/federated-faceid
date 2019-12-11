#!/usr/bin/env bash
set -e
export PYTHONPATH=.

checkpoint_dir="../../../resources/models/resnet50_triplet.pt"
lfw_dir="../../../data/lfw/data"
nv_dir="../../../data/nv"

python3 ./fedfaceid/train.py \
    --model_architecture "resnet50" \
    --dataset_local_dir ${lfw_dir}/train \
    --dataset_local_csv_file ${lfw_dir}/train_meta.csv\
    --dataset_remote_dir ${nv_dir}/nv-gen_cropped \
    --dataset_remote_csv_file ${nv_dir}/nv-gen_train_meta.csv \
    --checkpoint_path ${checkpoint_dir} \
    --learning_rate 0.001 \
    --embedding_dim 512 \
    --batch_size 64 \
    --triplet_loss_margin 0.2 \
    --output_dir "../../../outputs/resnet50_triplet_finetuned" \
    --lfw_dir "../../../data/lfw/data"
