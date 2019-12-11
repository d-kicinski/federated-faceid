#!/usr/bin/env bash
set -e
export PYTHONPATH=.

output_dir="../../../eval/resnet50_triplet"
checkpoint_dir="../../../resources/models/"
model=resnet50_triplet.pt

mkdir -p ${output_dir}

python3 ./facenet/evaluate_model.py \
    --checkpoint_path ${checkpoint_dir}/${model} \
    --output_dir ${output_dir} \
    --lfw_dir "../../../data/lfw/data"

