#!/usr/bin/env bash
set -e
export PYTHONPATH=./fedfaceid

python3 ./fedfaceid/facenet/evaluate_model.py \
    --checkpoint_path "../../../resources/models/resnet34_triplet.pt" \
    --output_dir "../../../output_dir_baseline" \
    --lfw_dir "../../../data/lfw/data" \
