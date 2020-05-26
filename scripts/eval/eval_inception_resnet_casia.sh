#!/usr/bin/env bash
set -e
export PYTHONPATH=./fedfaceid

python3 ./fedfaceid/facenet/evaluate_model.py \
    --model_architecture "inception_resnet_v1_casia" \
    --checkpoint_path "../../../resources/models/inception_resnet_casia_center.pt" \
    --output_dir "../../../output_dir_baseline" \
    --lfw_dir "../../../data/lfw/data" \
