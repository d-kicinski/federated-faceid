#!/usr/bin/env bash


#PYTHONPATH=. python ./scripts/python/data/align_faces.py \
#     --data_dir "../../../data/vggface2/train" \
#     --output_dir "../../../data/vggface2/train_cropped" \
#     --image_size 182 \
#     --margin 44 \
#     --num_workers 2

#PYTHONPATH=. python ./scripts/python/data/align_faces.py \
#    --data_dir "../../../data/vggface2/test" \
#    --output_dir "../../../data/vggface2/test_cropped" \
#    --image_size 182 \
#    --num_workers 10 \
#    --margin 44

PYTHONPATH=. python ./scripts/python/data/align_faces.py \
    --data_dir "../../../data/vggface2/test" \
    --output_dir "../../../data/vggface2/test_eval_cropped_for_test" \
    --num_workers 10 \
    --image_size 160 \
    --margin 32
