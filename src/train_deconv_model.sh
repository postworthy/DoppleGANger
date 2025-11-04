#!/bin/bash

python3 train.py \
    --total_epochs=10 \
    --reload_interval=10 \
    --save_epoch=1 \
    --tfrecord_shard_path="./data/celeba_hq_256/" \
    --model_weights="./models/MODEL_256x256_v7_BLOCKS2_latest" \
    --model_save_path="./models/MODEL_256x256_v7" \
    --deconv_only
    