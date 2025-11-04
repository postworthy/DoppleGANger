#!/bin/bash

python3 train_self.py \
    --total_epochs=1 \
    --reload_interval=1 \
    --tfrecord_shard_path="./data/kitchensink_256/" \
    --model_weights="./models/MODEL_256x256_v11_BLOCKS2_latest" \
    --model_save_path="./models/MODEL_256x256_SELF_SUPER" \