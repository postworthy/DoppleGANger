#!/bin/bash

python3 train.py \
    --train_superres \
    --total_epochs=1000 \
    --reload_interval=10 \
    --model_save_path="./models/superres_weights"