#!/bin/bash

##############################################################################################
# LEARNING RATE OBSERVATIONS
#
# 0.0001000 - GOOD INITIAL STARTING POINT - AFTER g_loss of ~18.0-17.0 switch to 0.00005
# 0.0000750 - SEEMS TO RESULT IN THE - MIN 11.x's OBSERVED
# 0.0000510 - ??
# 0.0000500 - SEEMS TO RESULT IN THE LOWEST g_loss - LOW 11.x's TO HIGH 10.x's OBSERVED 
# 0.0000490 - ??
# 0.0000375 - BEGINS DRIFTING TOWARDS LOWER id_loss AT THE EXPENSE OF recon_loss
# 0.0000250 - BEGINS DRIFTING TOWARDS LOWER id_loss AT THE EXPENSE OF recon_loss
# 0.0000125 - PREFERS id_loss ENOUGH TO MAKE RESULTING FACESWAPS WASHED OUT
##############################################################################################
export NCCL_DEBUG=INFO

python3 train.py \
    --total_epochs=1 \
    --reload_interval=1 \
    --save_epoch=1 \
    --tfrecord_shard_path="./data/kitchensink_256/" \
    --model_weights="./models/MODEL_256x256_SUPER_v14_BLOCKS2_latest" \
    --model_save_path="./models/MODEL_256x256_SUPER_v14" \
    --g_learning_rate=0.00005 \
    --num_blocks=2 \
    --train_superres \
    --randomize_shards \
    --use_emap \
    --max_batches=4000

    #--model_weights="./models/MODEL_256x256_SUPER_v6_BLOCKS2_latest" \
    #--model_weights="./models/MODEL_256x256_v18_BLOCKS2_latest" \
    #--model_weights="./models/MODEL_256x256_SUPER_v11_BLOCKS2_latest" \
    #--multi_gpu
    #--model_weights="./models/MODEL_ID_ONLY_2_BLOCKS2_latest" \
    #--tfrecord_shard_path="./data/kitchensink_256/" \
    #--tfrecord_shard_path="./data/celeba_liveportrait_256_upsampled/" \
    #--model_weights="./models/MODEL_256x256_ID_ONLY_v1_BLOCKS3_latest" \