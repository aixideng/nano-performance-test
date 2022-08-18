#!/bin/bash

DATA_TASK1_PATH="../data/task1"
TRAIN_PATH="${DATA_TASK1_PATH}/train-v0.3.csv.zip"
PRODUCT_PATH="${DATA_TASK1_PATH}/product_catalogue-v0.3.csv.zip"
MODEL_TYPE="GCN"

NUM_PROCESSES=4
NUM_EPOCHS=2
DEV_RATIO=0.01
TEST_SET_SIZE=0.05
VAL_INTERVAL=0.2
RANDOM_STATE=42
BATCH_SIZE=32
NUM_NEIGHBORS=1

python train.py \
    --nano \
    ${MODEL_TYPE} \
    ${TRAIN_PATH} \
    ${PRODUCT_PATH} \
    --num_processes ${NUM_PROCESSES} \
    --num_epochs ${NUM_EPOCHS} \
    --dev_ratio ${DEV_RATIO} \
    --test_set_size ${TEST_SET_SIZE} \
    --val_check_interval ${VAL_INTERVAL} \
    --random_state ${RANDOM_STATE} \
    --train_batch_size ${BATCH_SIZE} \
    --eval_batch_size ${BATCH_SIZE} \
    --num_neighbors ${NUM_NEIGHBORS}
