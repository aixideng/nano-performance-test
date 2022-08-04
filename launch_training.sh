#!/bin/bash

DATA_TASK1_PATH="./data"
PRODUCT_PATH="${DATA_TASK1_PATH}/product_catalogue-v0.3.csv.zip"
TRAIN_PATH="${DATA_TASK1_PATH}/train-v0.3.csv.zip"
MODEL_TYPE="cross_encoder"
TRAINER_MODE="nano"

RANDOM_STATE=42
TRAIN_BATCH_SIZE=32
DEV_RATIO=0.05
TEST_SET_SIZE=0.05
NUM_PROCESSES=1
NUM_EPOCHS=2
VAL_INTERVAL=0.2

python train.py \
    ${MODEL_TYPE} \
    ${TRAINER_MODE} \
    ${TRAIN_PATH} \
    ${PRODUCT_PATH} \
    --num_processes ${NUM_PROCESSES} \
    --num_epochs ${NUM_EPOCHS} \
    --val_check_interval ${VAL_INTERVAL} \
    --random_state ${RANDOM_STATE} \
    --dev_ratio ${DEV_RATIO} \
    --test_set_size ${TEST_SET_SIZE} \
    --train_batch_size ${TRAIN_BATCH_SIZE}
