#!/bin/bash

# Define variables
BACKBONE="mit_b4"
DATASET="LUAD-HistoSeg"
# DATASET="BCSS-WSSS"
STAGE="two"
# PHASE="0"
# BATCH_SIZE="32"
LEARNING_RATE="1e-4"
LR_POLICY="polyLR"
EPOCHS="20"
TASK="offline"
STEP="0"

# Execute command with variables
# python main.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 \
# --batch_size 32 --lr $LEARNING_RATE --lr_policy $LR_POLICY --epochs $EPOCHS --task $TASK --step $STEP

# python main.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 1 \
# --batch_size 24 --lr $LEARNING_RATE --lr_policy $LR_POLICY --epochs $EPOCHS --task $TASK --step $STEP

python test.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 --task $TASK