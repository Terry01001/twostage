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
WEIGHTS=(0.0 0.3 0.4 0.3)
LOG_DIR="./logs"
FIRST_RUN=true

# run normally
# echo "Training start"
# python main.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 --lr_policy $LR_POLICY --task $TASK --step 0 --weights ${WEIGHTS[@]}


# echo "Testing"
# python test.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 --task $TASK --step 0 --weights ${WEIGHTS[@]}


# params test
for w2 in $(seq 0.2 0.05 0.3);
do
    for w3 in $(seq 0.35 0.05 $(echo "0.8 - $w2" | bc));
    do
        w4=$(echo "1.0 - $w2 - $w3" | bc)
        WEIGHTS=(0.0 $w2 $w3 $w4)

        #echo "Training start with weights ${weights[@]}"
        #python main.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 --lr_policy $LR_POLICY --task $TASK --step 0 --weights ${WEIGHTS[@]}
        if [[ $(echo "($w2 == 0.2 && $w3 == 0.35 && $w4 == 0.45) || ($w2 == 0.2 && $w3 == 0.4 && $w4 == 0.4)" | bc) -eq 1 ]]; then
            echo "Testing with weights ${WEIGHTS[@]}"
            python test.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 --task $TASK --step 0 --weights ${WEIGHTS[@]} --log_dir $LOG_DIR --first_run
            FIRST_RUN=false
        else
            echo "Training start with weights ${WEIGHTS[@]}"
            python main.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 --lr_policy $LR_POLICY --task $TASK --step 0 --weights ${WEIGHTS[@]}


        # 縮排
            if [ $FIRST_RUN = true ]; then
                echo "Testing with weights ${weights[@]}"
                python test.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 --task $TASK --step 0 --weights ${WEIGHTS[@]} --log_dir $LOG_DIR #--first_run
                #FIRST_RUN=false
            else
                echo "Testing with weights ${weights[@]}"
                python test.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 --task $TASK --step 0 --weights ${WEIGHTS[@]} --log_dir $LOG_DIR 
            fi
        fi
    done
done



# Execute command with variables
# python main.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 0 \
# --batch_size 32 --lr $LEARNING_RATE --lr_policy $LR_POLICY --epochs $EPOCHS --task $TASK --step $STEP

# python main.py --backbone $BACKBONE --dataset $DATASET --stage $STAGE --phase 1 \
# --batch_size 24 --lr $LEARNING_RATE --lr_policy $LR_POLICY --epochs $EPOCHS --task $TASK --step $STEP