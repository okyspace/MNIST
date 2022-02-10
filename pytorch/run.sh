# update your hyperparams
export SEED=1
export BATCH_SIZE=32
export TEST_BATCH_SIZE=32
export MOMENTUM=0.5
export EPOCHS=3
export USE_CUDA=True
export LR=0.2
export LOG_INTERVAL=2
export SAVE_NAME=mnist.pt

# do not edit
python3 run_experiment.py \
  --seed $SEED \
  --momentum $MOMENTUM \
  --batch-size $BATCH_SIZE \
  --test-batch-size $TEST_BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --log-interval $LOG_INTERVAL \
  --save-name $SAVE_NAME
