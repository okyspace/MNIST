# update your hyperparams
SEED=1
BATCH_SIZE=2
TEST_BATCH_SIZE=2
MOMENTUM=0.5
EPOCHS=2
USE_CUDA=True
LR=0.2
LOG_INTERVAL=10
SAVE_NAME=mnist.pt

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
