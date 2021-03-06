# update your hyperparams
SEED=1
BATCH_SIZE=32
TEST_BATCH_SIZE=32
MOMENTUM=0.5
EPOCHS=3
USE_CUDA=True
LR=0.2
LOG_INTERVAL=2
SAVE_NAME=mnist.pt

# setup PYTHONPATH folder
EXPORT PYTHONPATH=$PYTHONPATH:${pwd}/src

# do not edit
python3 ./pytorch/run_experiment.py \
  --seed $SEED \
  --momentum $MOMENTUM \
  --batch-size $BATCH_SIZE \
  --test-batch-size $TEST_BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --log-interval $LOG_INTERVAL \
  --save-name $SAVE_NAME
