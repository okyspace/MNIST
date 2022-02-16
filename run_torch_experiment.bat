@ECHO OFF
TITLE Calling run_experiment.py to run experiment via ClearML...

REM Hyperparameters variables
SET SEED=1
SET BATCH_SIZE=16
SET TEST_BATCH_SIZE=32
SET MOMENTUM=0.5
SET EPOCHS=3
SET USE_CUDA=True
SET LR=0.2
SET LOG_INTERVAL=2
SET SAVE_NAME=mnist.pt

REM Setup PYTHONPATH to include main src folder
SET PYTHONPATH=%PYTHONPATH%;%~dp0%/src

REM Execute run_experiment.py to run experiment
python %~dp0%/src/pytorch/run_experiment.py^
 --seed %SEED% ^
 --batch-size %BATCH_SIZE% ^
 --test-batch-size %TEST_BATCH_SIZE% ^
 --momentum %MOMENTUM% ^
 --epochs %EPOCHS% ^
 --lr %LR% ^
 --log-interval %LOG_INTERVAL% ^
 --save-name %SAVE_NAME%
 