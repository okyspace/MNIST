@ECHO OFF
TITLE Calling run_experiment.py to run experiment via ClearML...

REM Hyperparameters variables. Tweak this to change the configuration
SET NO_CUDA=False
SET SEED=1
SET BATCH_SIZE=32
SET TEST_BATCH_SIZE=32
SET MOMENTUM=0.5
SET EPOCHS=3
SET LEARN_RATE=0.2
SET LOG_INTERVAL=100
SET SAVE_NAME=mnist.pt
SET USE_PRETRAINED=False
SET PRETRAINED_MODEL_NAME=mnist.pt

REM Setup PYTHONPATH to include main src folder
SET PYTHONPATH=%PYTHONPATH%;%~dp0%/src

REM Setup command string
SET CMD_STRING=python %~dp0%/src/pytorch/run_experiment.py^
    --no-cuda ^
    --seed %SEED% ^
    --batch-size %BATCH_SIZE% ^
    --test-batch-size %TEST_BATCH_SIZE% ^
    --momentum %MOMENTUM% ^
    --epochs %EPOCHS% ^
    --learn-rate %LEARN_RATE% ^
    --log-interval %LOG_INTERVAL% ^
    --save-name %SAVE_NAME%

IF %NO_CUDA%==True (
    SET CMD_STRING=%CMD_STRING%^
        --no-cuda
)

IF %USE_PRETRAINED%==True (
    SET CMD_STRING=%CMD_STRING%^
        --use-pretrained^
        --pretrained-model-name %PRETRAINED_MODEL_NAME%
)

REM Execute run_experiment.py to run ClearML experiment
%CMD_STRING%
