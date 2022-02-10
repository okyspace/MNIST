"""
This is the main module to run the MNIST experiment via ClearML.
To enable this code to run you will need to do the following:
    1. Setup your environment to connect to ClearML. Follow instructions on the ClearML website
    2. Ensure config_aip.py is properly configured
    3. Install packages as required
"""
import os
from clearml import Task, Logger
from config_aip import cfg_clearml, cfg_s3
from run_training import run_training
from args_training import get_args


def main():
    """Main function to connect to ClearML to run experiment"""

    # setup clearml
    os.environ["AWS_ACCESS_KEY_ID"] = cfg_s3["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg_s3["aws_secret_access_key"]

    Task.init(
        project_name=cfg_clearml["project_name"],
        task_name=cfg_clearml["task_name"],
        output_uri=cfg_clearml["output"],
    )

    # task = Task.init(
    #     project_name=cfg_clearml["project_name"], task_name=cfg_clearml["task_name"], \
    # output_uri=cfg_clearml["output"]
    # )
    # task.set_base_docker(cfg_clearml[docker_image])
    # task.execute_remotely(queue_name=cfg_clearml[queue_name], exit_process=True)

    # train and validate
    args = get_args()
    run_training(Logger, args)


if __name__ == "__main__":
    main()
