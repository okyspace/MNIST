import os
from clearml import Task, Logger
from config_aip import cfg_clearml, cfg_s3
from run_training import run_training


def main():
	# setup clearml
	os.environ["AWS_ACCESS_KEY_ID"] = cfg_s3['aws_access_key_id']
	os.environ["AWS_SECRET_ACCESS_KEY"] = cfg_s3['aws_secret_access_key']
	task = Task.init(project_name=cfg_clearml['project_name'], task_name=cfg_clearml['task_name'], output_uri=cfg_clearml['output'])
	# task.set_base_docker(cfg_clearml[docker_image])
	# task.execute_remotely(queue_name=cfg_clearml[queue_name], exit_process=True)

	# train and validate
	run_training(Logger)


if __name__ == '__main__':
    main()
