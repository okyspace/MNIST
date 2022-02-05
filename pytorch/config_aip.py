cfg_clearml = {
	'project_name': "MNIST",
	'task_name': "MNIST_Training",
	'queue_name': "queue-8gb-ram",
	'output': "s3://192.168.1.110:9000/clearml-models",
	'docker_image': 'quay.io/dhdevspace/mnist:pytorch',
        'git_user': 'okyspace',
        'git_pw': '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'
}

cfg_s3 = {
	'url': 'http://192.168.1.110:9000',
	'aws_access_key_id': 'admin',
	'aws_secret_access_key': 'password',
	'cert': '/etc/ssl/certs/ca-certificates.crt',
	'data_bucket': 'public-data',
	'model_bucket': 'clearml-models',
}
