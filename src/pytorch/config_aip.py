"""Configuration settings for training the model"""

cfg_clearml = {
    "project_name": "MNIST_WFJ",
    "task_name": "MNIST_Training",
    "queue_name": "queue-8gb-ram",
    "output": True,
    "docker_image": "quay.io/dhdevspace/mnist:pytorch",
    "git_user": "Sable021",
    "git_pw": "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",
}

cfg_s3 = {
    "url": "http://192.168.1.110:9000",
    "aws_access_key_id": "admin",
    "aws_secret_access_key": "password",
    "cert": "/etc/ssl/certs/ca-certificates.crt",
    "data_bucket": "public-data",
    "model_bucket": "clearml-models",
}
