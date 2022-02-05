import torch


def load_model(model, model_name):
	model_dict = model.state_dict()
	pretrained_dict = torch.load(model_name)

	if 0 == len(pretrained_dict):
		print('Could not load model from {}'.format(model_name))
		return

	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	return model


def save_model(model, model_name):
	torch.save(model.state_dict(), model_name)


def write_to_tensorboard(writer, key, val):
	writer.add_scalars(key, val)
