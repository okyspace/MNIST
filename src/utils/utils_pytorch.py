"""This module contains utility codes for loading/saving pytorch models."""
import torch

from pytorch.network import MNISTNet


def load_model(model_name):
    """
    Loads model weights from the given model name into the default MNISTNet model.

    Parameters
    ----------
    model_name: str
        Model name to load the weights.
    """
    model = MNISTNet()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_name)

    if 0 == len(pretrained_dict):
        print(f"Could not load model from {model_name}")
        return

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def save_model(model, model_name):
    """Saves model to model name

    Parameters
    ----------
    model: object
        Actual model to save.
    model_name: str
        Model save name.
    """
    torch.save(model.state_dict(), model_name)
