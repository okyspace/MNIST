"""This module retrieves the MNIST data from pytorch."""
import os
import torch
from torchvision import datasets, transforms


def get_dataloader(batch_size, is_train, to_shuffle):
    """
    Retrives the MNIST dataset using pytorch DataLoader.

    Parameters
    ----------
    batch_size: int
        Size of one batch
    is_train: bool
        Determines if data is training or testing.
    to_shuffle: bool
        Dertermines if data should be shuffled.
    """
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            os.path.join(".", "data"),
            train=is_train,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        ),
        batch_size=batch_size,
        shuffle=to_shuffle,
    )

    return data_loader
