import os
import torch
from torchvision import datasets, transforms


def get_dataloader(batch_size, is_train, to_shuffle):
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join('.', 'data'), train=is_train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=to_shuffle)    

    return data_loader
