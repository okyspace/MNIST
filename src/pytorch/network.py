"""This module contains the default pytorch neural network model for MNIST"""

import torch.nn as nn
import torch.nn.functional as fn


class MNISTNet(nn.Module):
    """
    Neural network model for classifying MNIST dataset
    '''
    Attributes
    ----------
    conv1: Conv2d
        1st 2D convolution layer
    conv2: Conv2d
        2nd 2D convolution layer
    fc1: Linear
        1st linear layer
    fc2: Linear
        2nd linear layer

    Methods
    -------
    forward(x=object)
        Constructs the neural network.
    """

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """Constructs the neural network.
        Parameters
        ----------
        x: object
            Holds the built network construct
        """
        x = fn.relu(self.conv1(x))
        x = fn.max_pool2d(x, 2, 2)
        x = fn.relu(self.conv2(x))
        x = fn.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = fn.relu(self.fc1(x))
        x = self.fc2(x)
        return fn.log_softmax(x, dim=1)
