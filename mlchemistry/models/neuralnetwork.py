import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """Neural Network Regression with Pytorch

    Parameters
    ----------
    hiddenlayers : tuple
        Structure of hidden layers in the neural network.
    epochs : int
        Number of full training cycles.
    convergence : dict
        Instead of using epochs, users can set a convergence criterion.
    """

    def __init__(self, hiddenlayers=(3, 3), epochs=100, convergence=None):
        super(NeuralNetwork, self).__init__()
        print('Number of hidden-layers: {}' .format(len(hiddenlayers)))
        self.hiddenlayers = hiddenlayers
        self.epochs = epochs
        self.fc1 = nn.Linear(8, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        """Forward propagation

        Parameters
        ----------
        x : list
            List of features.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

    def train(self, images):
        """Train the model

        Parameters
        ----------
        images : list
            List of images used for training the model
        """
        pass
