import torch

class NeuralNetwork(torch.nn.Module):
    """Neural Network Regression with Pytorch

    Parameters
    ----------
    hiddenlayers : tuple
        Structure of hidden layers in the neural network.
    epochs : int
        Number of full training cycles.
    """
    def __init__(self, hiddenlayers=(3, 2), epochs=None):
        if isinstance(hiddenlayers, int):
            hiddenlayers = (hiddenlayers)
        print('Number of hidden-layers: {}' .format(len(hiddenlayers)))

    def forward(self):
        """Forward propagation"""
        pass
