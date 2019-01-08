import torch
import torch.nn as nn
import torch.nn.functional as F

from mlchemistry.backends.operations import BackendOperations as backend

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
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 3)
        self.fc3 = nn.Linear(3, 1)
        self.backend = backend(torch)
        self.device = 'cpu'

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
        return x

    def train(self, feature_space, targets):
        """Train the model

        Parameters
        ----------
        feature_space : dict
            Dictionary with hashed feature space.
        targets : list
            The expected values that the model has to learn aka y.
        """


        targets = self.backend.from_numpy(targets)

        # Definition of weights
        w1 = torch.randn(8, 8, device=self.device, requires_grad=False)
        w2 = torch.randn(8, 3, device=self.device, requires_grad=True)
        w3 = torch.randn(3, 1, device=self.device, requires_grad=True)

        # Define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        for epoch in range(self.epochs):
            print(epoch)
            outputs = []

            for hash, fs in feature_space.items():
                image_energy = 0.

                tensorial = []
                for symbol, feature_vector in fs:
                    print(symbol, feature_vector)
                    atomic_energy = self.forward(self.backend.from_numpy(feature_vector))
                    tensorial.append(feature_vector)
                    print('atomic_energy', atomic_energy)
                    image_energy += atomic_energy

                tensorial_space = self.backend.from_numpy(tensorial)
                print('tensor shape', tensorial_space.shape)
                print('Energy for hash {} is {} with tensors'
                        .format(hash, self.forward(tensorial_space).sum()))
                print('Energy for hash {} is {}' .format(hash, image_energy))
                outputs.append(image_energy)
            print('outputs')
            print(outputs)
            print('targets')
            print(targets)
            outputs = torch.stack(outputs)

            criterion = nn.MSELoss()
            loss = torch.sqrt(criterion(outputs, targets))
            print('Loss function', loss)
            optimizer.zero_grad()  # clear previous gradients
            loss.backward()
            optimizer.step()
