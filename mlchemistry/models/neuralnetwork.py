import time
import datetime
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
    device : str
        Calculation can be run in the CPU or GPU.
    lr : float
        Learning rate.
    """

    def __init__(self, hiddenlayers=(3, 3), epochs=100, convergence=None,
                 device='cpu', lr=0.001, optimizer=None):
        super(NeuralNetwork, self).__init__()
        self.epochs = epochs
        self.device = device.lower()    # This is to assure we are in lowercase
        self.lr = lr
        self.optimizer = optimizer

        self.hiddenlayers = hiddenlayers

    def forward(self, X):
        """Forward propagation

        Parameters
        ----------
        X : list
            List of features.
        """
        for i, l in enumerate(self.linears):
            if i != self.out_layer_index:
                X = F.relu(l(X))
            else:
                X = l(X)
        return X

    def train(self, feature_space, targets):
        """Train the model

        Parameters
        ----------
        feature_space : dict
            Dictionary with hashed feature space.
        targets : list
            The expected values that the model has to learn aka y.
        """

        print()
        print('Model Training')
        print('==============')
        print('Number of hidden-layers: {}' .format(len(self.hiddenlayers)))
        print('Structure of Neural Net: {}' .format('(input, ' +
                                                    str(self.hiddenlayers)[1:-1]
                                                    + ', output)'))
        linears = []
        layers = range(len(self.hiddenlayers) + 1)

        for index in layers:
            # This is the input layer
            if index == 0:
                inp_dimension = len(list(feature_space.values())[0][0][-1])
                out_dimension = self.hiddenlayers[0]
            # This is the output layer
            elif index == len(self.hiddenlayers):
                inp_dimension = self.hiddenlayers[index - 1]
                out_dimension = 1
                self.out_layer_index = index
            # These are hidden-layers
            else:
                inp_dimension = self.hiddenlayers[index - 1]
                out_dimension = self.hiddenlayers[index]

            linears.append(nn.Linear(inp_dimension, out_dimension))

        # Stacking up the layers.
        self.linears = nn.ModuleList(linears)
        self.backend = backend(torch)
        targets = self.backend.from_numpy(targets)

        # Definition of weights
        # w1 = torch.randn(8, 3, device=self.device, requires_grad=True)
        # w2 = torch.randn(8, 3, device=self.device, requires_grad=True)
        # w3 = torch.randn(3, 1, device=self.device, requires_grad=True)

        # Define optimizer

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        print()
        print('{:6s} {:19s} {:8s}'.format('Epoch', 'Time Stamp','Loss'))
        print('{:6s} {:19s} {:8s}'.format('------',
                                          '-------------------','---------'))
        initial_time = time.time()

        for epoch in range(self.epochs):
            outputs = []
            self.optimizer.zero_grad()  # clear previous gradients

            for hash, fs in feature_space.items():
                image_energy = 0.

                tensorial = []
                for symbol, feature_vector in fs:
                    #print(symbol, feature_vector)
                    atomic_energy = \
                        self.forward(self.backend.from_numpy(feature_vector))
                    tensorial.append(feature_vector)
                    #print('atomic_energy', atomic_energy)
                    image_energy += atomic_energy

                tensorial_space = self.backend.from_numpy(tensorial)
                #print('tensor shape', tensorial_space.shape)
                #print('Energy for hash {} is {} with tensors'
                      #.format(hash, self.forward(tensorial_space).sum()))
                #print('Energy for hash {} is {}' .format(hash, image_energy))
                outputs.append(image_energy)
            #print('outputs')
            #print(outputs)
            #print('targets')
            #print(targets)
            outputs = torch.stack(outputs)
            loss = self.get_loss(outputs, targets)

            ts = time.time()
            ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d '
                                                              '%H:%M:%S')
            print('{:6d} {} {:8f}' .format(epoch, ts, loss))

        training_time = time.time() - initial_time
        print('outputs')
        print(outputs)
        print('targets')
        print(targets)
        for model in self.linears:
            print(list(model.parameters()))

    def get_loss(self, outputs, targets):
        """Get loss function value

        Parameters
        ----------
        outputs : list
            List or tensor with outputs from the Neural Networks.
        targets : list
            List or tensor with expected values.


        Returns
        -------
        loss : float
            Current value of loss function.
        """

        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(outputs, targets))
        loss.backward()
        self.optimizer.step()
        return loss
