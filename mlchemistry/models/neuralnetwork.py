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
    optimizer : object
        An optimizer class.
    activation_function : str
        The activation function.
    weight_decay : float
        L2 penalty.
    """

    def __init__(self, hiddenlayers=(3, 3), epochs=100, convergence=None,
                 device='cpu', lr=0.001, optimizer=None,
                 activation_function='tanh', weight_decay=1e-5):
        super(NeuralNetwork, self).__init__()
        self.epochs = epochs
        self.device = device.lower()    # This is to assure we are in lowercase
        self.lr = lr
        self.optimizer = optimizer
        self.activation_function = activation_function
        self.hiddenlayers = hiddenlayers
        self.weight_decay = weight_decay

    def forward(self, feature_vector):
        """Forward propagation

        Parameters
        ----------
        X : dict
            Dictionary with symbol keys and feature vector vector.
        """
        activation_function = {'tanh': torch.tanh, 'relu': F.relu}

        symbol, X = feature_vector

        X = self.backend.from_numpy(X)

        for i, l in enumerate(self.linears[symbol]):
            if i != self.out_layer_indices[symbol]:
                X = activation_function[self.activation_function](l(X))
            else:
                X = l(X)
        return X

    def train(self, feature_space, targets, data=None):
        """Train the model

        Parameters
        ----------
        feature_space : dict
            Dictionary with hashed feature space.
        targets : list
            The expected values that the model has to learn aka y.
        data : object
            Data object created from the handler.
        """

        print()
        print('Model Training')
        print('==============')
        print('Number of hidden-layers: {}' .format(len(self.hiddenlayers)))
        print('Structure of Neural Net: {}' .format('(input, ' +
                                                    str(self.hiddenlayers)[1:-1]
                                                    + ', output)'))
        layers = range(len(self.hiddenlayers) + 1)
        unique_element_symbols = data.unique_element_symbols['trainingset']


        symbol_model_pair = []
        self.out_layer_indices = {}

        for symbol in unique_element_symbols:
            linears = []
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
                    self.out_layer_indices[symbol] = index
                # These are hidden-layers
                else:
                    inp_dimension = self.hiddenlayers[index - 1]
                    out_dimension = self.hiddenlayers[index]

                _linear = nn.Linear(inp_dimension, out_dimension)
                nn.init.xavier_uniform(_linear.weight)
                linears.append(_linear)

            # Stacking up the layers.
            linears = nn.ModuleList(linears)
            symbol_model_pair.append([symbol, linears])

        self.linears = nn.ModuleDict(symbol_model_pair)
        print(self.linears)

        self.backend = backend(torch)
        targets = self.backend.from_numpy(targets)

        # Define optimizer

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                              weight_decay=self.weight_decay)

        print()
        print('{:6s} {:19s} {:8s}'.format('Epoch', 'Time Stamp', 'Loss'))
        print('{:6s} {:19s} {:8s}'.format('------',
                                          '-------------------', '---------'))
        initial_time = time.time()

        for epoch in range(self.epochs):
            outputs = []

            for hash, fs in feature_space.items():
                image_energy = 0.
                tensorial = []

                for feature_vector in fs:
                    atomic_energy = \
                        self.forward(feature_vector)
                    #tensorial.append(feature_vector)
                    image_energy += atomic_energy

                #tensorial_space = self.backend.from_numpy(tensorial)
                outputs.append(image_energy)

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

        print()
        for symbol in unique_element_symbols:
            model = self.linears[symbol]
            print('Parameters for {} symbol' .format(symbol))
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
        self.optimizer.zero_grad()  # clear previous gradients
        loss.backward()
        self.optimizer.step()
        return loss
