import time
import datetime
import torch
import torch.nn as nn

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
        Calculation can be run in the cpu or gpu.
    lr : float
        Learning rate.
    optimizer : object
        An optimizer class.
    activation : str
        The activation function.
    weight_decay : float
        L2 penalty.
    """

    def __init__(self, hiddenlayers=(3, 3), epochs=100, convergence=None,
                 device='cpu', lr=0.001, optimizer=None,
                 activation='relu', weight_decay=0.):
        super(NeuralNetwork, self).__init__()
        self.epochs = epochs
        self.device = device.lower()    # This is to assure we are in lowercase
        self.lr = lr
        self.optimizer = optimizer
        self.activation = activation
        self.hiddenlayers = hiddenlayers
        self.weight_decay = weight_decay
        self.backend = backend(torch)

    def forward(self, symbol, X):
        """Forward propagation

        Parameters
        ----------
        X : dict
            Dictionary with symbol keys and feature vector vector.
        """

        X = self.backend.from_numpy(X)
        X = self.linears[symbol](X)

        intercept_name = 'intercept_' + symbol
        slope_name = 'slope_' + symbol

        for name, param in self.named_parameters():
            if intercept_name == name:
                intercept = param
            elif slope_name == name:
                slope = param

        X = (slope * X) + intercept
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
            DataSet object created from the handler.
        """
        activation = {'tanh': nn.Tanh, 'relu': nn.ReLU}

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
        self.output_layer_index = {}

        for symbol in unique_element_symbols:
            linears = []

            intercept = (data.max_energy + data.min_energy) / 2.
            intercept = nn.Parameter(self.backend.from_numpy(intercept))

            slope = (data.max_energy - data.min_energy) / 2.
            slope = nn.Parameter(self.backend.from_numpy(slope))

            print(intercept, slope)
            intercept_name = 'intercept_' + symbol
            slope_name = 'slope_' + symbol

            self.register_parameter(intercept_name, intercept)
            self.register_parameter(slope_name, slope)

            for index in layers:
                # This is the input layer
                if index == 0:
                    inp_dimension = len(list(feature_space.values())[0][0][-1])
                    out_dimension = self.hiddenlayers[0]
                    _linear = nn.Linear(inp_dimension, out_dimension)
                    linears.append(_linear)
                    linears.append(activation[self.activation]())
                # This is the output layer
                elif index == len(self.hiddenlayers):
                    inp_dimension = self.hiddenlayers[index - 1]
                    out_dimension = 1
                    self.output_layer_index[symbol] = index
                    _linear = nn.Linear(inp_dimension, out_dimension)
                    linears.append(_linear)
                # These are hidden-layers
                else:
                    inp_dimension = self.hiddenlayers[index - 1]
                    out_dimension = self.hiddenlayers[index]
                    _linear = nn.Linear(inp_dimension, out_dimension)
                    linears.append(_linear)
                    linears.append(activation[self.activation]())


                #nn.init.xavier_uniform_(_linear.weight)

            # Stacking up the layers.
            linears = nn.Sequential(*linears)
            symbol_model_pair.append([symbol, linears])

        self.linears = nn.ModuleDict(symbol_model_pair)
        print(self.linears)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)#, mean=0, std=0.01)
                #nn.init.xavier_uniform_(m.weight)

        old_state_dict = {}
        for key in self.state_dict():
            old_state_dict[key] = self.state_dict()[key].clone()

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

                for symbol, feature_vector in fs:
                    atomic_energy = \
                        self.forward(symbol, feature_vector)
                    image_energy += atomic_energy

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

        new_state_dict = {}
        for key in self.state_dict():
            print(key)
            new_state_dict[key] = self.state_dict()[key].clone()

        for key in old_state_dict:
            if not (old_state_dict[key] == new_state_dict[key]).all():
                print('Diff in {}'.format(key))
            else:
                print('They are the same shit')

        print()
        for symbol in unique_element_symbols:
            model = self.linears[symbol]
            print('Optimized parameters for {} symbol' .format(symbol))
            for param in model.parameters():
                print(param)

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
        self.optimizer.zero_grad()  # clear previous gradients
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(outputs, targets))

        # L2 regularization does not seem to be the same as weight_decay.
        # See: https://arxiv.org/abs/1711.05101
        l2 = 0.

        for symbol in self.linears.keys():
            model = self.linears[symbol]
            for param in model.parameters():
                l2 += l2 + param.norm(2)

        loss = loss + (l2 * 1e-6)
        loss.backward()
        self.optimizer.step()
        return loss
