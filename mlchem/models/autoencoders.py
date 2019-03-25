import time
import datetime
import torch

from mlchem.backends.operations import BackendOperations as backend
from mlchem.data.visualization import parity
from mlchem.models.neuralnetwork import NeuralNetwork
from mlchem.utils import convert_elapsed_time

torch.set_printoptions(precision=10)


class AutoEncoder(NeuralNetwork, torch.nn.Module):
    """A Vanilla autoencoder with Pytorch

    Parameters
    ----------
    hiddenlayers : dict
        Dictionary with encoder, and decoder layers in the Auto Encoder.
    activation : str
        The activation function.

    Notes
    -----
    When defining the hiddenlayers keyword argument, input and output
    dimensions are automatically determined. For example, suppose you have an
    input data point with 10 dimensions and you want to autoencode with
    targets having 14 dimensions, and a latent space with 4 dimensions and just
    one layer with 5 nodes between input/latent and latent/output. Your
    hiddenlayers dictionary would look like this:

        >>> hiddenlayers = {'encoder': (5, 4), 'decoder': (4, 5)}

    That would generate an autoencoder with topology (10, 5, 4, 4, 5, 14).
    """

    NAME = 'AutoEncoder'

    @classmethod
    def name(cls):
        """Returns name of class"""

        return cls.NAME

    def __init__(self, hiddenlayers=None, activation='relu'):
        super(AutoEncoder, self).__init__()
        self.hiddenlayers = hiddenlayers
        self.activation = activation

    def prepare_model(self, input_dimension, output_dimension, data=None,
            purpose='training'):
        """Prepare the model

        Parameters
        ----------
        input_dimension : int
            Input's dimension.
        output_dimension : int
            Output's dimension.
        data : object
            DataSet object created from the handler.
        purpose : str
            Purpose of this model: 'training', 'inference'.
        """
        activation = {'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU,
                      'celu': torch.nn.CELU}

        #hl = len(self.hiddenlayers)
        if purpose == 'training':
            print()
            print('Model Training')
            print('==============')
            print('Model name: {}.'.format(self.name()))
            #print('Number of hidden-layers: {}' .format(hl))
            print('Structure of Neural Net: {}'
                  .format('(input, ' + str(self.hiddenlayers)[1:-1] +
                          ', output)'))

        unique_element_symbols = data.unique_element_symbols[purpose]

        symbol_encoder_pair = []
        symbol_decoder_pair = []
        self.output_layer_index = {}

        for symbol in unique_element_symbols:
            encoder = []
            encoder_layers = self.hiddenlayers['encoder']

            # The first encoder layer for symbol
            out_dimension = encoder_layers[0]
            _encoder = torch.nn.Linear(input_dimension,
                                       out_dimension)
            encoder.append(_encoder)
            encoder.append(activation[self.activation]())

            inp_dimension = out_dimension

            for nodes in encoder_layers:
                _encoder = torch.nn.Linear(inp_dimension, nodes)
                encoder.append(_encoder)
                encoder.append(activation[self.activation]())
                inp_dimension = nodes

            # Stacking up the layers.
            encoder = torch.nn.Sequential(*encoder)
            symbol_encoder_pair.append([symbol, encoder])

        self.encoders = torch.nn.ModuleDict(symbol_encoder_pair)
        print(self.encoders)

        if purpose == 'training':
            print(self.linears)
            # Iterate over all modules and just intialize those that are
            # a linear layer.
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    # nn.init.normal_(m.weight)   # , mean=0, std=0.01)
                    torch.nn.init.xavier_uniform_(m.weight)

    #def forward(self, X):
    #    """Forward propagation

    #    This is forward propagation and it returns the atomic energy.

    #    Parameters
    #    ----------
    #    X : list
    #        List of inputs in the feature space.

    #    Returns
    #    -------
    #    outputs : tensor
    #        A list of tensors with energies per image.
    #    """

    #    outputs = []

    #    for hash in X:
    #        image = X[hash]
    #        atomic_energies = []

    #        for symbol, x in image:
    #            x = self.linears[symbol](x)

    #            intercept_name = 'intercept_' + symbol
    #            slope_name = 'slope_' + symbol
    #            slope = getattr(self, slope_name)
    #            intercept = getattr(self, intercept_name)

    #            x = (slope * x) + intercept
    #            atomic_energies.append(x)

    #        atomic_energies = torch.cat(atomic_energies)
    #        image_energy = torch.sum(atomic_energies)
    #        outputs.append(image_energy)
    #    outputs = torch.stack(outputs)
    #    return outputs


#def train(inputs, targets, model=None, data=None, optimizer=None, lr=None,
#          weight_decay=None, regularization=None, epochs=100, convergence=None,
#          lossfxn=None):
#    """Train the model
#
#    Parameters
#    ----------
#    inputs : dict
#        Dictionary with hashed feature space.
#    epochs : int
#        Number of full training cycles.
#    targets : list
#        The expected values that the model has to learn aka y.
#    model : object
#        The NeuralNetwork class.
#    data : object
#        DataSet object created from the handler.
#    lr : float
#        Learning rate.
#    weight_decay : float
#        Weight decay passed to the optimizer. Default is 0.
#    regularization : float
#        This is the L2 regularization. It is not the same as weight decay.
#    convergence : dict
#        Instead of using epochs, users can set a convergence criterion.
#    lossfxn : obj
#        A loss function object.
#    """
#
#    # old_state_dict = {}
#
#    # for key in model.state_dict():
#    #     old_state_dict[key] = model.state_dict()[key].clone()
#
#    targets = torch.tensor(targets, requires_grad=False)
#
#    # Define optimizer
#    if optimizer is None:
#        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
#                                     weight_decay=weight_decay)
#
#    print()
#    print('{:6s} {:19s} {:8s}'.format('Epoch', 'Time Stamp', 'Loss'))
#    print('{:6s} {:19s} {:8s}'.format('------',
#                                      '-------------------', '---------'))
#    initial_time = time.time()
#
#    _loss = []
#    _rmse = []
#    epoch = 0
#
#    while True:
#        epoch += 1
#
#        outputs = model(inputs)
#
#        if lossfxn is None:
#            loss, rmse = loss_function(outputs, targets, optimizer, data)
#        else:
#            raise('I do not know what to do')
#
#        _loss.append(loss)
#        _rmse.append(rmse)
#
#        ts = time.time()
#        ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d '
#                                                          '%H:%M:%S')
#        print('{:6d} {} {:8e} {:8f}' .format(epoch, ts, loss, rmse))
#
#        if convergence is None and epoch == epochs:
#            break
#        elif (convergence is not None and rmse < convergence['energy']):
#            break
#
#    training_time = time.time() - initial_time
#
#    h, m, s = convert_elapsed_time(training_time)
#    print('Training finished in {} hours {} minutes {:.2f} seconds.'
#          .format(h, m, s))
#    print('outputs')
#    print(outputs)
#    print('targets')
#    print(targets)
#
#    import matplotlib.pyplot as plt
#    plt.plot(list(range(epoch)), _loss, label='loss')
#    plt.plot(list(range(epoch)), _rmse, label='rmse/atom')
#    plt.legend(loc='upper left')
#    plt.show()
#
#    parity(outputs.detach().numpy(), targets.detach().numpy())
