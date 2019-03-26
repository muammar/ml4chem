import datetime
import time
import torch

from collections import OrderedDict
from mlchem.models.loss import RMSELossAE
from mlchem.utils import convert_elapsed_time

torch.set_printoptions(precision=10)


class AutoEncoder(torch.nn.Module):
    """A Vanilla autoencoder with Pytorch


    AutoEncoders are very intersting models where usually the input is
    reconstructed (input equals output). These models are able to learn data
    codings in an unsupervised manner. They are composed by an encoder that
    takes an input and concentrate (encodes) the information in a lower/larger
    dimension (latent space). Subsequently, a decoder takes the latent space
    and tries to resconstruct the input. However, when the output is not equal
    to the input, the model learns how to 'translate' input into output e.g.
    image coloring.

    This module uses autoencoders for pipelines in chemistry.

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

        if purpose == 'training':
            print()
            print('Model Training')
            print('==============')
            print('Model name: {}.'.format(self.name()))
            print('Structure of Neural Net: {}'
                  .format('(input, ' + str(self.hiddenlayers)[1:-1] +
                          ', output)'))

        unique_element_symbols = data.unique_element_symbols[purpose]

        symbol_encoder_pair = []
        symbol_decoder_pair = []

        for symbol in unique_element_symbols:
            encoder = []
            encoder_layers = self.hiddenlayers['encoder']
            decoder = []
            decoder_layers = self.hiddenlayers['decoder']

            """
            Encoder
            """
            # The first encoder layer for symbol
            out_dimension = encoder_layers[0]
            _encoder = torch.nn.Linear(input_dimension,
                                       out_dimension)
            encoder.append(_encoder)
            encoder.append(activation[self.activation]())

            for inp_dim, out_dim in zip(encoder_layers, encoder_layers[1:]):
                _encoder = torch.nn.Linear(inp_dim, out_dim)
                encoder.append(_encoder)
                encoder.append(activation[self.activation]())

            # Stacking up the layers.
            encoder = torch.nn.Sequential(*encoder)
            symbol_encoder_pair.append([symbol, encoder])

            """
            Decoder
            """
            for inp_dim, out_dim in zip(decoder_layers, decoder_layers[1:]):
                _decoder = torch.nn.Linear(inp_dim, out_dim)
                decoder.append(_decoder)
                decoder.append(activation[self.activation]())

            # The last decoder layer for symbol
            inp_dim = out_dim
            _decoder = torch.nn.Linear(inp_dim,
                                       output_dimension)
            decoder.append(_decoder)
            # According to this video https://youtu.be/xTU79Zs4XKY?t=416
            # real numbered inputs need no activation function in the output
            # layer
            # decoder.append(activation[self.activation]())

            # Stacking up the layers.
            decoder = torch.nn.Sequential(*decoder)
            symbol_decoder_pair.append([symbol, decoder])

        self.encoders = torch.nn.ModuleDict(symbol_encoder_pair)
        self.decoders = torch.nn.ModuleDict(symbol_decoder_pair)
        print(self.encoders)
        print(self.decoders)

        if purpose == 'training':
            # Iterate over all modules and just intialize those that are
            # a linear layer.
            print('Initialization of weights.')
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    # nn.init.normal_(m.weight)   # , mean=0, std=0.01)
                    torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, X):
        """Forward propagation

        This method takes an input and applies encoder and decoder layers.

        Parameters
        ----------
        X : list
            List of inputs either raw or in the feature space.

        Returns
        -------
        outputs : tensor
            Decoded latent vector.
        """

        outputs = []
        for hash, image in X.items():
            for symbol, x in image:
                latent_vector = self.encoders[symbol](x)
                decoder = self.decoders[symbol](latent_vector)
                outputs.append(decoder)
        outputs = torch.stack(outputs)
        return outputs

    def get_latent_space(self, X, svm=False):
        """Get latent space for training MLChem

        This method takes an input and use the encoder to return latent space
        in the structure needed for training MLChem.

        Parameters
        ----------
        X : list
            List of inputs either raw or in the feature space.
        svm : bool
            Whether or not these latent vectors are going to be used for kernel
            methods.

        Returns
        -------
        latent_space : dict
            Latent space with structure: {'hash': [('H', [latent_vector]]}
        """

        latent_space = OrderedDict()

        for hash, image in X.items():
            latent_space[hash] = []
            for symbol, x in image:
                latent_vector = self.encoders[symbol](x)

                if svm:
                    _latent_vector = latent_vector.detach().numpy()
                else:
                    _latent_vector = latent_vector.detach()

                latent_space[hash].append((symbol, _latent_vector))

        return latent_space


def train(inputs, targets, model=None, data=None, optimizer=None, lr=None,
          weight_decay=None, regularization=None, epochs=100, convergence=None,
          lossfxn=None):
    """Train the model

    Parameters
    ----------
    inputs : dict
        Dictionary with hashed feature space.
    epochs : int
        Number of full training cycles.
    targets : list
        The expected values that the model has to learn aka y.
    model : object
        The NeuralNetwork class.
    data : object
        DataSet object created from the handler.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay passed to the optimizer. Default is 0.
    regularization : float
        This is the L2 regularization. It is not the same as weight decay.
    convergence : dict
        Instead of using epochs, users can set a convergence criterion.
            >>> convergence ={'rmse': 5e-3}

    lossfxn : obj
        A loss function object.
    """

    # old_state_dict = {}

    # for key in model.state_dict():
    #     old_state_dict[key] = model.state_dict()[key].clone()

    targets = torch.tensor(targets, requires_grad=False, dtype=torch.float)

    # Define optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)

    print()
    print('{:6s} {:19s} {:8s}'.format('Epoch', 'Time Stamp', 'Loss'))
    print('{:6s} {:19s} {:8s}'.format('------',
                                      '-------------------', '---------'))
    initial_time = time.time()

    _loss = []
    _rmse = []
    epoch = 0

    while True:
        epoch += 1
        outputs = model(inputs)

        if lossfxn is None:
            loss, rmse = RMSELossAE(outputs, targets, optimizer)
        else:
            raise('I do not know what to do')

        _loss.append(loss)
        _rmse.append(rmse)

        ts = time.time()
        ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d '
                                                          '%H:%M:%S')
        print('{:6d} {} {:8e} {:8f}' .format(epoch, ts, loss, rmse))

        if convergence is None and epoch == epochs:
            break
        elif (convergence is not None and rmse < convergence['rmse']):
            break

    training_time = time.time() - initial_time

    h, m, s = convert_elapsed_time(training_time)
    print('Training finished in {} hours {} minutes {:.2f} seconds.'
          .format(h, m, s))
    print('outputs')
    print(outputs)
    print('targets')
    print(targets)
