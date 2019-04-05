import dask
import datetime
import logging
import time
import torch

import numpy as np
from collections import OrderedDict
from mlchem.models.loss import MSELossAE
from mlchem.utils import convert_elapsed_time, get_chunks

torch.set_printoptions(precision=10)
logger = logging.getLogger()


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
            logger.info('Model Training')
            logger.info('==============')
            logger.info('Model name: {}.'.format(self.name()))
            logger.info('Structure of Autoencoder: {}'
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
        logger.info(self.encoders)
        logger.info(self.decoders)

        if purpose == 'training':
            # Iterate over all modules and just intialize those that are
            # a linear layer.
            logger.warning('Initialization of weights with Xavier Uniform by '
                           'default.')
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
          lossfxn=None, batch_size=None, device='cpu'):
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
    batch_size : int
        Number of data points per batch to use for training. Default is None.
    device : str
        Calculation can be run in the cpu or cuda (gpu).
    """

    # old_state_dict = {}

    # for key in model.state_dict():
    #     old_state_dict[key] = model.state_dict()[key].clone()

    if device == 'cuda':
        pass
        """
        logger.info('Moving data to CUDA...')

        targets = targets.cuda()
        _inputs = OrderedDict()

        for hash, f in inputs.items():
            _inputs[hash] = []
            for features in f:
                symbol, vector = features
                _inputs[hash].append((symbol, vector.cuda()))

        del inputs
        inputs = _inputs

        move_time = time.time() - initial_time
        h, m, s = convert_elapsed_time(move_time)
        logger.info('Data moved to GPU in {} hours {} minutes {:.2f} seconds.'
                    .format(h, m, s))
        """

    if batch_size is None:
        batch_size = len(inputs.values())

    if isinstance(batch_size, int):
        chunks = list(get_chunks(inputs, batch_size, svm=False))
        targets_ = list(get_chunks(targets, batch_size, svm=False))

    del targets

    targets = []

    for t in targets_:
        t = OrderedDict(t)
        vectors = []
        for hash in t.keys():
            features = t[hash]
            for symbol, vector in features:
                vectors.append(vector.detach().numpy())
        vectors = torch.tensor(vectors, requires_grad=False)
        targets.append(vectors)

    logging.info('Batch size: {} elements per batch.' .format(batch_size))

    # Define optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)

    logger.info('{:6s} {:19s} {:12s} {:9s}'.format('Epoch',
                                                   'Time Stamp',
                                                   'Loss',
                                                   'Rec Err'))
    logger.info('{:6s} {:19s} {:12s} {:9s}'.format('------',
                                                   '-------------------',
                                                   '------------',
                                                   '--------'))
    initial_time = time.time()

    _loss = []
    _rmse = []
    epoch = 0

    # Get client to send futures to the scheduler
    client = dask.distributed.get_client()

    while True:
        epoch += 1
        optimizer.zero_grad()  # clear previous gradients

        losses = []
        outputs_ = []
        accumulation = []
        grads = []
        # Accumulation of gradients
        for index, chunk in enumerate(chunks):
            accumulation.append(client.submit(train_batches, *(index, chunk,
                                                               targets, model,
                                                               lossfxn,
                                                               device)))

        dask.distributed.wait(accumulation)
        # accumulation = dask.compute(*accumulation, scheduler='distributed')
        accumulation = client.gather(accumulation)

        for index, chunk in enumerate(accumulation):
            outputs = chunk[0]
            loss = chunk[1]
            grad = np.array(chunk[2])
            losses.append(loss)
            outputs_.append(outputs)
            grads.append(grad)

        grads = sum(grads)

        for index, param in enumerate(model.parameters()):
            param.grad = torch.tensor(grads[index])

        loss = sum(losses)

        optimizer.step()

        rmse = []
        for index, chunk in enumerate(outputs_):
            rmse.append(torch.sqrt(torch.mean((chunk -
                        targets[index]).pow(2))).item())

        rmse = sum(rmse)

        _loss.append(loss.item())
        _rmse.append(rmse)

        ts = time.time()
        ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d '
                                                          '%H:%M:%S')
        logger.info('{:6d} {} {:8e} {:8f}' .format(epoch, ts, loss, rmse))

        if convergence is None and epoch == epochs:
            break
        elif (convergence is not None and rmse < convergence['rmse']):
            break

    training_time = time.time() - initial_time

    h, m, s = convert_elapsed_time(training_time)
    logger.info('Training finished in {} hours {} minutes {:.2f} seconds.'
                .format(h, m, s))
    logger.info('outputs')
    logger.info(outputs)
    logger.info('targets')
    logger.info(targets)
    return epoch, _loss, _rmse


def train_batches(index, chunk, targets, model, lossfxn, device):
    """A function that allows training per batches"""
    inputs = OrderedDict(chunk)
    outputs = model(inputs)

    if lossfxn is None:
        loss = MSELossAE(outputs, targets[index])
    else:
        raise('I do not know what to do')

    loss.backward()

    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.detach().numpy())

    return outputs, loss, gradients
