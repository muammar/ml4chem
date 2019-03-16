import time
import datetime
import dask
import numpy as np
from collections import OrderedDict


class KernelRidge(object):
    """Kernel Ridge Regression

    Parameters
    ----------
    sigma : float, list, or dict
        Length scale of the Gaussian in the case of RBF, exponential, and
        laplacian kernels. Default is 1. (float) and it computes isotropic
        kernels. Pass a list if you would like to compute anisotropic kernels,
        or a dictionary if you want sigmas for each model.

        Example:

            >>> sigma={'energy': {'H': value, 'O': value},
                       'forces': {'H': {0: value, 1: value, 2: value},
                              'O': {0: value, 1: value, 2: value}}}

        `value` can be a float or a list.
    kernel : str
        Choose the kernel. Available kernels are: 'linear', 'rbf', 'laplacian',
        and 'exponential'. Default is 'rbf'.
    lamda : float, or dictionary
        Strength of the regularization. If you pass a dictionary then force and
        energy will have different regularization:

            >>> lamda = {'energy': value, 'forces': value}

        Dictionaries are only used when performing Cholesky factorization.
    weights : dict
        Dictionary of weights.
    regressor : object
        Regressor class to be used.
    mode : str
        Atom- or image-centered mode.
    trainingimages : str
        Path to Trajectory file containing the images in the training set. This
        is useful for predicting new structures.
    version : str
        Version.
    fortran : bool
        Use fortran code.
    checkpoints : int
        Frequency with which to save parameter checkpoints upon training. E.g.,
        100 saves a checkpoint on each 100th training set.  Specify None for
        no checkpoints. Default is None.
    lossfunction : object
        Loss function object.
    cholesky : bool
        Whether or not we are using Cholesky decomposition to determine the
        weights. This method returns an unique set of regression coefficients.
    weights_independent : bool
        Whether or not the weights are going to be split for energy and forces.
    randomize_weights : bool
        If set to True, weights are randomly started when minimizing the L2
        loss function.
    forcetraining : bool
        Turn force training true.
    nnpartition : str
        Use per-atom energy partition from a neural network calculator.
        You have to set the path to .amp file. Useful for energy training with
        Cholesky factorization. Default is set to None.
    preprocessing : bool
        Preprocess training data.
    sum_rule : bool
        Whether or not we sum of fingerprintprime elements over a given axis.
        This applies np.sum(fingerprint_list, axis=0).

    Notes
    -----
        In the case of training total energies, we need to apply either an
        atomic decomposition Ansatz (ADA) during training or an energy
        partition scheme to the training set. ADA can be achieved based on
        Int. J.  Quantum Chem., vol. 115, no.  16, pp.  1051-1057, Aug. 2015".
        For an explanation of what they do, see the Master thesis by Sonja
        Mathias.

        http://wissrech.ins.uni-bonn.de/teaching/master/masterthesis_mathias_revised.pdf

        ADA is the default way of training total energies in this KernelRidge
        class.

        An energy partition scheme for  total energies can be obtained from an
        artificial neural network or methods such as the interacting quantum
        atoms theory (IQA). I implemented the nnpartition mode for which users
        can provide the path to a NN calculator and we take the energies
        per-atom from the function .calculate_atomic_energy(). The strategy
        would be to use train the NN with a very tight convergence criterion
        (1e-6 RSME).  Then, calling .calculate_atomic_energy() would give you
        the atomic energies for such set.

        For forces is a different history because we do know the derivative of
        the energy with respect to atom positions (a per-atom quantity).  So we
        rely on the method in the algorithm shown in Rupp, M. (2015).  Machine
        learning for quantum mechanics in a nutshell. International Journal of
        Quantum Chemistry, 115(16), 1058-1073.

    Parameters
    ----------
    hiddenlayers : tuple
        Structure of hidden layers in the neural network.
    activation : str
        The activation function.
    """

    NAME = 'KernelRidge'

    @classmethod
    def name(cls):
        """Returns name of class"""

        return cls.NAME

    def __init__(self, sigma=1., kernel='rbf', scheduler='distributed',
                 lamda=1e-5, weights=None, regressor=None, mode=None,
                 trainingimages=None, version=None, fortran=False,
                 checkpoints=None, lossfunction=None, cholesky=True,
                 weights_independent=True, randomize_weights=False,
                 forcetraining=False, preprocessing=False, nnpartition=None,
                 sum_rule=True):

        np.set_printoptions(precision=30, threshold=999999999)
        self.kernel = kernel
        self.sigma = sigma
        self.scheduler = scheduler

    def prepare_model(self, feature_space, reference_features, data=None,
                      purpose='training'):
        """Prepare the model

        Parameters
        ----------
        feature_space : dict
            A dictionary with hash, fingerprint structure.
        reference_features : dict
            A dictionary with raveled tuples of symbol, atomic fingerprint.
        data : object
            DataSet object created from the handler.
        purpose : str
            Purpose of this model: 'training', 'inference'.
        """
        self.fingerprint_map = []

        unique_element_symbols = data.unique_element_symbols[purpose]
        dim = len(reference_features)

        call = {'exponential': exponential, 'laplacian': laplacian, 'rbf': rbf}

        atomic_kernel_matrices = []

        for symbol in unique_element_symbols:
            # We start populating computations with delayed functions to
            # operate with dask's scheduler
            computations = []
            for hash, _feature_space in feature_space.items():
                f_map = []
                for i_symbol, i_afp in _feature_space:
                    f_map.append(1)
                    for j_symbol, j_afp in reference_features:
                        kernel = call[self.kernel](i_afp, j_afp,
                                                   i_symbol=i_symbol,
                                                   j_symbol=j_symbol,
                                                   sigma=self.sigma)
                        computations.append(kernel)
                self.fingerprint_map.append(f_map)

            # We compute the calculations with dask and the result is converted
            # to numpy array.
            kernel_matrix = np.array((dask.compute(*computations,
                                      scheduler=self.scheduler)))
            kernel_matrix = kernel_matrix.reshape(dim, dim)
            atomic_kernel_matrices.append((symbol, kernel_matrix))

        self.atomic_kernel_matrices = OrderedDict(atomic_kernel_matrices)

        # We build the LT matrix needed for ADA
        computations = []
        for index, feature_space in enumerate(feature_space.items()):
            computations.append(self.get_lt(index))

        self.LT = list(dask.compute(*computations, scheduler=self.scheduler))

    @dask.delayed
    def get_lt(self, index):
        """Return LT vectors

        Parameters
        ----------
        index : int
            Index of image.

        Returns
        -------
        _LT : list
            Returns a list that maps atomic fingerprints in the images.
        """
        _LT = []

        for i, group in enumerate(self.fingerprint_map):
            if i == index:
                for _ in group:
                    _LT.append(1.)
            else:
                for _ in group:
                    _LT.append(0.)
        return _LT

    def train(inputs, targets, model=None, data=None, optimizer=None, lr=None,
              weight_decay=None, regularization=None, epochs=100,
              convergence=None, lossfxn=None):
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
        lossfxn : obj
            A loss function object.
        """
        pass

        """
        #old_state_dict = {}

        #for key in model.state_dict():
        #    old_state_dict[key] = model.state_dict()[key].clone()

        targets = torch.tensor(targets, requires_grad=False)

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
                loss, rmse = loss_function(outputs, targets, optimizer, data)
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
            elif (convergence is not None and rmse < convergence['energy']):
                break

        training_time = time.time() - initial_time

        print('Training the model took {}...' .format(training_time))
        print('outputs')
        print(outputs)
        print('targets')
        print(targets)

        import matplotlib.pyplot as plt
        plt.plot(list(range(epoch)), _loss, label='loss')
        plt.plot(list(range(epoch)), _rmse, label='rmse/atom')
        plt.legend(loc='upper left')
        plt.show()

        parity(outputs.detach().numpy(), targets.detach().numpy())
        """


"""
Auxiliary functions to compute kernels
"""
@dask.delayed
def linear(feature_i, feature_j, i_symbol=None, j_symbol=None):
    """ Compute a linear kernel

    Parameters
    ----------
    feature_i : np.array
        Atomic fingerprint for central atom.
    feature_j : np.array
        Atomic fingerprint for j atom.
    i_symbol : str
        Chemical symbol for central atom.
    j_symbol : str
        Chemical symbol for j atom.

    Returns
    -------
    linear :float
        Linear kernel.
    """

    if i_symbol != j_symbol:
        return 0.
    else:
        linear = np.dot(feature_i, feature_j)
        return linear


@dask.delayed
def rbf(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.):
    """ Compute the rbf (AKA Gaussian) kernel.

    Parameters
    ----------
    feature_i : np.array
        Atomic fingerprint for central atom.
    feature_j : np.array
        Atomic fingerprint for j atom.
    i_symbol : str
        Chemical symbol for central atom.
    j_symbol : str
        Chemical symbol for j atom.
    sigma : float, or list.
        Gaussian width. If passed as a list or np.darray, kernel can become
        anisotropic.

    Returns
    -------
    rbf :float
        RBF kernel.
    """

    if i_symbol != j_symbol:
        return 0.
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert(len(sigma) == len(feature_i) and
                   len(sigma) == len(feature_j)), "Length of sigma does not " \
                                                  "match atomic fingerprint " \
                                                  "length."
            sigma = np.array(sigma)
            anisotropic_rbf = np.exp(-(np.sum(np.divide(np.square(
                              np.subtract(feature_i, feature_j)),
                                          (2. * np.square(sigma))))))
            return anisotropic_rbf
        else:
            rbf = np.exp(-(np.linalg.norm(feature_i - feature_j) ** 2.) /
                         (2. * sigma ** 2.))
            return rbf


@dask.delayed
def exponential(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.):
    """ Compute the exponential kernel

    Parameters
    ----------
    feature_i : np.array
        Atomic fingerprint for central atom.
    feature_j : np.array
        Atomic fingerprint for j atom.
    i_symbol : str
        Chemical symbol for central atom.
    j_symbol : str
        Chemical symbol for j atom.
    sigma : float, or list.
        Gaussian width.

    Returns
    -------
    exponential : float
        Exponential kernel.
    """

    if i_symbol != j_symbol:
        return 0.
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert(len(sigma) == len(feature_i) and
                   len(sigma) == len(feature_j)), "Length of sigma does not " \
                                                  "match atomic fingerprint " \
                                                  "length."
            sigma = np.array(sigma)
            anisotropic_exp = np.exp(-(np.sqrt(np.sum(np.square(
                          np.divide(np.subtract(feature_i, feature_j),
                                               (2. * np.square(sigma))))))))
            return anisotropic_exp
        else:
            exponential = np.exp(-(np.linalg.norm(feature_i - feature_j)) /
                                 (2. * sigma ** 2))
            return exponential


@dask.delayed
def laplacian(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.):
    """ Compute the laplacian kernel

    Parameters
    ----------
    feature_i : np.array
        Atomic fingerprint for central atom.
    feature_j : np.array
        Atomic fingerprint for j atom.
    i_symbol : str
        Chemical symbol for central atom.
    j_symbol : str
        Chemical symbol for j atom.
    sigma : float
        Gaussian width.

    Returns
    -------
    laplacian : float
        Laplacian kernel.
    """

    if i_symbol != j_symbol:
        return 0.
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert(len(sigma) == len(feature_i) and
                   len(sigma) == len(feature_j)), "Length of sigma does not " \
                                                  "match atomic fingerprint " \
                                                  "length."
            sigma = np.array(sigma)

            sum_ij = np.sum(np.square(np.divide(np.subtract(feature_i,
                                                            feature_j),
                                                sigma)))

            anisotropic_lap = np.exp(-(np.sqrt(sum_ij)))
            return anisotropic_lap
        else:
            laplacian = np.exp(-(np.linalg.norm(feature_i - feature_j)) /
                               sigma)
        return laplacian
