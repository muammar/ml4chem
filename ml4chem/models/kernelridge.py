import dask
import datetime
import logging
import time
import numpy as np
from ml4chem.utils import convert_elapsed_time, get_chunks
from collections import OrderedDict
from scipy.linalg import cholesky

logger = logging.getLogger()


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
    trainingimages : str
        Path to Trajectory file containing the images in the training set. This
        is useful for predicting new structures.
    cholesky : bool
        Whether or not we are using Cholesky decomposition to determine the
        weights. This method returns an unique set of regression coefficients.
    weights_independent : bool
        Whether or not the weights are going to be split for energy and forces.
    forcetraining : bool
        Turn force training true.
    nnpartition : str
        Use per-atom energy partition from a neural network calculator.
        You have to set the path to .amp file. Useful for energy training with
        Cholesky factorization. Default is set to None.
    scheduler : str
        The scheduler to be used with the dask backend.
    sum_rule : bool
        Whether or not we sum of fingerprintprime elements over a given axis.
        This applies np.sum(fingerprint_list, axis=0).
    batch_size : int
        Number of elements per batch in order to split computations. Useful
        when number of local chemical environments is too large.
    weights : dict
        Dictionary of weights.

    Notes
    -----
        In the case of training total energies, we need to apply either an
        atomic decomposition Ansatz (ADA) during training or an energy
        partition scheme to the training set. ADA can be achieved based on
        Ref. 1.
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
        rely on the method in the algorithm shown by Rupp in Ref. 2.

    References
    ----------
    1. Bartók, A. P. & Csányi, G. Gaussian approximation potentials: A brief
    tutorial introduction. Int. J. Quantum Chem. 115, 1051–1057 (2015).
    2. Rupp, M. Machine learning for quantum mechanics in a nutshell. Int. J.
       Quantum Chem. 115, 1058–1073 (2015).
    """

    NAME = "KernelRidge"

    @classmethod
    def name(cls):
        """Returns name of class"""

        return cls.NAME

    def __init__(
        self,
        sigma=1.0,
        kernel="rbf",
        scheduler="distributed",
        lamda=1e-5,
        trainingimages=None,
        checkpoints=None,
        cholesky=True,
        weights_independent=True,
        forcetraining=False,
        nnpartition=None,
        sum_rule=True,
        batch_size=None,
        weights=None,
        **kwargs
    ):

        np.set_printoptions(precision=30, threshold=999999999)
        self.kernel = kernel
        self.sigma = sigma
        self.scheduler = scheduler
        self.lamda = lamda
        self.batch_size = batch_size

        # Let's add parameters that are going to be stored in the .params json
        # file.
        self.params = OrderedDict()
        self.params["name"] = self.name()
        self.params["type"] = "svm"
        self.params["class_name"] = self.__class__.__name__

        # This is a very general way of not forgetting to save variables
        _params = vars()

        # Delete useless variables
        del _params["self"]

        for k, v in _params.items():
            if v is not None:
                self.params[k] = v

        # Everything that is added here is not going to be part of the json
        # params file
        self.fingerprint_map = []

        if weights is None:
            self.weights = {}
        else:
            self.weights = weights

    def prepare_model(
        self, feature_space, reference_features, data=None, purpose="training"
    ):
        """Prepare the Kernel Ridge Regression model

        Parameters
        ----------
        feature_space : dict
            A dictionary with hash, fingerprint structure.
        reference_features : dict
            A dictionary with raveled tuples of symbol, atomic fingerprint.
        data : object
            Data object created from the handler.
        purpose : str
            Purpose of this model: 'training', 'inference'.


        Notes
        -----
        This method builds the atomic kernel matrices and the LT vectors needed
        to apply the atomic decomposition Ansatz.
        """
        if purpose == "training":
            now = datetime.datetime.now()
            logger.info(" ")
            logger.info("Model")
            logger.info("=====")
            logger.info(
                "Module accessed on {}.".format(now.strftime("%Y-%m-%d %H:%M:%S"))
            )
            logger.info("Model name: {}.".format(self.name()))
            logger.info("Kernel parameters:")
            logger.info("    - Kernel function: {}.".format(self.kernel))
            logger.info("    - Sigma: {}.".format(self.sigma))
            logger.info("    - Lamda: {}.".format(self.lamda))
            logger.info(" ")

        dim = len(reference_features)

        """
        Atomic kernel matrices
        """

        logger.info("Computing Kernel Matrix...")
        # We start populating computations with delayed functions to
        # operate with dask's scheduler
        kernel_matrix = self.get_kernel_matrix(
            feature_space, reference_features, purpose=purpose
        )

        # futures = self.get_kernel_matrix(
        #     feature_space, reference_features, purpose=purpose
        # )

        # if self.batch_size is not None:
        #     futures = list(get_chunks(futures, self.batch_size))
        #     logger.info(
        #         "    The calculations are in batches of {}.".format(self.batch_size)
        #     )

        # We compute the calculations with dask and the result is converted
        # to numpy array.

        # if self.batch_size is None:
        #     kernel_matrix = client.gather(futures)
        # else:
        #     kernel_matrix = []
        #     for chunk in futures:
        #         kernel_matrix += client.gather(chunk)

        # FIXME probably not very efficient yet.
        # Found at https://stackoverflow.com/a/36250972/1995261

        _K = np.zeros([dim, dim])
        indices_upper = np.triu_indices(dim)
        _K[indices_upper] = kernel_matrix
        self.K = _K.T + _K
        np.fill_diagonal(self.K, np.diag(_K))
        del _K

    def get_kernel_matrix(self, feature_space, reference_features, purpose):
        """Get kernel matrix delayed computations


        Parameters
        ----------
        features : dict
            Dictionary with hash and features.
        reference_space : array
            Array with reference feature space.
        purpose : str
            Purpose of this kernel matrix. Accepted arguments are 'training',
            and 'inference'.

        Returns
        -------
        kernel_matrix
            List with kernel matrix values.
        """
        initial_time = time.time()

        call = {"exponential": exponential, "laplacian": laplacian, "rbf": rbf}

        if self.batch_size is None:
            chunks = [feature_space]
        else:
            chunks = list(get_chunks(feature_space, self.batch_size))
            logger.info(
                "    The calculations are distributed in {} batches of {} molecules.".format(
                    len(chunks), self.batch_size
                )
            )

        counter = 0
        kernel_matrix = []

        for c, chunk in enumerate(chunks):
            chunk_initial_time = time.time()
            logger.info("        Computing kernel functions for chunk {}...".format(c))
            intermediates = []

            if isinstance(chunk, dict) is False:
                chunk = OrderedDict(chunk)

            if isinstance(chunk, dict):
                if isinstance(reference_features, dict):
                    # This is the case when the reference_features are a
                    # dictionary, too.
                    reference_features = list(reference_features.values())[0]

                reference_lenght = len(reference_features)
                for hash, _feature_space in chunk.items():
                    f_map = []
                    for i_symbol, i_afp in _feature_space:
                        i_symbol = decode(i_symbol)
                        f_map.append(1)

                        if purpose == "training":

                            for j in range(counter, reference_lenght):
                                j_symbol, j_afp = reference_features[j]

                                kernel = call[self.kernel](
                                    i_afp, j_afp, i_symbol, j_symbol, self.sigma
                                )

                                intermediates.append(kernel)
                            counter += 1
                        else:
                            for j_symbol, j_afp in reference_features:
                                j_symbol = decode(j_symbol)
                                kernel = call[self.kernel](
                                    i_afp, j_afp, i_symbol, j_symbol, self.sigma
                                )
                                intermediates.append(kernel)
                    self.fingerprint_map.append(f_map)
            else:
                for i_symbol, i_afp in chunk:
                    for j_symbol, j_afp in reference_features:
                        i_symbol = decode(i_symbol)
                        j_symbol = decode(j_symbol)

                        kernel = call[self.kernel](
                            i_afp, j_afp, i_symbol, j_symbol, self.sigma
                        )
                        intermediates.append(kernel)
            kernel_matrix += dask.compute(intermediates, scheduler=self.scheduler)[0]
            del intermediates

            chunk_final_time = time.time() - chunk_initial_time
            h, m, s = convert_elapsed_time(chunk_final_time)
            logger.info(
                "          ...finished in {} hours {} minutes {:.2f} "
                "seconds.".format(h, m, s)
            )
            # dask.distributed.wait(kernel_matrix)

        del reference_features

        # kernel_matrix = client.gather(kernel_matrix)
        build_time = time.time() - initial_time
        h, m, s = convert_elapsed_time(build_time)
        logger.info(
            "Kernel matrix built in {} hours {} minutes {:.2f} "
            "seconds.".format(h, m, s)
        )

        """
        LT Vectors
        """
        # We build the LT matrix needed for ADA
        if purpose == "training":
            self.LT = []
            logger.info("Building LT matrix")
            computations = []
            for index, feature_space in enumerate(feature_space.items()):
                computations.append(self.get_lt(index))

            if self.batch_size is not None:
                computations = list(get_chunks(computations, self.batch_size))
                logger.info(
                    "    The calculations are distributed in {} batches of {} molecules.".format(
                        len(computations), self.batch_size
                    )
                )
                for chunk in computations:
                    self.LT += dask.compute(*chunk, scheduler=self.scheduler)

                self.LT = np.array(self.LT)
                del computations
                del chunk
                lt_time = time.time() - initial_time
                h, m, s = convert_elapsed_time(lt_time)
                logger.info(
                    "LT matrix built in {} hours {} minutes {:.2f} seconds.".format(
                        h, m, s
                    )
                )

        return kernel_matrix

    def train(self, inputs, targets, data=None):
        """Train the model

        Parameters
        ----------
        inputs : dict
            Dictionary with hashed feature space.
        targets : list
            The expected values that the model has to learn aka y.
        data : object
            Data object created from the handler.

        """

        if isinstance(self.lamda, dict):
            lamda = self.lamda["energy"]
        else:
            lamda = self.lamda

        size = len(targets)
        I_e = np.identity(size)
        K = self.LT.dot(self.K).dot(self.LT.T)
        del self.LT
        logger.info("Size of the Kernel matrix is {}.".format(K.shape))
        logger.info("Starting Cholesky Factorization...")
        cholesky_U = cholesky((K + (lamda * I_e)))
        logger.info("Cholesky Factorization finished...")
        betas = np.linalg.solve(cholesky_U.T, targets)
        _weights = np.linalg.solve(cholesky_U, betas)
        _weights = [
            w * g
            for index, w in enumerate(_weights)
            for g in self.fingerprint_map[index]
        ]
        self.weights["energy"] = _weights

    def get_potential_energy(self, features, reference_space, purpose):
        """Get potential energy with Kernel Ridge

        Parameters
        ----------
        features : dict
            Dictionary with hash and features.
        reference_space : array
            Array with reference feature space.
        purpose : str
            Purpose of this function: 'training', 'inference'.

        Returns
        -------
        energy
            Energy of a molecule.
        """
        client = dask.distributed.get_client()
        reference_space = reference_space[b"reference_space"]
        futures = self.get_kernel_matrix(features, reference_space, purpose=purpose)
        kernel = np.array(client.gather(futures))
        weights = np.array(self.weights["energy"])
        dim = int(kernel.shape[0] / weights.shape[0])
        kernel = kernel.reshape(dim, len(weights))
        energy_per_atom = np.dot(kernel, weights)
        energy = np.sum(energy_per_atom)
        return energy

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
            Returns a list that maps atomic features in the images.
        """
        _LT = []

        for i, group in enumerate(self.fingerprint_map):
            if i == index:
                for _ in group:
                    _LT.append(1.0)
            else:
                for _ in group:
                    _LT.append(0.0)
        return _LT

    def get_sigma(self, sigma, forcetraining=False):
        """Function to build sigma

        Parameters
        ----------
        sigma : float, list or dict.
            This is user's raw input for sigma.
        forcetraining : bool
            Whether or not force training is set to true.

        Returns
        -------
        _sigma : dict
            Universal sigma dictionary for KernelRidge.
        """

        _sigma = {}

        if isinstance(sigma, int):
            for symbol in unique_element_symbols:
                _sigma[symbol] = sigma

        return _sigma


"""
Auxiliary functions
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
        return 0.0
    else:
        linear = np.dot(feature_i, feature_j)
        return linear


@dask.delayed
def rbf(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.0):
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
        return 0.0
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert len(sigma) == len(feature_i) and len(sigma) == len(feature_j), (
                "Length of sigma does not " "match atomic fingerprint " "length."
            )
            sigma = np.array(sigma)
            anisotropic_rbf = np.exp(
                -(
                    np.sum(
                        np.divide(
                            np.square(np.subtract(feature_i, feature_j)),
                            (2.0 * np.square(sigma)),
                        )
                    )
                )
            )
            return anisotropic_rbf
        else:
            rbf = np.exp(
                -(np.linalg.norm(feature_i - feature_j) ** 2.0) / (2.0 * sigma ** 2.0)
            )
            return rbf


@dask.delayed
def exponential(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.0):
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
        return 0.0
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert len(sigma) == len(feature_i) and len(sigma) == len(feature_j), (
                "Length of sigma does not " "match atomic fingerprint " "length."
            )
            sigma = np.array(sigma)
            anisotropic_exp = np.exp(
                -(
                    np.sqrt(
                        np.sum(
                            np.square(
                                np.divide(
                                    np.subtract(feature_i, feature_j),
                                    (2.0 * np.square(sigma)),
                                )
                            )
                        )
                    )
                )
            )
            return anisotropic_exp
        else:
            exponential = np.exp(
                -(np.linalg.norm(feature_i - feature_j)) / (2.0 * sigma ** 2)
            )
            return exponential


@dask.delayed
def laplacian(feature_i, feature_j, i_symbol=None, j_symbol=None, sigma=1.0):
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
        return 0.0
    else:
        if isinstance(sigma, list) or isinstance(sigma, np.ndarray):
            assert len(sigma) == len(feature_i) and len(sigma) == len(feature_j), (
                "Length of sigma does not " "match atomic fingerprint " "length."
            )
            sigma = np.array(sigma)

            sum_ij = np.sum(
                np.square(np.divide(np.subtract(feature_i, feature_j), sigma))
            )

            anisotropic_lap = np.exp(-(np.sqrt(sum_ij)))
            return anisotropic_lap
        else:
            laplacian = np.exp(-(np.linalg.norm(feature_i - feature_j)) / sigma)
        return laplacian


def decode(symbol):
    """Decode from binary to string

    Parameters
    ----------
    symbol : binary
        A string in binary form, e.g. b'hola'.

    Returns
    -------
    str
        Symbol as a string.
    """
    try:
        return symbol.decode()
    except AttributeError:
        return symbol
