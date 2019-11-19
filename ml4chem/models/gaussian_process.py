import dask
import logging
import numpy as np
from collections import OrderedDict
from ml4chem.models.kernelridge import KernelRidge
from scipy.linalg import cholesky

logger = logging.getLogger()


class GaussianProcess(KernelRidge):
    """Gaussian Process Regression

    This method is based on the KernelRidge regression class of ML4Chem.

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
        This regressor applies the atomic decomposition Ansatz (ADA). For
        more information check the Notes on the KernelRidge class.
    """

    NAME = "GaussianProcess"

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
    ):
        super(KernelRidge, self).__init__()

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
        delete = ["self", "__class__"]

        for param in delete:
            del _params[param]

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
        energy, variance
            Energy of a molecule and its respective variance.
        """
        reference_space = reference_space[b"reference_space"]
        computations = self.get_kernel_matrix(features, reference_space, purpose)
        kernel = np.array(dask.compute(*computations, scheduler=self.scheduler))
        weights = np.array(self.weights["energy"])
        dim = int(kernel.shape[0] / weights.shape[0])
        kernel = kernel.reshape(dim, len(weights))
        energy_per_atom = np.dot(kernel, weights)
        energy = np.sum(energy_per_atom)
        variance = self.get_variance(features, kernel, reference_space, purpose)
        return energy, variance

    def get_variance(self, features, ks, reference_space, purpose):
        """Compute predictive variance

        Parameters
        ----------
        features : dict
            Dictionary with data point to be predicted.
        ks : array
            Variance between data point and reference space.
        reference_space : list
            Reference space used to compute kernel.
        purpose : str
            Purpose of this function: 'training', 'inference'.

        Returns
        -------
        variance
            Predictive variance.
        """
        K = self.get_kernel_matrix(reference_space, reference_space, purpose)
        K = np.array(dask.compute(*K, scheduler=self.scheduler))
        dim = int(np.sqrt(len(K)))
        K = K.reshape(dim, dim)

        if isinstance(self.lamda, dict):
            lamda = self.lamda["energy"]
        else:
            lamda = self.lamda

        size = K.shape[0]
        I_e = np.identity(size)
        cholesky_U = cholesky((K + (lamda * I_e)))
        betas = np.linalg.solve(cholesky_U.T, ks.T)

        variance = ks.dot(np.linalg.solve(cholesky_U, betas))
        kxx = self.get_kernel_matrix(features, features, purpose)
        kxx = np.array(dask.compute(*kxx, scheduler=self.scheduler))
        dim = int(np.sqrt(len(kxx)))
        kxx = kxx.reshape(dim, dim)
        variance = np.sum(kxx - variance)

        return variance
