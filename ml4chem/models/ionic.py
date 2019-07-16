import dask
import datetime
import logging
import time
import torch
import random
import scipy.special
import numpy as np
from collections import OrderedDict
from ml4chem.metrics import compute_rmse
from ml4chem.models.loss import AtomicMSELoss
from ml4chem.optim.handler import get_optimizer, get_lr_scheduler
from ml4chem.utils import convert_elapsed_time, get_chunks

# Setting precision and starting logger object
torch.set_printoptions(precision=10)
logger = logging.getLogger()


class NeuralNetwork(torch.nn.Module):
    """Atom-centered Neural Network Regression with Pytorch

    This model is based on the Ref. 1 by Behler and Parrinello.

    Parameters
    ----------
    hiddenlayers : tuple
        Structure of hidden layers in the neural network.
    activation : str
        The activation function.

    References
    ----------
    1. Behler, J. & Parrinello, M. Generalized Neural-Network Representation
       of High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98,
       146401 (2007).
    2. Khorshidi, A. & Peterson, A. A. Amp : A modular approach to machine
       learning in atomistic simulations. Comput. Phys. Commun. 207, 310–324
       (2016).
    """

    NAME = "PytorchIonicPotentials"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(
        self,
        hiddenlayers=(3, 3),
        activation="relu",
        energies=None,
        alpha_dict=ALPHA_DICT,
        **kwargs
    ):
        super(NeuralNetwork, self).__init__()
        self.hiddenlayers = hiddenlayers
        self.activation = activation
        self.energies = energies
        self.alpha = alpha_dict
        self.latent_space = None

    def prepare_model(self, input_dimension, data=None, purpose="training"):
        """Prepare the model

        Parameters
        ----------
        input_dimension : int
            Input's dimension.
        data : object
            DataSet object created from the handler.
        purpose : str
            Purpose of this model: 'training', 'inference'.
        """
        self.input_dimension = input_dimension

        activation = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
            "celu": torch.nn.CELU,
        }

        hl = len(self.hiddenlayers)
        if purpose == "training":
            logger.info(" ")
            logger.info("Model Training")
            logger.info("==============")
            logger.info("Model name: {}.".format(self.name()))
            logger.info("Number of hidden-layers: {}".format(hl))
            logger.info(
                "Structure of Neural Net: {}".format(
                    "(input, " + str(self.hiddenlayers)[1:-1] + ", output)"
                )
            )
            logger.info(" ")
        layers = range(len(self.hiddenlayers) + 1)
        try:
            unique_element_symbols = data.unique_element_symbols[purpose]
        except TypeError:
            unique_element_symbols = data.get_unique_element_symbols(purpose=purpose)
            unique_element_symbols = unique_element_symbols[purpose]

        symbol_model_pair = []

        for symbol in unique_element_symbols:
            linears = []

            intercept_name = "intercept_" + symbol
            slope_name = "slope_" + symbol
            alpha_name = "alpha_" + symbol

            if purpose == "training":
                intercept = (data.max_energy + data.min_energy) / 2.0
                intercept = torch.nn.Parameter(
                    torch.tensor(intercept, requires_grad=True)
                )
                slope = (data.max_energy - data.min_energy) / 2.0
                slope = torch.nn.Parameter(torch.tensor(slope, requires_grad=True))
                alpha = self.alpha[symbol]
                alpha = torch.nn.Parameter(torch.tensor(alpha, requires_grad=True))
                self.register_parameter(alpha_name, alpha)
                self.register_parameter(intercept_name, intercept)
                self.register_parameter(slope_name, slope)

            elif purpose == "inference":
                intercept = torch.nn.Parameter(torch.tensor(0.0))
                slope = torch.nn.Parameter(torch.tensor(0.0))
                alpha = torch.nn.Parameter(torch.tensor(0.0))
                alpha = torch.nn.Parameter(torch.tensor(alpha, requires_grad=True))
                self.register_parameter(alpha_name, alpha)
                self.register_parameter(intercept_name, intercept)
                self.register_parameter(slope_name, slope)

            for index in layers:
                # This is the input layer
                if index == 0:
                    out_dimension = self.hiddenlayers[0]
                    _linear = torch.nn.Linear(input_dimension, out_dimension)
                    linears.append(_linear)
                    linears.append(activation[self.activation]())
                # This is the output layer
                elif index == len(self.hiddenlayers):
                    inp_dimension = self.hiddenlayers[index - 1]
                    out_dimension = 1
                    _linear = torch.nn.Linear(inp_dimension, out_dimension)
                    linears.append(_linear)
                # These are hidden-layers
                else:
                    inp_dimension = self.hiddenlayers[index - 1]
                    out_dimension = self.hiddenlayers[index]
                    _linear = torch.nn.Linear(inp_dimension, out_dimension)
                    linears.append(_linear)
                    linears.append(activation[self.activation]())

            # Stacking up the layers.
            linears = torch.nn.Sequential(*linears)
            symbol_model_pair.append([symbol, linears])

        self.linears = torch.nn.ModuleDict(symbol_model_pair)

        if purpose == "training":
            logger.info(self.linears)
            # Iterate over all modules and just intialize those that are
            # a linear layer.
            logger.warning(
                "Initialization of weights with Xavier Uniform by " "default."
            )
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    # nn.init.normal_(m.weight)   # , mean=0, std=0.01)
                    torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, X, atoms):
        # Q is iterated charges of size (N+1)x1
        # A_ij is of size (N+1)x(N+1)
        # N = number of atoms
        # alpha is a hyperparameter of size (N+1)
        # Q, A_ij, alpha=None

        """Forward propagation

        This is forward propagation and it returns the atomic energy.

        Parameters
        ----------
        X : list
            List of inputs in the feature space.

        Returns
        -------
        outputs : tensor
            A list of tensors with energies per image.
        """

        outputs = []
        latent_space = OrderedDict()
        for hash in X:
            latent_space[hash] = self.get_latent_space(X, atoms, hash)
            atomic_energies = latent_space[hash][0]
            A_ij = latent_space[hash][1]
            electronegativities = latent_space[hash][2]
            charges = latent_space[hash][3]

            eq1 = 0.0
            for i in range(len(atoms[hash])):
                eq1 += (
                    atomic_energies[i]
                    - electronegativities[i] * charges[i]
                    + 0.5 * A_ij[i][i] * (charges[i] ** 2)
                )
            eq2 = 0.0
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms[hash])):
                    eq2 = charges[i] * charges[j] * A_ij[i][j]
            total_energy = eq1 + eq2

            outputs.append(total_energy)
        outputs = torch.stack(outputs)
        return outputs, latent_space

    # gets atomic energies, atomic charges, electronegativities
    def get_latent_space(self, X, atoms, hash):
        atomic_energies = self.get_atomic_energies(X, hash)
        electronegativities = self.get_electronegativities(X, hash)
        atomic_charges = self.get_atomic_charges(X, atoms, hash)
        A_ij, symbols = self.calculate_Aij(X, atoms, hash)
        image = X[hash]
        latent_space = []
        latent_space.append(atomic_energies)
        latent_space.append(A_ij)
        latent_space.append(electronegativities)
        latent_space.append(atomic_charges)
        latent_space.append(symbols)
        return latent_space

    def get_atomic_energies(self, X, hash):
        image = X[hash]
        atomic_energies = []
        for symbol, x in image:
            if isinstance(symbol, bytes):
                symbol = symbol.decode("utf-8")
            x = self.linears[symbol](x)
            intercept_name = "intercept_" + symbol
            slope_name = "slope_" + symbol
            slope = getattr(self, slope_name)
            intercept = getattr(self, intercept_name)
            x = (slope * x) + intercept
            atomic_energies.append(x)
        return atomic_energies

    def get_atomic_charges(self, X, atoms, hash):
        electronegativities = self.get_electronegativities(X, hash)
        A_ij, symbols = self.calculate_Aij(X, atoms, hash)
        charges = np.linalg.solve(A_ij, electronegativities.detach().numpy())
        return charges

    def get_electronegativities(self, X, hash):
        electronegativities = []
        image = X[hash]
        for symbol, x in image:
            if isinstance(symbol, bytes):
                symbol = symbol.decode("utf-8")
            x = self.linears[symbol](x)
            electronegativities.append(-x)
        electronegativities = torch.cat(electronegativities)
        electronegativities = torch.cat((electronegativities, torch.zeros(1)))
        return electronegativities

    def calculate_Aij(self, X, atoms, hash):
        A_ij = []
        alpha = self.alpha
        positions = atoms[hash].get_positions()
        symbols = atoms[hash].get_chemical_symbols()
        N = len(atoms[hash])
        A_ij = torch.ones(size=(N + 1, N + 1))
        A_ij[N][N] = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    # make sure alphas are correct
                    lamba = 1 / (alpha[symbols[i]] * (2 ** 0.5))
                    hardness = EN_HD[symbols[i]][1]
                    A_ij[i][j] = hardness + 2 * lamba / (np.pi ** 0.5)
                else:
                    # changes non_diagonals
                    distance = 0
                    x = positions[i][0] - positions[j][0]
                    x *= x
                    y = positions[i][1] - positions[j][1]
                    y *= y
                    z = positions[i][2] - positions[j][2]
                    z *= z
                    distance = (x + y + z) ** (0.5)

                    lamba = 1 / (
                        (alpha[symbols[i]] ** 0.5 + alpha[symbols[j]] ** 0.5) ** 0.5
                    )
                    A_ij[i][j] = scipy.special.erf(lamba * distance) / distance
        return A_ij, symbols


class train(object):
    """Train the model

    Parameters
    ----------
    inputs : dict
        Dictionary with hashed feature space.
    targets : list
        The expected values that the model has to learn aka y.
    model : object
        The NeuralNetwork class.
    data : object
        DataSet object created from the handler.
    optimizer : tuple
        The optimizer is a tuple with the structure:
            >>> ('adam', {'lr': float, 'weight_decay'=float})

    epochs : int
        Number of full training cycles.
    regularization : float
        This is the L2 regularization. It is not the same as weight decay.
    convergence : dict
        Instead of using epochs, users can set a convergence criterion.
    lossfxn : obj
        A loss function object.
    device : str
        Calculation can be run in the cpu or cuda (gpu).
    batch_size : int
        Number of data points per batch to use for training. Default is None.
    lr_scheduler : tuple
        Tuple with structure: scheduler's name and a dictionary with keyword
        arguments.

        >>> lr_scheduler = ('ReduceLROnPlateau',
                            {'mode': 'min', 'patience': 10})
    """

    def __init__(
        self,
        inputs,
        targets,
        model=None,
        data=None,
        optimizer=(None, None),
        regularization=None,
        epochs=100,
        convergence=None,
        lossfxn=None,
        device="cpu",
        batch_size=None,
        lr_scheduler=None,
        atoms=None,
    ):
        self.atoms = atoms

        self.initial_time = time.time()

        atoms_per_image = data.atoms_per_image

        if batch_size is None:
            batch_size = len(inputs.values())

        if isinstance(batch_size, int):
            # Data batches
            chunks = list(get_chunks(inputs, batch_size, svm=False))
            targets = list(get_chunks(targets, batch_size, svm=False))
            atoms_per_image = list(get_chunks(atoms_per_image, batch_size, svm=False))

        logger.info(" ")
        logging.info("Batch Information")
        logging.info("-----------------")
        logging.info("Number of batches: {}.".format(len(chunks)))
        logging.info("Batch size: {} elements per batch.".format(batch_size))
        logger.info(" ")

        atoms_per_image = torch.tensor(
            atoms_per_image, requires_grad=False, dtype=torch.float
        )

        targets = torch.tensor(targets, requires_grad=False)

        if device == "cuda":
            logger.info("Moving data to CUDA...")

            atoms_per_image = atoms_per_image.cuda()
            targets = targets.cuda()
            _inputs = OrderedDict()

            for hash, f in inputs.items():
                _inputs[hash] = []
                for features in f:
                    symbol, vector = features
                    _inputs[hash].append((symbol, vector.cuda()))

            inputs = _inputs

            move_time = time.time() - self.initial_time
            h, m, s = convert_elapsed_time(move_time)
            logger.info(
                "Data moved to GPU in {} hours {} minutes {:.2f} \
                         seconds.".format(
                    h, m, s
                )
            )
            logger.info(" ")

        # Define optimizer
        self.optimizer_name, self.optimizer = get_optimizer(
            optimizer, model.parameters()
        )

        if lr_scheduler is not None:
            self.scheduler = get_lr_scheduler(self.optimizer, lr_scheduler)

        logger.info(" ")
        logger.info("Starting training...")
        logger.info(" ")

        logger.info(
            "{:6s} {:19s} {:12s} {:8s} {:8s}".format(
                "Epoch", "Time Stamp", "Loss", "RMSE/img", "RMSE/atom"
            )
        )
        logger.info(
            "{:6s} {:19s} {:12s} {:8s} {:8s}".format(
                "------", "-------------------", "------------", "--------", "---------"
            )
        )
        self.atoms_per_image = atoms_per_image
        self.convergence = convergence
        self.device = device
        self.epochs = epochs
        self.model = model
        self.lr_scheduler = lr_scheduler

        # Data scattering
        client = dask.distributed.get_client()
        self.chunks = [client.scatter(chunk) for chunk in chunks]
        self.targets = [client.scatter(target) for target in targets]

        if lossfxn is None:
            self.lossfxn = AtomicMSELoss
            # self.model.latent_space = temp.latent_space
        else:
            self.lossfxn = lossfxn

        # Let the hunger games begin...
        self.trainer()

    def trainer(self):
        """Run the training class"""

        converged = False
        _loss = []
        _rmse = []
        epoch = 0

        while not converged:
            epoch += 1
            self.optimizer.zero_grad()  # clear previous gradients

            loss, outputs_, latent_space = train.closure(
                self,
                self.chunks,
                self.targets,
                self.model,
                self.lossfxn,
                self.atoms_per_image,
                self.device,
                self.atoms,
            )
            self.model.latent_space = latent_space
            # We step the optimizer
            if self.optimizer_name != "LBFGS":
                self.optimizer.step()
            else:
                # self.optimizer.extra_arguments = args
                options = {"closure": self.closure, "current_loss": loss, "max_ls": 10}
                self.optimizer.step(options)

            # RMSE per image and per/atom
            client = dask.distributed.get_client()

            rmse = client.submit(compute_rmse, *(outputs_, self.targets))
            atoms_per_image = self.atoms_per_image.view(1, -1)
            rmse_atom = client.submit(
                compute_rmse, *(outputs_, self.targets, atoms_per_image)
            )
            rmse = rmse.result()
            rmse_atom = rmse_atom.result()

            _loss.append(loss.item())
            _rmse.append(rmse)

            # In the case that lr_scheduler is not None
            if self.lr_scheduler is not None:
                self.scheduler.step(loss)

            ts = time.time()
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d " "%H:%M:%S")
            logger.info(
                "{:6d} {} {:8e} {:8f} {:8f}".format(epoch, ts, loss, rmse, rmse_atom)
            )

            if self.convergence is None and epoch == self.epochs:
                converged = True
            elif self.convergence is not None and rmse < self.convergence["energy"]:
                converged = True

        training_time = time.time() - self.initial_time

        h, m, s = convert_elapsed_time(training_time)
        logger.info(
            "Training finished in {} hours {} minutes {:.2f} seconds.".format(h, m, s)
        )

    @classmethod
    def closure(
        Cls, self, chunks, targets, model, lossfxn, atoms_per_image, device, atoms
    ):
        """Closure

        This method clears previous gradients, iterates over batches,
        accumulates the gradients, reduces the gradients, update model
        params, and finally returns loss and outputs_.
        """

        outputs_ = []
        # Get client to send futures to the scheduler
        client = dask.distributed.get_client()

        running_loss = torch.tensor(0, dtype=torch.float)
        accumulation = []
        grads = []

        # Accumulation of gradients
        for index, chunk in enumerate(chunks):
            accumulation.append(
                client.submit(
                    train.train_batches,
                    *(
                        index,
                        chunk,
                        targets,
                        model,
                        lossfxn,
                        atoms_per_image,
                        device,
                        atoms,
                    )
                )
            )
        dask.distributed.wait(accumulation)
        accumulation = client.gather(accumulation)

        latent_space = []
        grads = []
        for outputs, loss, grad, latent_chunk in accumulation:
            outputs_.append(outputs)
            running_loss += loss
            grad = np.array(grad)
            grads.append(grad)
            latent_space.append(latent_chunk)

        grads = sum(grads)

        for index, param in enumerate(model.parameters()):
            param.grad = torch.tensor(grads[index])

        del accumulation
        del grads

        return running_loss, outputs_, latent_space

    @classmethod
    def train_batches(
        Cls, index, chunk, targets, model, lossfxn, atoms_per_image, device, atoms
    ):
        """A function that allows training per batches


        Parameters
        ----------
        index : int
            Index of batch.
        chunk : tensor or list
            Tensor with input data points in batch with index.
        targets : tensor or list
            The targets.
        model : obj
            Pytorch model to perform forward() and get gradients.
        lossfxn : obj
            A loss function object.
        atoms_per_image : list
            Atoms per image because we are doing atom-centered methods.
        device : str
            Are we running cuda or cpu?

        Returns
        -------
        loss : tensor
            The loss function of the batch.
        """
        inputs = OrderedDict(chunk)
        latent_space = None
        outputs, latent_space = model(inputs, atoms)

        loss = lossfxn(outputs, targets[index], atoms_per_image[index])
        loss.backward()

        gradients = []

        for param in model.parameters():
            try:
                gradient = param.grad.detach().numpy()
            except AttributeError:
                # This exception catches  the case where an image does not
                # contain variable that is following the gradient of certain
                # atom. For example, suppose two batches with 2 molecules each.
                # In the first batch we have only C, H, O but it turns out that
                # N is also available only in the sencond batch. The
                # contribution to the gradient of batch 1 to the N gradients is
                # 0.
                gradient = 0.0
            gradients.append(gradient)

        return outputs, loss, gradients, latent_space


EN_HD = {
    "H": (0.264, 0.236),
    "He": (0.443, 0.461),
    "Li": (0.11, 0.088),
    "Be": (0.162, 0.18),
    "B": (0.158, 0.147),
    "C": (0.23, 0.184),
    "N": (0.266, 0.268),
    "O": (0.277, 0.223),
    "F": (0.383, 0.258),
    "Ne": (0.374, 0.418),
    "Na": (0.104, 0.084),
    "Mg": (0.133, 0.148),
    "Al": (0.118, 0.102),
    "Si": (0.175, 0.124),
    "P": (0.206, 0.179),
    "S": (0.229, 0.152),
    "Cl": (0.305, 0.172),
    "Ar": (0.271, 0.308),
    "K": (0.089, 0.071),
    "Ca": (0.113, 0.112),
    "Sc": (0.124, 0.117),
    "Ti": (0.127, 0.124),
    "V": (0.134, 0.114),
    "Cr": (0.137, 0.112),
    "Mn": (0.127, 0.146),
    "Fe": (0.148, 0.142),
    "Co": (0.157, 0.133),
    "Ni": (0.162, 0.119),
    "Cu": (0.165, 0.119),
    "Zn": (0.162, 0.184),
    "Ga": (0.118, 0.102),
    "Ge": (0.168, 0.123),
    "As": (0.195, 0.165),
    "Se": (0.216, 0.142),
    "Br": (0.279, 0.155),
    "Kr": (0.239, 0.276),
    "Rb": (0.086, 0.068),
    "Sr": (0.106, 0.109),
    "Y": (0.12, 0.12),
    "Zr": (0.13, 0.114),
    "Nb": (0.141, 0.107),
    "Mo": (0.144, 0.117),
    "Tc": (0.144, 0.124),
    "Ru": (0.154, 0.116),
    "Rh": (0.158, 0.116),
    "Pd": (0.164, 0.143),
    "Ag": (0.163, 0.115),
    "Cd": (0.152, 0.178),
    "In": (0.113, 0.099),
    "Sn": (0.155, 0.115),
    "Sb": (0.177, 0.139),
    "Te": (0.202, 0.129),
    "I": (0.248, 0.136),
    "Xe": (0.208, 0.238),
    "Cs": (0.08, 0.063),
    "Ba": (0.098, 0.093),
    "La": (0.113, 0.092),
    "Hf": (0.129, 0.122),
    "Ta": (0.145, 0.133),
    "W": (0.159, 0.13),
    "Re": (0.145, 0.143),
    "Os": (0.175, 0.135),
    "Ir": (0.194, 0.136),
    "Pt": (0.204, 0.126),
    "Au": (0.212, 0.127),
    "Hg": (0.183, 0.201),
    "Tl": (0.119, 0.105),
    "Pb": (0.143, 0.13),
    "Bi": (0.151, 0.117),
    "Po": (0.18, 0.129),
    "At": (0.216, 0.127),
    "Rn": (0.185, 0.21),
    "Fr": (0.084, 0.066),
    "Ra": (0.099, 0.095),
    "Ac": (0.101, 0.089),
    "Rf": (0, None),
    "Db": (0, None),
    "Sg": (0, None),
    "Bh": (0, None),
    "Hs": (0, None),
    "Mt": (0, None),
    "Ds": (0, None),
    "Rg": (0, None),
    "Cn": (0, None),
    "Nh": (0, None),
    "Fl": (0, None),
    "Mc": (0, None),
    "Lv": (0, None),
    "Ts": (0, None),
    "Og": (0, None),
    "Ce": (0.112, 0.091),
    "Pr": (0.118, 0.083),
    "Nd": (0.137, 0.066),
    "Pm": (0.105, 0.1),
    "Sm": (0.107, 0.101),
    "Eu": (0.106, 0.102),
    "Gd": (0.116, 0.11),
    "Tb": (0.129, 0.086),
    "Dy": (0.116, 0.103),
    "Ho": (0.117, 0.104),
    "Er": (0.118, 0.106),
    "Tm": (0.133, 0.095),
    "Yb": (0.115, 0.115),
    "Lu": (0.104, 0.095),
    "Th": (0.137, 0.094),
    "Pa": (0.118, 0.098),
    "U": (0.124, 0.104),
    "Np": (0.124, 0.106),
    "Pu": (0.102, 0.12),
    "Am": (0.112, 0.108),
    "Cm": (0.115, 0.105),
    "Bk": (0.082, 0.145),
    "Cf": (0.097, 0.134),
    "Es": (0.112, 0.123),
    "Fm": (0.126, 0.113),
    "Md": (0.139, 0.103),
    "No": (0.079, 0.165),
    "Lr": (0.084, 0.096),
}

ALPHA_DICT = {
    "H": 1.0,
    "He": 1.0,
    "Li": 1.0,
    "Be": 1.0,
    "B": 1.0,
    "C": 1.0,
    "N": 1.0,
    "O": 1.0,
    "F": 1.0,
    "Ne": 1.0,
    "Na": 1.0,
    "Mg": 1.0,
    "Al": 1.0,
    "Si": 1.0,
    "P": 1.0,
    "S": 1.0,
    "Cl": 1.0,
    "Ar": 1.0,
    "K": 1.0,
    "Ca": 1.0,
    "Sc": 1.0,
    "Ti": 1.0,
    "V": 1.0,
    "Cr": 1.0,
    "Mn": 1.0,
    "Fe": 1.0,
    "Co": 1.0,
    "Ni": 1.0,
    "Cu": 1.0,
    "Zn": 1.0,
    "Ga": 1.0,
    "Ge": 1.0,
    "As": 1.0,
    "Se": 1.0,
    "Br": 1.0,
    "Kr": 1.0,
    "Rb": 1.0,
    "Sr": 1.0,
    "Y": 1.0,
    "Zr": 1.0,
    "Nb": 1.0,
    "Mo": 1.0,
    "Tc": 1.0,
    "Ru": 1.0,
    "Rh": 1.0,
    "Pd": 1.0,
    "Ag": 1.0,
    "Cd": 1.0,
    "In": 1.0,
    "Sn": 1.0,
    "Sb": 1.0,
    "Te": 1.0,
    "I": 1.0,
    "Xe": 1.0,
    "Cs": 1.0,
    "Ba": 1.0,
    "La": 1.0,
    "Hf": 1.0,
    "Ta": 1.0,
    "W": 1.0,
    "Re": 1.0,
    "Os": 1.0,
    "Ir": 1.0,
    "Pt": 1.0,
    "Au": 1.0,
    "Hg": 1.0,
    "Tl": 1.0,
    "Pb": 1.0,
    "Bi": 1.0,
    "Po": 1.0,
    "At": 1.0,
    "Rn": 1.0,
    "Fr": 1.0,
    "Ra": 1.0,
    "Ac": 1.0,
    "Rf": 1.0,
    "Db": 1.0,
    "Sg": 1.0,
    "Bh": 1.0,
    "Hs": 1.0,
    "Mt": 1.0,
    "Ds": 1.0,
    "Rg": 1.0,
    "Cn": 1.0,
    "Nh": 1.0,
    "Fl": 1.0,
    "Mc": 1.0,
    "Lv": 1.0,
    "Ts": 1.0,
    "Og": 1.0,
    "Ce": 1.0,
    "Pr": 1.0,
    "Nd": 1.0,
    "Pm": 1.0,
    "Sm": 1.0,
    "Eu": 1.0,
    "Gd": 1.0,
    "Tb": 1.0,
    "Dy": 1.0,
    "Ho": 1.0,
    "Er": 1.0,
    "Tm": 1.0,
    "Yb": 1.0,
    "Lu": 1.0,
    "Th": 1.0,
    "Pa": 1.0,
    "U": 1.0,
    "Np": 1.0,
    "Pu": 1.0,
    "Am": 1.0,
    "Cm": 1.0,
    "Bk": 1.0,
    "Cf": 1.0,
    "Es": 1.0,
    "Fm": 1.0,
    "Md": 1.0,
    "No": 1.0,
    "Lr": 1.0,
}
