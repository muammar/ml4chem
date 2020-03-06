import dask
import datetime
import logging
import time
import torch

import numpy as np
import pandas as pd
from collections import OrderedDict
from ml4chem.metrics import compute_rmse
from ml4chem.atomistic.models.base import DeepLearningModel, DeepLearningTrainer
from ml4chem.atomistic.models.loss import AtomicMSELoss
from ml4chem.optim.handler import get_optimizer, get_lr_scheduler, get_lr
from ml4chem.utils import convert_elapsed_time, get_chunks, get_number_of_parameters
from pprint import pformat


# Setting precision and starting logger object
torch.set_printoptions(precision=10)
logger = logging.getLogger()


class NeuralNetwork(DeepLearningModel):
    """Atom-centered Neural Network Regression with Pytorch

    This model is based on Ref. 1 by Behler and Parrinello.

    Parameters
    ----------
    hiddenlayers : tuple
        Structure of hidden layers in the neural network.
    activation : str
        Activation functions. Supported "tanh", "relu", or "celu".

    References
    ----------
    1. Behler, J. & Parrinello, M. Generalized Neural-Network Representation
       of High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98,
       146401 (2007).
    2. Khorshidi, A. & Peterson, A. A. Amp : A modular approach to machine
       learning in atomistic simulations. Comput. Phys. Commun. 207, 310–324
       (2016).
    """

    NAME = "PytorchPotentials"

    @classmethod
    def name(cls):
        """Returns name of class"""

        return cls.NAME

    def __init__(self, hiddenlayers=(3, 3), activation="relu", **kwargs):
        super(DeepLearningModel, self).__init__()
        self.hiddenlayers = hiddenlayers
        self.activation = activation

    def prepare_model(self, input_dimension, data=None, purpose="training"):
        """Prepare the model

        Parameters
        ----------
        input_dimension : int
            Input's dimension.
        data : object
            Data object created from the handler.
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
            logger.info("Model")
            logger.info("=====")
            now = datetime.datetime.now()
            logger.info(
                "Module accessed on {}.".format(now.strftime("%Y-%m-%d %H:%M:%S"))
            )
            logger.info("Model name: {}.".format(self.name()))
            logger.info("Number of hidden-layers: {}".format(hl))
            logger.info(
                "Structure of Neural Net: {}".format(
                    "(input, " + str(self.hiddenlayers)[1:-1] + ", output)"
                )
            )

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

            if purpose == "training":
                intercept = (data.max_energy + data.min_energy) / 2.0
                intercept = torch.nn.Parameter(
                    torch.tensor(intercept, requires_grad=True)
                )
                slope = (data.max_energy - data.min_energy) / 2.0
                slope = torch.nn.Parameter(torch.tensor(slope, requires_grad=True))

                self.register_parameter(intercept_name, intercept)
                self.register_parameter(slope_name, slope)
            elif purpose == "inference":
                intercept = torch.nn.Parameter(torch.tensor(0.0))
                slope = torch.nn.Parameter(torch.tensor(0.0))
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
            total_params, train_params = get_number_of_parameters(self)
            logger.info("Total number of parameters: {}.".format(total_params))
            logger.info("Number of training parameters: {}.".format(train_params))
            logger.info(" ")
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

    def forward(self, X):
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

        for hash in X:
            image = X[hash]
            atomic_energies = []

            for symbol, x in image:
                # FIXME this conditional can be removed after de/serialization
                # is fixed.
                if isinstance(symbol, bytes):
                    symbol = symbol.decode("utf-8")
                x = self.linears[symbol](x)

                intercept_name = "intercept_" + symbol
                slope_name = "slope_" + symbol
                slope = getattr(self, slope_name)
                intercept = getattr(self, intercept_name)

                x = (slope * x) + intercept
                atomic_energies.append(x)

            atomic_energies = torch.cat(atomic_energies)
            image_energy = torch.sum(atomic_energies)
            outputs.append(image_energy)
        outputs = torch.stack(outputs)
        return outputs

    def get_activations(self, images, model=None, numpy=True):
        """Get activations of each hidden-layer

        This function allows to extract activations of each hidden-layer of
        the neural network. 

        Parameters
        ----------
        image : dict
           Image with structure hash, features. 
        model : object
            A ML4Chem model object.
        numpy : bool
            Whether we want numpy arrays or tensors. 


        Returns
        -------
        activations : DataFrame
            A DataFrame with activations for each layer.  
        """

        activations = []
        columns = ["Hash", "atom.index", "atom.symbol"]

        if model is None:
            model = self

        model.eval()

        for hash, data in images.items():
            for index, (symbol, features) in enumerate(data):

                counter = 0
                layer_counter = 0
                for l, layer in enumerate(model.linears[symbol].modules()):
                    if isinstance(layer, torch.nn.Linear) and counter == 0:
                        x = layer(features)

                        if numpy:
                            data_ = [hash, index, symbol, x.detach().numpy()]
                        else:
                            data_ = [hash, index, symbol, x]

                        layer_column_name = f"layer{layer_counter}"

                        if layer_column_name not in columns:
                            columns.append(layer_column_name)

                        counter += 1
                        layer_counter += 1

                    elif isinstance(layer, torch.nn.Linear) and counter > 0:
                        x = layer(x)

                        if numpy:
                            data_.append(x.detach().numpy())
                        else:
                            data_.append(x)

                        layer_column_name = f"layer{layer_counter}"
                        if layer_column_name not in columns:
                            columns.append(layer_column_name)

                        counter += 1
                        layer_counter += 1

                activations.append(data_)
                del data_

        # Create DataFrame from lists
        df = pd.DataFrame(activations, columns=columns)

        return df


class train(DeepLearningTrainer):
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
        Data object created from the handler.
    optimizer : tuple
        The optimizer is a tuple with the structure:
            >>> ('adam', {'lr': float, 'weight_decay'=float})

    epochs : int
        Number of full training cycles.
    regularization : float
        This is the L2 regularization. It is not the same as weight decay.
    convergence : dict
        Instead of using epochs, users can set a convergence criterion.
        Supported keys are "training" and "test".
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
    uncertainty : list
        A list of uncertainties that are used to penalize during the loss
        function evaluation.
    checkpoint : dict
        Set checkpoints. Dictionary with following structure:

        >>> checkpoint = {"label": label, "checkpoint": 100, "path": ""}
        
        `label` refers to the name used to save the checkpoint, `checkpoint`
        is a integer or -1 for saving all epochs, and the path is where the
        checkpoint is stored. Default is None and no checkpoint is saved.
    test : dict
        A dictionary used to compute the error over a validation/test set
        during training procedures.

        >>>  test = {"features": test_space, "targets": test_targets, "data": data_test}

        The keys,values of the dictionary are: 

        - "data": a `Data` object.
        - "targets": test set targets. 
        - "features": a feature space obtained using `features.calculate()`.

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
        uncertainty=None,
        checkpoint=None,
        test=None,
    ):

        self.initial_time = time.time()

        if lossfxn is None:
            lossfxn = AtomicMSELoss

        logger.info("")
        logger.info("Training")
        logger.info("========")
        logger.info(f"Convergence criteria: {convergence}")
        logger.info(f"Loss function: {lossfxn.__name__}")
        if uncertainty is not None:
            logger.info("Options:")
            logger.info(f"    - Uncertainty penalization: {pformat(uncertainty)}")
        logger.info("")

        atoms_per_image = data.atoms_per_image

        if batch_size is None:
            batch_size = len(inputs.values())

        if isinstance(batch_size, int):
            # Data batches
            chunks = list(get_chunks(inputs, batch_size, svm=False))
            targets = list(get_chunks(targets, batch_size, svm=False))
            atoms_per_image = list(get_chunks(atoms_per_image, batch_size, svm=False))

            if uncertainty != None:
                uncertainty = list(get_chunks(uncertainty, batch_size, svm=False))
                uncertainty = [
                    torch.tensor(u, requires_grad=False, dtype=torch.float)
                    for u in uncertainty
                ]

        logger.info("")
        logging.info("Batch Information")
        logging.info("-----------------")
        logging.info("Number of batches: {}.".format(len(chunks)))
        logging.info("Batch size: {} elements per batch.".format(batch_size))
        logger.info(" ")

        atoms_per_image = [
            torch.tensor(n_atoms, requires_grad=False, dtype=torch.float)
            for n_atoms in atoms_per_image
        ]

        targets = [torch.tensor(t, requires_grad=False) for t in targets]

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

        self.atoms_per_image = atoms_per_image
        self.convergence = convergence
        self.device = device
        self.epochs = epochs
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.lossfxn = lossfxn
        self.checkpoint = checkpoint
        self.test = test

        # Data scattering
        client = dask.distributed.get_client()
        self.chunks = [client.scatter(chunk) for chunk in chunks]
        self.targets = [client.scatter(target) for target in targets]

        if uncertainty != None:
            self.uncertainty = [client.scatter(u) for u in uncertainty]
        else:
            self.uncertainty = uncertainty

        # Let the hunger games begin...
        self.trainer()

    def trainer(self):
        """Run the training class"""

        logger.info(" ")
        logger.info("Starting training...\n")

        if self.test is None:
            logger.info(
                "{:6s} {:19s} {:12s} {:12s} {:8s}".format(
                    "Epoch", "Time Stamp", "Loss", "Error/img", "Error/atom"
                )
            )
            logger.info(
                "{:6s} {:19s} {:12s} {:8s} {:8s}".format(
                    "------",
                    "-------------------",
                    "------------",
                    "------------",
                    "------------",
                )
            )

        else:
            test_features = self.test.get("features", None)
            test_targets = self.test.get("targets", None)
            test_data = self.test.get("data", None)

            logger.info(
                "{:6s} {:19s} {:12s} {:12s} {:12s} {:12s} {:16s}".format(
                    "Epoch",
                    "Time Stamp",
                    "Loss",
                    "Error/img",
                    "Error/atom",
                    "Error/img (t)",
                    "Error/atom (t)",
                )
            )
            logger.info(
                "{:6s} {:19s} {:12s} {:8s} {:8s} {:8s} {:8s}".format(
                    "------",
                    "-------------------",
                    "------------",
                    "------------",
                    "------------",
                    "------------",
                    "------------",
                )
            )

        converged = False
        _loss = []
        _rmse = []
        epoch = 0

        client = dask.distributed.get_client()

        while not converged:
            epoch += 1

            self.optimizer.zero_grad()  # clear previous gradients
            loss, outputs_ = train.closure(
                self.chunks,
                self.targets,
                self.uncertainty,
                self.model,
                self.lossfxn,
                self.atoms_per_image,
                self.device,
            )
            # We step the optimizer
            if self.optimizer_name != "LBFGS":
                self.optimizer.step()
            else:
                options = {"closure": self.closure, "current_loss": loss, "max_ls": 10}
                self.optimizer.step(options)

            # RMSE per image and per/atom

            rmse = client.submit(compute_rmse, *(outputs_, self.targets))
            atoms_per_image = torch.cat(self.atoms_per_image)

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
                print("Epoch {} lr {}".format(epoch, get_lr(self.optimizer)))

            ts = time.time()
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d " "%H:%M:%S")

            if self.test is None:
                logger.info(
                    "{:6d} {} {:8e} {:4e} {:4e}".format(
                        epoch, ts, loss.detach(), rmse, rmse_atom
                    )
                )
            else:
                test_model = self.model.eval()
                test_predictions = test_model(test_features).detach()
                rmse_test = client.submit(
                    compute_rmse, *(test_predictions, test_targets)
                )

                atoms_per_image_test = torch.tensor(
                    test_data.atoms_per_image, requires_grad=False
                )
                rmse_atom_test = client.submit(
                    compute_rmse,
                    *(test_predictions, test_targets, atoms_per_image_test),
                )

                rmse_test = rmse_test.result()
                rmse_atom_test = rmse_atom_test.result()

                logger.info(
                    "{:6d} {} {:8e} {:4e} {:4e} {:4e} {:4e}".format(
                        epoch,
                        ts,
                        loss.detach(),
                        rmse,
                        rmse_atom,
                        rmse_test,
                        rmse_atom_test,
                    )
                )

            if self.checkpoint is not None:
                self.checkpoint_save(epoch, self.model, **self.checkpoint)

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
        Cls, chunks, targets, uncertainty, model, lossfxn, atoms_per_image, device
    ):
        """Closure

        This class method clears previous gradients, iterates over batches,
        accumulates the gradients, reduces the gradients, update model
        params, and finally returns loss and outputs_.

        Parameters
        ----------
        Cls : object
            Class object.
        chunks : tensor or list
            Tensor with input data points in batch with index.
        targets : tensor or list
            The targets.
        uncertainty : list
            A list of uncertainties that are used to penalize during the loss
            function evaluation.
        model : obj
            Pytorch model to perform forward() and get gradients.
        lossfxn : obj
            A loss function object.
        atoms_per_image : list
            Atoms per image because we are doing atom-centered methods.
        device : str
            Are we running cuda or cpu?
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
                        uncertainty,
                        model,
                        lossfxn,
                        atoms_per_image,
                        device,
                    ),
                )
            )
        dask.distributed.wait(accumulation)
        accumulation = client.gather(accumulation)

        for outputs, loss, grad in accumulation:
            grad = np.array(grad)
            running_loss += loss
            outputs_.append(outputs)
            grads.append(grad)

        grads = sum(grads)

        for index, param in enumerate(model.parameters()):
            param.grad = torch.tensor(grads[index])

        del accumulation
        del grads

        return running_loss, outputs_

    @classmethod
    def train_batches(
        Cls, index, chunk, targets, uncertainty, model, lossfxn, atoms_per_image, device
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
        uncertainty : list
            A list of uncertainties that are used to penalize during the loss
            function evaluation.
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
        outputs = model(inputs)

        if uncertainty == None:
            loss = lossfxn(outputs, targets[index], atoms_per_image[index])
        else:
            loss = lossfxn(
                outputs, targets[index], atoms_per_image[index], uncertainty[index]
            )
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
                # N is also available only in the second batch. The
                # contribution of the total gradient from the first batch for N is 0.
                gradient = 0.0
            gradients.append(gradient)

        return outputs, loss, gradients
