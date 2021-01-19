import dask
import datetime
import logging
import time
import torch

import numpy as np
import pandas as pd
from collections import OrderedDict
from ml4chem.metrics import compute_mae
from ml4chem.atomistic.models.base import DeepLearningModel, DeepLearningTrainer
from ml4chem.atomistic.models.loss import AtomicMSELoss
from ml4chem.optim.handler import get_optimizer, get_lr_scheduler, get_lr
from ml4chem.utils import convert_elapsed_time, get_chunks, get_number_of_parameters
from pprint import pformat


# Setting precision and starting logger object
torch.set_printoptions(precision=10)
logger = logging.getLogger()


class NeuralNetwork(DeepLearningModel, torch.nn.Module):
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

    NAME = "RetentionTimes"

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
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Module accessed on {now}.")
            logger.info(f"Model name: {self.name()}.")
            logger.info(f"Number of hidden-layers: {hl}")

            nn_structure = f"(input, {str(self.hiddenlayers)[1:-1]}, output)"
            logger.info(f"Structure of Neural Net: {nn_structure}")

        layers = range(len(self.hiddenlayers) + 1)

        try:
            unique_element_symbols = data.unique_element_symbols[purpose]
        except TypeError:
            unique_element_symbols = data.get_unique_element_symbols(purpose=purpose)
            unique_element_symbols = unique_element_symbols[purpose]

        symbol_model_pair = []

        for symbol in unique_element_symbols:
            linears = []

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
                    linears.append(activation["relu"]())
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
            logger.info(f"Total number of parameters: {total_params}.")
            logger.info(f"Number of training parameters: {train_params}.")
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

    def forward(self, X, conditions):
        """Forward propagation

        This is forward propagation and it returns the atomic energy.

        Parameters
        ----------
        X : dict
            Dictionary of inputs in the feature space.
        condition : dict
            A dict of tensors per atom type with conditions. 

        Returns
        -------
        outputs 
            A dict of tensors with energies per atom.
        """
        outputs = {}

        for symbol, tensors in X.items():
            if isinstance(symbol, bytes):
                symbol = symbol.decode("utf-8")
            x = self.linears[symbol](tensors)
            # intercept_name = "intercept_" + symbol
            # slope_name = "slope_" + symbol
            # slope = getattr(self, slope_name)
            # intercept = getattr(self, intercept_name)
            # x = (slope * x) + intercept
            x = (x.squeeze() * conditions[symbol]).sum(dim=1)
            outputs[symbol] = x

        return outputs

    def get_forces(self, energies, coordinates):
        for index, energy in enumerate(energies):
            forces = torch.autograd.grad(
                energy, coordinates[index], retain_graph=True, create_graph=True
            )
            print(energy)
        raise NotImplementedError

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
        columns = ["hash", "atom.index", "atom.symbol"]

        if model is None:
            model = self

        model.eval()

        for hash, data in images.items():
            for index, (symbol, features) in enumerate(data):

                counter = 0
                layer_counter = 0
                for _, layer in enumerate(model.linears[symbol].modules()):
                    if isinstance(layer, torch.nn.Linear) and counter == 0:
                        x = layer(features)

                        if numpy:
                            data_ = [hash, index, symbol, x.detach_().numpy()]
                        else:
                            data_ = [hash, index, symbol, x.detach_()]

                        layer_column_name = f"layer{layer_counter}"

                        if layer_column_name not in columns:
                            columns.append(layer_column_name)

                        counter += 1
                        layer_counter += 1

                    elif isinstance(layer, torch.nn.Linear) and counter > 0:
                        x = layer(x)

                        if numpy:
                            data_.append(x.detach_().numpy())
                        else:
                            data_.append(x.detach_())

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
    forcetraining : bool, optional
        Activate force training. Default is False.

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
        forcetraining=False,
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
            # Input data batches
            chunks = get_chunks(inputs, batch_size, svm=False)
            atoms_per_image = list(get_chunks(atoms_per_image, batch_size, svm=False))

            for key, values in targets.items():
                targets[key] = list(get_chunks(values, batch_size, svm=False))

            if forcetraining:
                coordinates = list(get_chunks(data.coordinates, batch_size, svm=False))
                coordinates = [
                    torch.tensor(c, requires_grad=True, dtype=torch.float)
                    for c in coordinates
                ]

            if uncertainty != None:
                uncertainty = list(get_chunks(uncertainty, batch_size, svm=False))
                uncertainty = [
                    torch.tensor(u, requires_grad=False, dtype=torch.float)
                    for u in uncertainty
                ]

        # vectorization
        chunks, conditions = model.feature_preparation(chunks, data)

        logger.info("")
        logging.info("Batch Information")
        logging.info("-----------------")
        logging.info(f"Number of batches: {len(chunks)}.")
        logging.info(f"Batch size: {batch_size} elements per batch.")
        logger.info(" ")

        atoms_per_image = [
            torch.tensor(n_atoms, requires_grad=False, dtype=torch.float)
            for n_atoms in atoms_per_image
        ]

        for key, targets_ in targets.items():
            targets[key] = [torch.tensor(t, requires_grad=False) for t in targets_]

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
                f"Data moved to GPU in {h} hours {m} minutes {s:.2f} \
                         seconds."
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
        self.forcetraining = forcetraining
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.lossfxn = lossfxn
        self.checkpoint = checkpoint
        self.test = test

        # Data scattering
        logger.info(f"Scattering data to workers...")
        client = dask.distributed.get_client()
        # self.chunks = [client.scatter(chunk) for chunk in chunks]
        self.chunks = client.scatter(chunks, broadcast=True)
        self.conditions = client.scatter(conditions, broadcast=True)
        # self.conditions = [client.scatter(condition) for condition in conditions]
        # self.targets = [client.scatter(target) for target in targets]
        self.targets = OrderedDict()
        for key, targets_ in targets.items():
            self.targets[key] = [client.scatter(target) for target in targets_]

        if forcetraining:
            self.coordinates = [client.scatter(c) for c in coordinates]

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
            test_features, conditions = self.model.feature_preparation(
                test_features, test_data
            )

            # The preparation above is returning a list so the dictionary is
            # inside and has to be indexed.
            test_features, conditions = test_features[0], conditions[0]

            # FIXME adding [] is an ugly hack that has to be fixed.
            test_targets = [
                torch.tensor(list(test_targets.values())[0], requires_grad=False)
            ]

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
        epoch = 0

        client = dask.distributed.get_client()

        while not converged:
            epoch += 1

            self.optimizer.zero_grad()  # clear previous gradients
            args = [
                self.chunks,
                self.conditions,
                self.targets,
                self.uncertainty,
                self.model,
                self.lossfxn,
                self.atoms_per_image,
                self.device,
                self.forcetraining,
            ]

            # if self.forcetraining:
            #     args.append(self.coordinates)
            # else:
            #     args.append(None)

            loss, outputs = train.closure(*args)

            # We step the optimizer
            if self.optimizer_name != "LBFGS":
                self.optimizer.step()
            else:
                options = {"closure": self.closure, "current_loss": loss, "max_ls": 10}
                self.optimizer.step(options)

            # MAE per image and per/atom

            rmse = OrderedDict()
            for key, out in outputs.items():
                rmse_ = client.submit(compute_mae, *(out, self.targets[key]))
                rmse[key] = rmse_.result()

            atoms_per_image = torch.cat(self.atoms_per_image)

            rmse_atom = OrderedDict()
            for key, out in outputs.items():
                rmse_atom_ = client.submit(
                    compute_mae, *(out, self.targets[key], atoms_per_image)
                )
                rmse_atom[key] = rmse_atom_.result()
            # rmse = rmse.result()
            # rmse_atom = rmse_atom.result()
            # In the case that lr_scheduler is not None
            if self.lr_scheduler is not None:
                self.scheduler.step(loss)
                print("Epoch {} lr {}".format(epoch, get_lr(self.optimizer)))

            ts = time.time()
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d " "%H:%M:%S")

            if self.test is None:
                info = f"{epoch:6d} {ts} {loss.detach():8e}"

                for key in rmse.keys():
                    info += f" {rmse[key]:4e} {rmse_atom[key]:4e}"

                logger.info(info)
            else:
                test_model = self.model.eval()
                test_predictions = test_model(test_features, conditions)
                # FIXME adding [] is an ugly hack that has to be fixed.
                test_predictions = [torch.stack(list(test_predictions.values())).sum(0)]

                rmse_test = client.submit(
                    compute_mae, *(test_predictions, test_targets)
                )

                atoms_per_image_test = torch.tensor(
                    test_data.atoms_per_image, requires_grad=False
                )
                rmse_atom_test = client.submit(
                    compute_mae,
                    *(test_predictions, test_targets, atoms_per_image_test),
                )

                rmse_test = rmse_test.result()
                rmse_atom_test = rmse_atom_test.result()

                rmse = list(rmse.values())[0]
                rmse_atom = list(rmse_atom.values())[0]

                logger.info(
                    f"{epoch:6d} {ts} {loss.detach():8e} {rmse:4e} {rmse_atom:4e} {rmse_test:4e} {rmse_atom_test:4e}"
                )

            if self.checkpoint is not None:
                self.checkpoint_save(epoch, self.model, **self.checkpoint)

            if self.convergence is None and epoch == self.epochs:
                converged = True
            elif (
                self.convergence is not None
                and rmse["energies"] < self.convergence["energy"]
                and self.epochs is None
            ):
                converged = True
            elif (
                self.convergence is not None
                and rmse["energies"] < self.convergence["energy"]
                or epoch == self.epochs
            ):
                converged = True

        training_time = time.time() - self.initial_time

        h, m, s = convert_elapsed_time(training_time)
        logger.info(f"Training finished in {h} hours {m} minutes {s:.2f} seconds.")

    @classmethod
    def closure(
        Cls,
        chunks,
        conditions,
        targets,
        uncertainty,
        model,
        lossfxn,
        atoms_per_image,
        device,
        forcetraining,
        # coordinates,
    ):
        """Closure

        This class method clears previous gradients, iterates over batches,
        accumulates the gradients, reduces the gradients, update model
        params, and finally returns loss and outputs.

        Parameters
        ----------
        Cls : object
            Class object.
        chunks : tensor or list
            Tensor with input data points in batch with index.
        targets : tensor or list.
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
        forcetraining : bool, optional
            Activate force training. Default is False.
        coordinates : list
            List of coordinates of atoms. Useful for force training.
        """

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
                        conditions,
                        targets,
                        uncertainty,
                        model,
                        lossfxn,
                        atoms_per_image,
                        device,
                        forcetraining,
                    ),
                )
            )
        dask.distributed.wait(accumulation)
        accumulation = client.gather(accumulation)

        outputs = OrderedDict()

        for outputs_, loss, grad in accumulation:
            grad = np.array(grad)
            grads.append(grad)
            running_loss += loss

            for key, out in outputs_.items():
                if key not in outputs.keys():
                    outputs[key] = []
                outputs[key].append(out)

        grads = sum(grads)

        for index, param in enumerate(model.parameters()):
            param.grad = torch.tensor(grads[index])

        del accumulation
        del grads

        return running_loss, outputs

    @classmethod
    def train_batches(
        Cls,
        index,
        chunk,
        conditions,
        targets,
        uncertainty,
        model,
        lossfxn,
        atoms_per_image,
        device,
        forcetraining,
        # coordinates,
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
        forcetraining : bool, optional
            Activate force training. Default is False.
        coordinates : list
            List of coordinates of atoms. Useful for force training.

        Returns
        -------
        loss : tensor
            The loss function of the batch.
        """
        outputs_ = model(chunk, conditions[index])
        outputs_ = torch.stack(list(outputs_.values())).sum(0)

        outputs = {list(targets.keys())[0]: outputs_}

        if forcetraining:
            forces = model.get_forces(outputs["energies"], coordinates[index])

        loss = 0.0
        if uncertainty == None:
            for key, targets_ in targets.items():
                loss += lossfxn(
                    outputs[key], targets_[index].result(), atoms_per_image[index]
                )
        else:
            for key, targets_ in targets.items():
                loss += lossfxn(
                    outputs[key],
                    targets_[index].result(),
                    atoms_per_image[index],
                    uncertainty[index],
                )

        loss.backward()

        gradients = []

        for param in model.parameters():
            try:
                gradient = param.grad.detach().numpy()
            except AttributeError:
                # This exception catches  the case where an image does not
                # contain a variable that is following the gradient of certain
                # atom. For example, suppose two batches with 2 molecules each.
                # In the first batch we have only C, H, O atoms but it turns
                # out that N is also available only in the second batch. The
                # contribution of the total gradient from the first batch for N
                # is 0.
                gradient = 0.0
            gradients.append(gradient)

        return outputs, loss, gradients
