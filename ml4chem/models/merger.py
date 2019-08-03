import dask
import datetime
import inspect
import time
import torch
import logging
import numpy as np
from collections import OrderedDict
from ml4chem.metrics import compute_rmse
from ml4chem.utils import convert_elapsed_time, dynamic_import, get_chunks, lod_to_list
from ml4chem.optim.handler import get_optimizer, get_lr_scheduler

# Setting precision and starting logger object
torch.set_printoptions(precision=10)
logger = logging.getLogger()


class ModelMerger(torch.nn.Module):
    """Model Merger

    A class that can merge models and train them simultaneously. Models are
    executed sequentially. It is assumed that outputs of model1 are the
    inputs of model2. This behavior can be modified by adding `extra_funcs`
    to call external functions.

    Parameters
    ----------
    models : list
        A list of models.
            >>> models = [list of models]
    """

    NAME = "Merged"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(self, models):
        super(ModelMerger, self).__init__()
        self.models = models

    def forward(self, X, models):
        """Forward propagation

        Parameters
        ----------
        X : list
            List of models' inputs.
        models : list
            List of model objects.

        Returns
        -------
        outputs
            A list with the forward propagation evaluation.
        """

        outputs = []
        for i, model in enumerate(models):
            # _output = []
            name = model.name()
            x = X[i]
            if inspect.ismethod(x):
                x = X[i - 1]

            if name == "AutoEncoder":
                x = OrderedDict(x)
            elif name == "PytorchPotentials":
                x = X[i](OrderedDict(x))

            output = model(x)
            # _output.append(output)

            # outputs.append(_output)
            outputs.append(output)
        return outputs

    def train(
        self,
        inputs,
        targets,
        data=None,
        optimizer=(None, None),
        regularization=None,
        epochs=100,
        convergence=None,
        lossfxn=None,
        device="cpu",
        batch_size=None,
        lr_scheduler=None,
        independent_loss=True,
        loss_weights=None,
    ):

        self.epochs = epochs

        logger.info(" ")
        logging.info("Model Merger")
        logging.info("============")
        logging.info("Merging the following models:")

        for model in self.models:
            logging.info("    - {}.".format(model.name()))

        logging.info("Loss functions:")

        if loss_weights is None:
            self.loss_weights = [1.0 / len(lossfxn) for l in lossfxn]
        else:
            self.loss_weights = loss_weights

        for index, l in enumerate(lossfxn):
            logging.info(
                "    - Name: {}; Weight: {}.".format(
                    l.__name__, self.loss_weights[index]
                )
            )

        # If no batch_size provided then the whole training set length is the batch.
        if batch_size is None:
            batch_size = len(inputs.values())

        if isinstance(batch_size, int):
            chunks = []
            for inputs_ in inputs:

                if inspect.ismethod(inputs_):
                    chunks.append(inputs_)
                else:
                    chunks.append(list(get_chunks(inputs_, batch_size, svm=False)))

            targets = [
                list(get_chunks(target, batch_size, svm=False)) for target in targets
            ]
            atoms_per_image = list(
                get_chunks(data.atoms_per_image, batch_size, svm=False)
            )

        if lossfxn is None:
            self.lossfxn = [None for model in self.models]
        else:
            self.lossfxn = lossfxn

        self.device = device

        # Population of extra Attributes needed by the models, and further data
        # preprocessing

        for index, loss in enumerate(lossfxn):
            _args, _varargs, _keywords, _defaults = inspect.getargspec(loss)
            if "latent" in _args:
                train = dynamic_import(
                    "train", "ml4chem.models", alt_name="autoencoders"
                )
                self.inputs_chunk_vals = train.get_inputs_chunks(chunks[index])

        parameters = []
        for index, model in enumerate(self.models):
            parameters += model.parameters()
            if model.name() == "PytorchPotentials":
                # These models require targets as tensors
                self.atoms_per_image = torch.tensor(
                    atoms_per_image, requires_grad=False, dtype=torch.float
                )
                _targets = [
                    torch.tensor(batch, requires_grad=False) for batch in targets[index]
                ]
                targets[index] = _targets
                del _targets
            elif model.name() == "AutoEncoder":
                targets[index] = lod_to_list(targets[index])

        # Data scattering
        client = dask.distributed.get_client()

        # self.targets = [client.scatter(target) for target in targets]
        self.targets = [target for target in targets]

        self.chunks = []

        for i, chunk in enumerate(chunks):
            if inspect.ismethod(chunk) is False:
                self.chunks.append(client.scatter(chunk))
            else:
                # This list comprehension is useful to have the same number of
                # functions as the same number of chunks without users' input.
                chunk = [chunk for _ in range(len(self.targets[i]))]
                self.chunks.append(chunk)

        del chunks

        logger.info(" ")
        logging.info("Batch Information")
        logging.info("-----------------")
        logging.info("Number of batches:")
        for index, c in enumerate(self.chunks):
            logging.info("    - Model {}, {}.".format(index, len(c)))
        logging.info("Batch size: {} elements per batch.\n".format(batch_size))

        # Define optimizer

        self.optimizer_name, self.optimizer = get_optimizer(optimizer, parameters)

        if lr_scheduler is not None:
            self.scheduler = get_lr_scheduler(self.optimizer, lr_scheduler)

        logger.info(" ")
        logger.info("Starting training...")
        logger.info(" ")

        logger.info(
            "{:6s} {:19s} {:12s} {:8s}".format(
                "Epoch", "Time Stamp", "Loss", "RMSE (ave)"
            )
        )
        logger.info(
            "{:6s} {:19s} {:12s} {:8s}".format(
                "------", "-------------------", "------------", "--------------"
            )
        )

        converged = False
        epoch = 0

        if independent_loss is False:
            # Convert list of chunks from [[a, c], [b, d]] to [[a, b], [c, d]]
            self.chunks = list(map(list, zip(*self.chunks)))

        old_state_dict = {}

        for key in self.models[1].state_dict():
            old_state_dict[key] = self.models[1].state_dict()[key].clone()

        while not converged:
            epoch += 1

            self.optimizer.zero_grad()  # clear previous gradients

            if independent_loss:
                losses = []
                for model_index, model in enumerate(self.models):
                    name = model.name()
                    loss, outputs = self.closure(
                        model_index, model, independent_loss, name=name
                    )
                    losses.append(loss)

            else:
                loss, outputs = self.closure(index, self.models, independent_loss)

            rmse = []
            for i, model in enumerate(self.models):
                rmse.append(compute_rmse(outputs[i], self.targets[i]))
            # print(outputs[1])
            # print(targets[1])

            # print(rmse)
            _rmse = np.average(rmse)

            if self.optimizer_name != "LBFGS":
                self.optimizer.step()
            else:
                options = {"closure": self.closure, "current_loss": loss, "max_ls": 10}
                self.optimizer.step(options)

            ts = time.time()
            ts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d " "%H:%M:%S")
            logger.info("{:6d} {} {:8e} {:8f}".format(epoch, ts, loss, _rmse))

            if convergence is None and epoch == self.epochs:
                converged = True
            elif convergence is not None and all(
                i <= convergence["rmse"] for i in rmse
            ):
                converged = True
                new_state_dict = {}

                for key in self.models[1].state_dict():
                    new_state_dict[key] = self.models[1].state_dict()[key].clone()

                for key in old_state_dict:
                    if not (old_state_dict[key] == new_state_dict[key]).all():
                        print("Diff in {}".format(key))
                    else:
                        print("No diff in {}".format(key))

            # print(rmse)

    def closure(self, index, model, independent_loss, name=None):
        """Closure

        This method clears previous gradients, iterates over batches,
        accumulates the gradients, reduces the gradients, update model
        params, and finally returns loss and outputs_.

        Parameters
        ----------
        index : int
            Index of model.
        model : obj
            Model object.
        independent_loss : bool
            Whether or not models' weight are optimized independently.
        name : str, optional
            Model class's name, by default None.

        Returns
        -------
        loss, outputs
            A tuple with loss function magnitudes and tensor with outputs.
        """

        client = dask.distributed.get_client()

        if name == "PytorchPotentials" and independent_loss:
            train = dynamic_import("train", "ml4chem.models", alt_name="neuralnetwork")

            inputs = []
            for chunk in self.chunks[index - 1]:
                inputs_ = self.chunks[index](OrderedDict(chunk))
                inputs.append(client.scatter(inputs_))

            loss, outputs_ = train.closure(
                inputs,
                self.targets[index],
                model,
                self.lossfxn[index],
                self.atoms_per_image,
                self.device,
            )
            return loss, outputs_

        elif name == "AutoEncoder" and independent_loss:
            train = dynamic_import("train", "ml4chem.models", alt_name="autoencoders")
            # The indexing [0] is needed because of the way the array is built
            # above.
            targets = self.targets[index]

            loss, outputs_ = train.closure(
                self.chunks[index],
                targets,
                model,
                self.lossfxn[index],
                self.device,
                self.inputs_chunk_vals,
            )
            return loss, outputs_
        else:

            running_loss = torch.tensor(0, dtype=torch.float)
            accumulation = []

            for index, chunk in enumerate(self.chunks):
                accumulation.append(
                    client.submit(
                        self.train_batches,
                        *(
                            index,
                            chunk,
                            self.targets,
                            self.models,
                            self.lossfxn,
                            self.atoms_per_image,
                            self.device,
                        )
                    )
                )

            dask.distributed.wait(accumulation)
            accumulation = client.gather(accumulation)

            grads = {}
            outputs_ = {}
            losses = {}
            for model_index, (outputs, loss, grad) in enumerate(accumulation):
                for model_index in range(len(self.models)):
                    if model_index not in grads.keys():
                        grads[model_index] = []
                        outputs_[model_index] = []
                        losses[model_index] = []
                    running_loss += loss[model_index]
                    losses[model_index].append(loss[model_index])
                    grads[model_index].append(np.array(grad[model_index]))
                    outputs_[model_index].append(outputs[model_index])

            # Sum gradients per model
            for key, grad in grads.items():
                grads[key] = sum(grad)

            # Update the gradients of the model
            for model_index, model in enumerate(self.models):
                for index, param in enumerate(model.parameters()):
                    param.grad = torch.tensor(grads[model_index][index])

            return running_loss, outputs_

    def train_batches(
        self, chunk_index, chunk, targets, models, lossfxn, atoms_per_image, device
    ):
        outputs = self.forward(chunk, models)

        batch_loss = torch.tensor(0, dtype=torch.float)

        losses = []
        for model_index, model in enumerate(models):
            # _losses = []
            name = model.name()
            # for j, output in enumerate(outputs[i]):

            output = outputs[model_index]
            if name == "PytorchPotentials":
                loss = lossfxn[model_index](
                    output,
                    targets[model_index][chunk_index],
                    atoms_per_image[chunk_index],
                )

            elif name == "AutoEncoder":
                loss = lossfxn[model_index](output, targets[model_index][chunk_index])

            batch_loss += loss * self.loss_weights[model_index]
            losses.append(loss)

        # We sum the loss of all models and backward propagate them
        batch_loss.backward()

        gradients = []
        for model in models:
            _gradients = []
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
                    gradient = np.float(0.0)
                _gradients.append(gradient)

            gradients.append(_gradients)

        return outputs, losses, gradients
