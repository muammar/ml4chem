import dask
import inspect
import torch
import logging
from collections import OrderedDict
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

    def forward(self, X):
        """Forward propagation
        
        Parameters
        ----------
        X : list
            List of model inputs. 
        
        Returns
        -------
        x
            A list with the forward propagation evaluation.
        """

        outputs = []
        for i, model in enumerate(self.models):
            _output = []
            name = model.name()
            _X = X[i]
            if inspect.ismethod(_X):
                _X = X[i - 1]

            for j, x in enumerate(_X):
                # print(i, j, name)
                if name == "AutoEncoder":
                    x = OrderedDict(x)
                    output = model(x)
                elif name == "PytorchPotentials":
                    x = X[i](OrderedDict(x))
                    output = model(x)
                
                _output.append(output)

            outputs.append(_output)
        return outputs

    def train(
        self,
        inputs=None,
        targets=None,
        data=None,
        optimizer=(None, None),
        regularization=None,
        epochs=100,
        convergence=None,
        lossfxn=None,
        device="cpu",
        batch_size=None,
        lr_scheduler=None,
        independent_loss=True
    ):

        logger.info(" ")
        logging.info("Model Merger")
        logging.info("============")
        logging.info("Merging the following models:")

        for model in self.models:
            logging.info("    - {}.".format(model.name()))

        # If no batch_size provided then the whole training set length is the batch.
        if batch_size is None:
            batch_size = len(inputs.values())

        if isinstance(batch_size, int):
            chunks = []
            for inputs_ in inputs:
                if inspect.ismethod(inputs_) is False:
                    chunks.append(list(get_chunks(inputs_, batch_size, svm=False)))
                else:
                    chunks.append(inputs_)

            targets = [list(get_chunks(target, batch_size, svm=False)) for target in targets]
            atoms_per_image = list(get_chunks(data.atoms_per_image, batch_size, svm=False))

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
                train = dynamic_import("train", "ml4chem.models", alt_name="autoencoders")
                self.inputs_chunk_vals = train.get_inputs_chunks(chunks[index])


        parameters = []
        for index, model in enumerate(self.models):
            parameters += model.parameters()
            if model.name() == 'PytorchPotentials':
                # These models require targets as tensors
                self.atoms_per_image = torch.tensor(atoms_per_image,
                                                    requires_grad=False,
                                                    dtype=torch.float)
                _targets = [torch.tensor(batch, requires_grad=False) for batch in targets[index]]
                targets[index] = _targets
                del _targets
            elif model.name() == 'AutoEncoder':
                targets[index] = lod_to_list(targets[index])

        # Data scattering
        client = dask.distributed.get_client()

        self.chunks = []
        for chunk in chunks:
            if inspect.ismethod(inputs_) is False:
                self.chunks.append(client.scatter(chunk))
            else:
                self.chunks.append(chunk)

        self.targets = [client.scatter(target) for target in targets]

        logger.info(" ")
        logging.info("Batch Information")
        logging.info("-----------------")
        logging.info("Number of batches: {}.".format(len(self.chunks)))
        logging.info("Batch size: {} elements per batch.".format(batch_size))
        logger.info(" ")

        # Define optimizer

        self.optimizer_name, self.optimizer = get_optimizer(
            optimizer, parameters
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

        converged = False
        epoch = 0

        while not converged:
            epoch += 1

            self.optimizer.zero_grad()  # clear previous gradients

            if independent_loss:
                losses = []
                for index, model in enumerate(self.models):
                    name = model.name()
                    loss, outputs = self.closure(index, model, independent_loss, name=name)
                    losses.append(loss)

                print(losses)
            else:
                    loss, outputs = self.closure(index, self.models, independent_loss)
                    print(loss)
                    loss.backward()

            if self.optimizer_name != "LBFGS":
                self.optimizer.step()
            else:
                options = {"closure": self.closure, "current_loss": loss, "max_ls": 10}
                self.optimizer.step(options)

            #raise NotImplementedError

    def closure(self, index, model, independent_loss, name=None):

        if name == "PytorchPotentials" and independent_loss:
            train = dynamic_import("train", "ml4chem.models", alt_name="neuralnetwork")
            client = dask.distributed.get_client()

            inputs = []
            for chunk in self.chunks[index-1]:
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
                self.chunks[index], targets, model, self.lossfxn[index], self.device, self.inputs_chunk_vals
            )
            return loss, outputs_
        else:
            outputs = self.forward(self.chunks)

            running_loss = torch.tensor(0, dtype=torch.float)

            for i, model in enumerate(self.models):
                name = model.name()
                for j, output in enumerate(outputs[i]):

                    if name == 'PytorchPotentials':
                        loss = self.lossfxn[i](output, self.targets[i][j].result(), self.atoms_per_image[index])

                    elif name == 'AutoEncoder':
                        targets = self.targets[i][j].result()
                        loss = self.lossfxn[i](output, targets)
                    print(i, j, loss)
                    running_loss += loss
            
            return running_loss, outputs
