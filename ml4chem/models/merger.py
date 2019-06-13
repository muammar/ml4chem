import dask
import inspect
import torch
import logging
from collections import OrderedDict
from ml4chem.utils import convert_elapsed_time, dynamic_import, get_chunks
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
        models = self.models 
 
        _x = []

        for index, model in models:
            if index == 0:
                x = model(X)
            else:
                x = model(x)
        return x

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
    ):

        logger.info(" ")
        logging.info("Model Merger")
        logging.info("============")
        logging.info("Merging the following models:")

        for model in self.models:
            print(model)
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

            #chunks = [list(get_chunks(inputs_, batch_size, svm=False)) for inputs_ in inputs if inspect.ismethod(inputs_) is False]
            targets = [list(get_chunks(target, batch_size, svm=False)) for target in targets]
            atoms_per_image = list(get_chunks(data.atoms_per_image, batch_size, svm=False))


        if lossfxn is None:
            self.lossfxn = [None for model in self.models]
        else:
            self.lossfxn = lossfxn

        self.device = device

        # Population of extra Attributes needed by the models

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

        # Data scattering
        client = dask.distributed.get_client()

        self.chunks = []
        for chunk in chunks:
            if inspect.ismethod(inputs_) is False:
                self.chunks.append(client.scatter(chunk))
            else:
                self.chunks.append(chunk)

        #self.chunks = [client.scatter(chunk) for chunk in chunks]
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

            losses = []
            for index, model in enumerate(self.models):
                name = model.name()
                loss, outputs = self.closure(index, name, model)
                losses.append(loss)

            print(losses)
            if self.optimizer_name != "LBFGS":
                self.optimizer.step()
            else:
                options = {"closure": self.closure, "current_loss": loss, "max_ls": 10}
                self.optimizer.step(options)

            #raise NotImplementedError

    def closure(self, index, name, model):

        if name == "PytorchPotentials":
            train = dynamic_import("train", "ml4chem.models", alt_name="neuralnetwork")
            client = dask.distributed.get_client()

            inputs = []
            for chunk in self.chunks[index-1]:
                inputs_ = self.chunks[index](OrderedDict(chunk))
                #try:
                #    print(inputs_['b57e74028a8b592a0e35f83ee13f601fd9fa5dbf'][0][1])
                #except:
                #    pass
                inputs.append(client.scatter(inputs_))

            loss, outputs_ = train.closure(
                inputs,
                self.targets[index],
                model,
                self.lossfxn[index],
                self.atoms_per_image,
                self.device,
            )

        elif name == "AutoEncoder":
            train = dynamic_import("train", "ml4chem.models", alt_name="autoencoders")
            # The indexing [0] is needed because of the way the array is built
            # above.
            targets = self.targets[index][0]

            loss, outputs_ = train.closure(
                self.chunks[index], targets, model, self.lossfxn[index], self.device, self.inputs_chunk_vals
            )
        
        return loss, outputs_
