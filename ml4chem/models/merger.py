import dask
import inspect
import torch
import logging
from ml4chem.utils import convert_elapsed_time, dynamic_import, get_chunks

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
    merge : dict
        A dictionary with keys: "models", and "extrafuncs". The
        structure of the dictionary is the following:

        merge = {'models': [list of models],
                 'extra_funcs': [list of extra functions]
                }
    """

    NAME = "Merged"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(self, merge):
        super(ModelMerger, self).__init__()
        self.merge = merge

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
        models = self.merge.get("models")

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

        models = self.merge.get("models")

        logger.info(" ")
        logging.info("Model Merger")
        logging.info("============")
        logging.info("Merging the following models:")

        for model in models:
            logging.info("    - {}.".format(model.name()))

        # If no batch_size provided then the whole training set length is the batch.
        if batch_size is None:
            batch_size = len(inputs.values())

        if isinstance(batch_size, int):
            chunks = [list(get_chunks(inputs_, batch_size, svm=False)) for inputs_ in inputs]
            targets = [list(get_chunks(target, batch_size, svm=False)) for target in targets]
            atoms_per_image = list(get_chunks(data.atoms_per_image, batch_size, svm=False))


        if lossfxn is None:
            self.lossfxn = [None for model in models]
        else:
            self.lossfxn = lossfxn

        self.device = device

        # Population of extra Attributes needed by the models

        for index, loss in enumerate(lossfxn):
            _args, _varargs, _keywords, _defaults = inspect.getargspec(loss)
            if "latent" in _args:
                train = dynamic_import("train", "ml4chem.models", alt_name="autoencoders")
                self.inputs_chunk_vals = train.get_inputs_chunks(chunks[index])


        for index, model in enumerate(models):
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
        self.chunks = [client.scatter(chunk) for chunk in chunks]
        self.targets = [client.scatter(target) for target in targets]

        logger.info(" ")
        logging.info("Batch Information")
        logging.info("-----------------")
        logging.info("Number of batches: {}.".format(len(self.chunks)))
        logging.info("Batch size: {} elements per batch.".format(batch_size))
        logger.info(" ")

        converged = False
        epoch = 0

        while not converged:
            epoch += 1

            for index, model in enumerate(models):
                name = model.name()
                loss = self.closure(index, name, model)

            raise NotImplementedError

    def closure(self, index, name, model):

        loss_array = []

        if name == "PytorchPotentials":
            train = dynamic_import("train", "ml4chem.models", alt_name="neuralnetwork")

            loss, outputs_ = train.closure(
                self.chunks[index],
                self.targets[index],
                model,
                self.lossfxn[index],
                self.atoms_per_image,
                self.device,
            )
            loss_array.append(loss)

        elif name == "AutoEncoder":
            train = dynamic_import("train", "ml4chem.models", alt_name="autoencoders")
            # The [0] is needed because of the way the array is built above.
            targets = self.targets[index][0]

            loss, outputs_ = train.closure(
                self.chunks[index], targets, model, self.lossfxn[index], self.device, self.inputs_chunk_vals
            )
            loss_array.append(loss)