import torch
import logging
from ml4chem.utils import convert_elapsed_time, dynamic_import, get_chunks

# Setting precision and starting logger object
torch.set_printoptions(precision=10)
logger = logging.getLogger()


class ModelMerger(torch.nn.Module):
    """Model Merger

    A class that can merge models and train them simultaneously. Models are
    executed sequentially, and are taken care of. 

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

        logger.info(" ")
        logging.info("Model Merger")
        logging.info("============")
        logging.info("Merging the following models:")

        for model in self.merge.get("models"):
            logging.info("    - {}.".format(model.name()))

        atoms_per_image = data.atoms_per_image

        # If no batch_size provided then the whole training set length is the batch.
        if batch_size is None:
            batch_size = len(inputs.values())

        if isinstance(batch_size, int):
            self.chunks = list(get_chunks(inputs, batch_size, svm=False))
            self.targets = list(get_chunks(targets, batch_size, svm=False))
            self.atoms_per_image = list(
                get_chunks(atoms_per_image, batch_size, svm=False)
            )

        logger.info(" ")
        logging.info("Batch Information")
        logging.info("-----------------")
        logging.info("Number of batches: {}.".format(len(self.chunks)))
        logging.info("Batch size: {} elements per batch.".format(batch_size))
        logger.info(" ")

        models = self.merge.get("models")

        if lossfxn is None:
            self.lossfxn = [None for model in models]
        else:
            self.lossfxn = lossfxn

        self.device = device

        converged = False
        epoch = 0
        while not converged:
            epoch += 1
            losses = []

            for index, model in enumerate(models):
                name = model.name()
                loss = self.closure(index, name, model)
                print(name)

        raise NotImplementedError

    def closure(self, index, name, model):

        if name == "PytorchPotentials":
            train = dynamic_import("train", "ml4chem.models", alt_name="neuralnetwork")

            loss, outputs_ = train.closure(
                self.chunks,
                self.targets,
                model,
                self.lossfxn[index],
                self.atoms_per_image,
                self.device,
            )
        elif name == "AutoEncoder":
            train = dynamic_import("train", "ml4chem.models", alt_name="autoencoders")

            loss, outputs_ = train.closure(
                self.chunks, self.targets, model, self.lossfxn[index], self.device
            )
            raise NotImplementedError
