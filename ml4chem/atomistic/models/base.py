from ml4chem.atomistic import Potentials
from abc import ABC, abstractmethod


class DeepLearningModel(ABC):
    @abstractmethod
    def name(cls):
        """Return name of the class"""
        return cls.NAME

    @abstractmethod
    def __init__(self, **kwargs):
        """Arguments needed to instantiate the model"""
        pass

    @abstractmethod
    def prepare_model(self, **kwargs):
        """Prepare model for training or inference"""
        pass

    @abstractmethod
    def forward(self, X):
        """Forward propagation pass"""
        pass


class DeepLearningTrainer(ABC, object):
    def checkpoint_save(self, epoch, model, label=None, checkpoint=None, path=""):
        """Checkpoint saver

        A method that saves the checkpoint of a model during training.

        Parameters
        ----------
        epoch : int
            Epoch number.
        model : object
            A DeepLearning object.
        label : str, optional
            String with checkpoint label, by default None.
        checkpoint : int, optional
            Set checkpoints. If set to 100, at each 100 epoch the model will be
            saved. Use -1 to save each epoch. Default is None.
        path : str, optional
            Path to save the checkpoint, by default "".
        """

        if label is None:
            label = f"checkpoint-{epoch}"
        else:
            label = f"{label}-checkpoint-{epoch}"

        if checkpoint is None:
            pass
        elif checkpoint == -1:
            Potentials.save(model=model, label=label, path=path)
        elif epoch % checkpoint == 0:
            Potentials.save(model=model, label=label, path=path)
