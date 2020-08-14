import types
import torch
from ml4chem.atomistic import Potentials
from abc import ABC, abstractmethod
from collections import OrderedDict


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

    def feature_preparation(self, features, data, purpose="training"):
        """Vectorized data structure
        
        Parameters
        ----------
        features : dict, iter
            An iterator or dictionary. 
        data : obj
            An ML4Chem data object. 
        purpose : str, optional
            Purpose of the features, by default "training"
        
        Returns
        -------
        rearrengements, conditions
            Rearranged features and conditions. 
        """

        data.get_largest_number_atoms(purpose)

        rearrengements = []
        conditions = []

        if isinstance(features, OrderedDict):
            features = [features]

        if isinstance(features, (list, types.GeneratorType)):
            for chunk in features:
                if isinstance(chunk, OrderedDict) == False:
                    chunk = OrderedDict(chunk)
                rearrange = {
                    symbol: [] for symbol in data.unique_element_symbols[purpose]
                }

                for _, values in chunk.items():
                    image = {}
                    for symbol, features_ in values:
                        if symbol not in image.keys():
                            image[symbol] = []
                        image[symbol].append(features_.float())

                    for symbol in data.unique_element_symbols[purpose]:
                        tensors = image.get(symbol)

                        if tensors == None:
                            tensors = [torch.zeros(self.input_dimension)]

                        tensors = torch.stack(tensors)

                        tensor_size = tensors.size()[0]
                        if tensor_size < data.largest_number_atoms[symbol]:

                            diff = data.largest_number_atoms[symbol] - tensor_size
                            expand = torch.zeros(diff, self.input_dimension)
                            tensors = torch.cat([tensors, expand])

                        rearrange[symbol].append(tensors)

                rearrange = {
                    symbol: torch.stack(tensors)
                    for symbol, tensors in rearrange.items()
                }

                condition = {}

                for symbol, tensors in rearrange.items():
                    condition[symbol] = (tensors.sum(dim=2) != 0).float()

                rearrengements.append(rearrange)
                conditions.append(condition)

        return rearrengements, conditions


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
