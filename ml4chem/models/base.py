from abc import ABC, abstractmethod
import torch


class DeepLearningModel(ABC, torch.nn.Module):
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
