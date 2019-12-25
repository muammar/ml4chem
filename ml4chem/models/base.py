from abc import ABC, abstractmethod


class DeepLearningModel(ABC):
    @abstractmethod
    def name(cls):
        pass

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def prepare_model(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, X):
        pass
