from abc import ABC, abstractmethod


class AtomisticFeatures(ABC):
    @abstractmethod
    def name(cls):
        """Return name of the class"""
        pass

    @abstractmethod
    def __init__(self, **kwargs):
        """Arguments needed to instantiate Features"""
        pass

    @abstractmethod
    def calculate(self, **kwargs):
        """Calculate features"""
        pass

    @abstractmethod
    def to_pandas(self):
        """Convert features to pandas DataFrame"""
        pass
