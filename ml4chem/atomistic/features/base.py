import torch
import numpy as np
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

    def restack_image(self, index, image, scaled_feature_space, svm):
        """Restack images to correct dictionary's structure to train

        Parameters
        ----------
        index : int
            Index of original hashed image.
        image : obj
            An ASE image object.
        scaled_feature_space : np.array
            A numpy array with scaled features.

        Returns
        -------
        hash, features : tuple
            Hash of image and its corresponding features.
        """
        hash, image = image

        if scaled_feature_space is not None:
            features = []
            for j, atom in enumerate(image):
                symbol = atom.symbol

                scaled = scaled_feature_space[index][j]

                if isinstance(scaled, tuple):
                    symbol, scaled = scaled

                if isinstance(scaled, np.ndarray) is False:
                    scaled = scaled.compute()

                if svm is False:
                    scaled = torch.tensor(
                        scaled, requires_grad=False, dtype=torch.float,
                    )
                features.append((symbol, scaled))

        return hash, features

    def restack_atom(self, image_index, atom, scaled_feature_space):
        """Restack atoms to a raveled list to use with SVM

        Parameters
        ----------
        image_index : int
            Index of original hashed image.
        atom : object
            An atom object.
        scaled_feature_space : np.array
            A numpy array with the scaled features

        Returns
        -------
        symbol, features : tuple
            The hashed key image and its corresponding features.
        """

        features = scaled_feature_space[image_index][atom.index]

        if isinstance(features, tuple):
            symbol, features = features
        else:
            symbol = atom.symbol

        return symbol, features
