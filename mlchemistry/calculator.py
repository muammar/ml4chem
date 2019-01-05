from ase.calculators.calculator import Calculator, Parameters
from ase.io import Trajectory

from mlchemistry.data.handler import Data
from mlchemistry.backends.available import available_backends


class MlChemistry(Calculator, object):
    """Machine-Learning for Chemistry

    Parameters
    ----------
    fingerprints : object
        Local chemical environments to build the feature space.
    model : object
        Machine-learning model to perform training.
    """
    # This is needed by ASE
    implemented_properties = ['energy', 'forces']

    def __init__(self, fingerprints=None, model=None):
        print('Starting MLChem')
        self.fingerprints = fingerprints
        self.available_backends = available_backends()
        print('Available backends', self.available_backends)

        self.model = model

    def load(self):
        """docstring for load"""
        pass

    def train(self, training_set):
        """Method to train models

        Parameters
        ----------
        training_set : object, list
            List containing the training set.
        """
        data_handler = Data()

        # Raw input and targets aka y
        training_set, targets = data_handler.prepare_images(training_set)

        # Mapping raw positions into a feature space aka X
        feature_space = self.fingerprints.calculate_features(training_set)

        # Now let's train
        self.model.train(feature_space, targets)

    def calculate(self):
        """docstring for calculate"""
        pass
