from ase.calculators.calculator import Calculator, Parameters
from ase.io import Trajectory
from mlchem.utils import prepare_images

class MlChem(Calculator, object):
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
        pass

    def load(self):
        """docstring for load"""
        pass

    def train(self, training_set):
        """Method to train the models

        Parameters
        ----------
        training_set : object, list
            List containing the training set.
        """
        training_set = prepare_images(training_set)
        print(len(training_set.keys()))

    def calculate(self):
        """docstring for calculate"""
        pass
