from ase.calculators.calculator import Calculator, Parameters
from ase.io import Trajectory

from mlchemistry.data.handler import DataSet
from mlchemistry.backends.available import available_backends


class Potentials(Calculator, object):
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
        self.fingerprints = fingerprints
        self.available_backends = available_backends()
        print('Available backends', self.available_backends)

        self.model = model

    def load(self):
        """docstring for load"""
        pass

    def train(self, training_set, epochs=100, lr=0.001, convergence=None,
              device='cpu',  optimizer=None, weight_decay=0.,
              regularization=0.):
        """Method to train models

        Parameters
        ----------
        training_set : object, list
            List containing the training set.
        epochs : int
            Number of full training cycles.
        lr : float
            Learning rate.
        convergence : dict
            Instead of using epochs, users can set a convergence criterion.
        device : str
            Calculation can be run in the cpu or gpu.
        optimizer : object
            An optimizer class.
        weight_decay : float
            Weight decay passed to the optimizer. Default is 0.
        regularization : float
            This is the L2 regularization. It is not the same as weight decay.
        """

        data_handler = DataSet(training_set, self.model, purpose='training')

        # Raw input and targets aka X, y
        training_set, targets = data_handler.get_images(purpose='training')

        # Mapping raw positions into a feature space aka X
        feature_space = self.fingerprints.calculate_features(training_set,
                                                             data=data_handler)

        # Now let's train
        # Fixed fingerprint dimension
        input_dimension = len(list(feature_space.values())[0][0][-1])
        self.model.prepare_model(input_dimension, data=data_handler)

        from mlchemistry.models.neuralnetwork import train
        train(feature_space, targets, model=self.model, data=data_handler,
              optimizer=optimizer, lr=lr, weight_decay=weight_decay,
              regularization=regularization, epochs=epochs)

    def calculate(self):
        """docstring for calculate"""
        pass
