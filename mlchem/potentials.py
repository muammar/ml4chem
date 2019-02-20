from ase.calculators.calculator import Calculator

from mlchem.data.handler import DataSet
from mlchem.backends.available import available_backends


class Potentials(Calculator, object):
    """Machine-Learning for Chemistry

    This class is highly inspired on the Atomistic Machine-Learning package
    (Amp).

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

    @classmethod
    def load(self, path, backend=None):
        """Load a model
        Parameters
        ----------
        path : str
            The path to load the model for inference.
        backed : str
            Backend used for building the model 'pytorch'.
        """
        backend = backend.lower()

        #if backend == 'pytorch':
        #    import torch
        #    from mlchem.models.neuralnetwork import NeuralNetwork
        #    model.load_state_dict(torch.load(path))


    def save(self, model, path=None):
        """Save a model

        Parameters
        ----------
        model : obj
            The model to be saved.
        path : str
            The path where to save the model.
        """

        model_name = model.name()

        if path is None:
            path = 'model.mlchem'

        if model_name == 'PytorchPotentials':
            import torch
            torch.save(model.state_dict(), path)

    def train(self, training_set, epochs=100, lr=0.001, convergence=None,
              device='cpu',  optimizer=None, lossfxn=None, weight_decay=0.,
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
        lossfxn : object
            A loss function object.
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

        self.model.train(feature_space, targets, model=self.model,
                         data=data_handler, optimizer=optimizer, lr=lr,
                         weight_decay=weight_decay,
                         regularization=regularization, epochs=epochs,
                         convergence=convergence, lossfxn=lossfxn)

        self.save(self.model)

    def calculate(self):
        """docstring for calculate"""
        pass
