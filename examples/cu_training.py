import sys
sys.path.append('../')
from ase.io import Trajectory
from mlchem import Potentials
from mlchem.fingerprints import Gaussian
from mlchem.models.neuralnetwork import NeuralNetwork

# Load the images with ASE
images = Trajectory('cu_training.traj')

# Arguments for fingerprinting the images
normalized = True

# Arguments for building the model
n = 10
activation = 'relu'

# Arguments for training the potential
convergence = {'energy': 5e-3}
epochs = 100
lr = 1e-4
weight_decay = 0.
regularization = 0.
cores = 4

calc = Potentials(fingerprints=Gaussian(cutoff=6.5, normalized=normalized,
                                        save_scaler='cu_training',
                                        cores=cores),
                  model=NeuralNetwork(hiddenlayers=(n, n),
                                      activation=activation),
                  label='cu_training'
                  )

calc.train(training_set=images, epochs=epochs, lr=lr,
           weight_decay=weight_decay, regularization=regularization,
           convergence=convergence)
