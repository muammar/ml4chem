import logging
import sys
sys.path.append('../../')
from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
from mlchem import Potentials
from mlchem.fingerprints import Gaussian
from mlchem.models.neuralnetwork import NeuralNetwork


def train():
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
    lr = 1.e-2
    weight_decay = 0.
    regularization = 0.

    calc = Potentials(fingerprints=Gaussian(cutoff=6.5, normalized=normalized,
                                            save_scaler='cu_training.scaler'),
                      model=NeuralNetwork(hiddenlayers=(n, n),
                                          activation=activation),
                      label='cu_training')

    optimizer = ('adam', {'lr': lr, 'weight_decay': weight_decay})
    calc.train(training_set=images, epochs=epochs,
               regularization=regularization, convergence=convergence,
               optimizer=optimizer)


if __name__ == '__main__':
    logging.basicConfig(filename='cu_training.log', level=logging.INFO,
                        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    train()
