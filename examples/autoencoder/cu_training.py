from ase.io import Trajectory
import torch
from dask.distributed import Client, LocalCluster
from collections import OrderedDict
import sys
sys.path.append('../../')
from mlchem.fingerprints import Gaussian
from mlchem.models.autoencoders import AutoEncoder
from mlchem.data.handler import DataSet


def train():
    # Load and process the images
    images = Trajectory('cu_training.traj')
    """
    Preparing inputs
    """
    data_handler = DataSet(images, purpose='training')
    training_set, targets = data_handler.get_images(purpose='training')

    inputs = OrderedDict()
    for hash, image in training_set.items():
        inputs[hash] = []
        for atom in image:
            symbol = atom.symbol
            position = torch.tensor(atom.position, requires_grad=True,
                                    dtype=torch.float)
            inputs[hash].append((symbol, position))


    # Arguments for fingerprinting the images
    normalized = True

    fingerprints = Gaussian(cutoff=6.5, normalized=normalized,
                            save_scaler='cu_training')
    outputs = fingerprints.calculate_features(training_set, data=data_handler,
                                             purpose='training', svm=False)

    # Arguments for building the model
    n = 10
    activation = 'relu'

    # Arguments for training the potential
    convergence = {'energy': 5e-3}
    epochs = 100
    lr = 1e-4
    weight_decay = 0.
    regularization = 0.


#    calc.train(training_set=images, epochs=epochs, lr=lr,
#               weight_decay=weight_decay, regularization=regularization,
#               convergence=convergence)
#

if __name__ == '__main__':
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    train()
