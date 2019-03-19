from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
import sys
sys.path.append('../../')
from mlchem import Potentials
from mlchem.fingerprints import Gaussian
from mlchem.models.kernelridge import KernelRidge


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
    lr = 1e-4
    weight_decay = 0.
    regularization = 0.

    calc = Potentials(fingerprints=Gaussian(cutoff=6.5, normalized=normalized,
                                            save_scaler='cu_training'),
                      model=KernelRidge(),
                      label='cu_training')

    calc.train(training_set=images, epochs=epochs, lr=lr,
               weight_decay=weight_decay, regularization=regularization,
               convergence=convergence)


if __name__ == '__main__':
    cluster = LocalCluster(n_workers=8, threads_per_worker=2)
    client = Client(cluster, asyncronous=True)
    train()
