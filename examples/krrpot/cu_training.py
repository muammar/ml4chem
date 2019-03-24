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
    batch_size = 160

    calc = Potentials(fingerprints=Gaussian(cutoff=6.5, normalized=normalized,
                                            save_scaler='cu_training'),
                      model=KernelRidge(batch_size=batch_size),
                      label='cu_training')

    calc.train(training_set=images)


if __name__ == '__main__':
    cluster = LocalCluster(n_workers=8, threads_per_worker=2)
    client = Client(cluster, asyncronous=True)
    train()
