import sys
from ase.io import Trajectory
from dask.distributed import Client, LocalCluster

sys.path.append("../../")
from ml4chem.atomistic import Potentials
from ml4chem.atomistic.features import Gaussian
from ml4chem.atomistic.models.gaussian_process import GaussianProcess
from ml4chem.utils import logger


def train():
    # Load the images with ASE
    images = Trajectory("cu_training.traj")

    # Arguments for fingerprinting the images
    normalized = True

    calc = Potentials(
        features=Gaussian(
            cutoff=6.5, normalized=normalized, save_preprocessor="cu_training.scaler"
        ),
        # model=GaussianProcess(batch_size=batch_size),
        model=GaussianProcess(),
        label="cu_training",
    )

    calc.train(training_set=images)


if __name__ == "__main__":
    logger()
    cluster = LocalCluster()
    client = Client(cluster)
    train()
