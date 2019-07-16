import sys
import torch
from ase.io import Trajectory
from dask.distributed import Client, LocalCluster

sys.path.append("../../")
from ml4chem.potentials import Potentials
from ml4chem.fingerprints.gaussian import Gaussian
from ml4chem.models.ionic import NeuralNetwork, train
from ml4chem.utils import logger
from ml4chem.data.handler import DataSet
from ml4chem.optim.handler import get_optimizer


def train():
    # Load the images with ASE
    images = Trajectory("cu_training.traj")

    # Arguments for fingerprinting the images
    normalized = True

    # Arguments for building the model
    n = 3
    activation = "tanh"

    # Arguments for training the potential
    convergence = None
    epochs = 15
    lr = 1.0e-2
    weight_decay = 0.0
    regularization = 0.0

    calc = Potentials(
        fingerprints=Gaussian(
            cutoff=6.5,
            normalized=normalized,
            save_preprocessor="model.scaler",
            angular_type="G4",
        ),
        model=NeuralNetwork(hiddenlayers=(n, n), activation=activation),
        label="cu_training_ionic",
        path="",
        ionic=True,
    )
    optimizer = ("adam", {"lr": lr, "weight_decay": weight_decay})

    calc.train(
        training_set=images,
        epochs=epochs,
        regularization=regularization,
        convergence=convergence,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    logger()
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    train()
