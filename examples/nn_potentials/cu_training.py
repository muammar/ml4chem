import sys
from ase.io import Trajectory
from dask.distributed import Client, LocalCluster

sys.path.append("../../")
from ml4chem import Potentials
from ml4chem.atomistic.features import Gaussian
from ml4chem.atomistic.models.neuralnetwork import NeuralNetwork
from ml4chem.utils import logger


def train():
    # Load the images with ASE
    images = Trajectory("cu_training.traj")

    # Arguments for fingerprinting the images
    normalized = True

    # Arguments for building the model
    n = 10
    activation = "relu"

    # Arguments for training the potential
    convergence = {"energy": 5e-3}
    epochs = 100
    lr = 1.0e-2
    weight_decay = 0.0
    regularization = 0.0

    calc = Potentials(
        features=Gaussian(
            cutoff=6.5, normalized=normalized, save_preprocessor="model.scaler"
        ),
        model=NeuralNetwork(hiddenlayers=(n, n), activation=activation),
        label="cu_training",
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
    logger(filename="cu_training.log")
    cluster = LocalCluster()
    client = Client(cluster)
    train()
