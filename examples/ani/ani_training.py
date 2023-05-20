import sys
from ase.io import Trajectory
from dask.distributed import Client, LocalCluster

sys.path.append("../../")
from ml4chem.atomistic import Potentials
from ml4chem.atomistic.features.aev import AEV
from ml4chem.atomistic.models.neuralnetwork import NeuralNetwork
from ml4chem.utils import logger
import numpy as np


def train():
    # Load the images with ASE
    images = Trajectory("training.traj")

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

    rcr = 5.1
    eta_r = [19.70000]
    rs = [
        8.0000000e01,
        1.0687500e00,
        1.3375000e00,
        1.6062500e00,
        1.8750000e00,
        2.1437500e00,
        2.4125000e00,
        2.6812500e00,
        2.9500000e00,
        3.2187500e00,
        3.4875000e00,
        3.7562500e00,
        4.0250000e00,
        4.2937500e00,
        4.5625000e00,
        4.8312500e00,
    ]

    rca = 3.5
    eta_a = [12.50000]
    rs_a = [
        8.0000000e-01,
        1.1375000e00,
        1.4750000e00,
        1.8125000e00,
        2.1500000e00,
        2.4875000e00,
        2.8250000e00,
        3.1625000e00,
    ]
    zetas = [14.10000]
    thetas = [3.9269908e-01, 1.1780972e00, 1.9634954e00, 2.7488936e00]

    custom = {
        "G2": {"etas": eta_r, "Rs": rs},
        "G4": {"etas": eta_a, "zetas": zetas, "Rs_a": rs_a, "thetas": thetas},
    }

    cutoff = {"radial": rcr, "angular": rca}

    calc = Potentials(
        features=AEV(
            cutoff=cutoff,
            normalized=normalized,
            custom=custom,
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
    logger("training.log")
    cluster = LocalCluster()
    client = Client(cluster)
    train()
