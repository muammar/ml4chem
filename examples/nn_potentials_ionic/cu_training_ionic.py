import sys
import torch
from ase.io import Trajectory, write, read
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
    images1 = Trajectory("cu_training_save.traj")
    images2 = Trajectory("xyz.traj")
    images1 = Trajectory("cu_training_save.traj")

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

    charges = {"HH": 0.0, "OO": -2.0, "OHH": 0.0, "OOHH":0.0, "CuCuCuCu": 1.0, "HHOO":0.0}

    calc = Potentials(
        fingerprints=Gaussian(
            cutoff=1.5,
            normalized=normalized,
            save_preprocessor="model.scaler",
            angular_type="G4",
            overwrite=True,
        ),
        model=NeuralNetwork(hiddenlayers=(n, n), activation=activation,
         charges=charges, data_type="diatomic_molecules", parameterize_alpha_hardness=True),
        label="cu_training_ionic",
        
    )
    optimizer = ("adam", {"lr": lr, "weight_decay": weight_decay})

    calc.train(
        training_set=images2,
        epochs=epochs,
        regularization=regularization,
        convergence=convergence,
        optimizer=optimizer,
    )

    data_handler = DataSet(images2, purpose="training")
    # Raw input and targets aka X, y
    training_set, targets = data_handler.get_images(purpose="training")

    latent_space = calc.model.latent_space
    latent_space = latent_space[0]
    temp = 0.0
    energies = []
    errors = []
    counter = 0
    average = 0.0
    for hash in latent_space:
        energies.append(latent_space[hash][0])
        temp = 0.0
        for j in range(len(latent_space[hash][0])):
            temp += latent_space[hash][0][j]
        temp = temp - targets[counter]
        temp /= targets[counter]
        temp = abs(temp)
        temp *= 100
        average += temp
        errors.append(temp)
        if temp >= 1:
            print("OVER 1 PERCENT ERROR")
        counter += 1
    print("errors")
    print(errors)
    average /= len(latent_space)
    print("average error: " + str(average))
    print("printing latent_space")
    print(latent_space)


if __name__ == "__main__":
    logger()
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    train()
