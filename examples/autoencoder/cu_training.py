import sys

sys.path.append("../../")
from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
from ml4chem import Potentials
from ml4chem.data.handler import Data
from ml4chem.features import Gaussian
from ml4chem.models.autoencoders import AutoEncoder, train
from ml4chem.data.serialization import dump
from ml4chem.utils import logger


def autoencode():
    # Load the images with ASE
    images = Trajectory("cu_training.traj")
    purpose = "training"

    # Arguments for fingerprinting the images
    normalized = True

    """
    Data Structure Preparation
    """
    data_handler = Data(images, purpose=purpose)
    training_set, energy_targets = data_handler.get_data(purpose=purpose)

    """
    Let's create the targets of the model
    """
    features = Gaussian(
        cutoff=6.5, normalized=normalized, save_preprocessor="cu_training.scaler"
    )

    targets = features.calculate(
        training_set, data=data_handler, purpose=purpose, svm=False
    )
    output_dimension = len(list(targets.values())[0][0][1])

    """
    Building AutoEncoder
    """
    # Arguments for building the model
    hiddenlayers = {"encoder": (20, 10, 4), "decoder": (4, 10, 20)}
    activation = "tanh"
    autoencoder = AutoEncoder(hiddenlayers=hiddenlayers, activation=activation)

    data_handler.get_unique_element_symbols(images, purpose=purpose)
    autoencoder.prepare_model(output_dimension, output_dimension, data=data_handler)
    # Arguments for training the potential
    convergence = {"rmse": 5e-2}
    epochs = 2000
    lr = 1e-3
    weight_decay = 0
    regularization = None

    optimizer = ("adam", {"lr": lr, "weight_decay": weight_decay, "amsgrad": True})

    inputs = targets
    train(
        inputs,
        targets,
        model=autoencoder,
        data=data_handler,
        optimizer=optimizer,
        regularization=regularization,
        epochs=epochs,
        convergence=convergence,
        lossfxn=None,
        device="cpu",
    )

    latent_space = autoencoder.get_latent_space(targets, svm=True)

    dump(latent_space, filename="cu_training.latent")

    Potentials.save(autoencoder)

    return latent_space, energy_targets, data_handler


if __name__ == "__main__":
    logger(filename="cu_training.log")
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    inputs, outputs, data_handler = autoencode()
