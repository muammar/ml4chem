import logging
import sys

sys.path.append("../../")
from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
from ml4chem.data.handler import DataSet
from ml4chem.fingerprints import LatentFeatures
from ml4chem.models.autoencoders import AutoEncoder, train
from ml4chem.data.serialization import load
import json
import torch


def autoencode():
    # Load the images with ASE
    latent_space = load("cu_training.latent")
    print("Latent space from file")
    print(latent_space)

    images = Trajectory("cu_training.traj")
    purpose = "training"

    # Arguments for fingerprinting the images
    normalized = True

    data_handler = DataSet(images, purpose=purpose)
    images, energies = data_handler.get_images(purpose=purpose)

    fingerprints = (
        "Gaussian",
        {
            "cutoff": 6.5,
            "normalized": normalized,
            "save_preprocessor": "inference.scaler",
        },
    )
    encoder = {"model": "model.ml4c", "params": "model.params"}
    preprocessor = ("MinMaxScaler", {"feature_range": (-1, 1)})

    fingerprints = LatentFeatures(
        features=fingerprints,
        encoder=encoder,
        preprocessor=preprocessor,
        save_preprocessor="latent_space_min_max.scaler",
    )
    fingerprints = fingerprints.calculate_features(
        images, purpose=purpose, data=data_handler, svm=False
    )

    print("Latent space from LatentFeatures class")
    print(fingerprints)


if __name__ == "__main__":
    # logging.basicConfig(filename='cu_inference.log', level=logging.INFO,
    logging.basicConfig(
        level=logging.INFO, format="%(filename)s:%(lineno)s %(levelname)s:%(message)s"
    )
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    autoencode()
