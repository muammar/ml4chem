import sys

sys.path.append("../../")
from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
from ml4chem.data.handler import DataSet
from ml4chem.fingerprints import LatentFeatures
from ml4chem.data.serialization import load
from ml4chem.utils import logger


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
    images, energies = data_handler.get_data(purpose=purpose)

    fingerprints = (
        "Gaussian",
        {
            "cutoff": 6.5,
            "normalized": normalized,
            "save_preprocessor": "inference.scaler",
        },
    )
    encoder = {"model": "ml4chem.ml4c", "params": "ml4chem.params"}
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
    logger("cu_inference.log")
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    autoencode()
