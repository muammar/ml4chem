from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
import sys

sys.path.append("../../")
from ml4chem.data.handler import Data
from ml4chem.atomistic.features import Cartesian
from ml4chem.atomistic.models.autoencoders import AutoEncoder
from ml4chem.atomistic.models.neuralnetwork import NeuralNetwork
from ml4chem.atomistic.models.merger import ModelMerger
from ml4chem.atomistic.models.loss import MSELoss
from ml4chem.atomistic import Potentials
from ml4chem.utils import logger
from ml4chem.atomistic.models.loss import AtomicMSELoss
from ml4chem.data.serialization import dump


def hybrid():
    # Load the images with ASE, and prepare data handler
    images = Trajectory("cu_training.traj")
    purpose = "training"

    latent_dimension = 32
    data_handler = Data(images, purpose=purpose)
    data_handler.get_unique_element_symbols(images, purpose=purpose)
    training_set, energy_targets = data_handler.get_data(purpose=purpose)

    # Preprocessor setup
    preprocessor = ("MinMaxScaler", {"feature_range": (-1, 1)})

    """
    Preparing the input
    """
    features = Cartesian(
        preprocessor=preprocessor, save_preprocessor="cartesian.scaler"
    )
    _inputs = features.calculate(training_set, data=data_handler)

    """
    Building AutoEncoder Model1
    """
    # Arguments for building the model
    hiddenlayers = {
        "encoder": (144, 72, latent_dimension),
        "decoder": (latent_dimension, 72, 144),
    }
    # hiddenlayers = {"encoder": (2, 2, 2), "decoder": (2, 2, 2)}
    activation = "tanh"
    autoencoder = AutoEncoder(hiddenlayers=hiddenlayers, activation=activation)
    autoencoder.prepare_model(3, 3, data=data_handler)

    """
    Building the ml potential model
    """

    # Arguments for building the model
    n = 40
    activation = "tanh"

    nn = NeuralNetwork(hiddenlayers=(n, n), activation=activation)
    nn.prepare_model(latent_dimension, data=data_handler)

    models = [autoencoder, nn]
    losses = [MSELoss, AtomicMSELoss]
    # losses = [EncoderMapLoss, AtomicMSELoss]

    merged = ModelMerger(models)
    # Arguments for training the potential
    convergence = {"rmse": [1.5e-1, 1.0e-1]}
    lr = 1e-4
    weight_decay = 1e-5
    regularization = None

    # Optimizer
    optimizer = ("adam", {"lr": lr, "weight_decay": weight_decay, "amsgrad": True})
    lr_scheduler = None

    inputs = [_inputs, autoencoder.get_latent_space]
    targets = [_inputs, energy_targets]
    batch_size = 2

    merged.train(
        inputs=inputs,
        targets=targets,
        data=data_handler,
        regularization=regularization,
        convergence=convergence,
        optimizer=optimizer,
        device="cpu",
        batch_size=batch_size,
        lr_scheduler=lr_scheduler,
        lossfxn=losses,
        independent_loss=True,
    )

    for index, model in enumerate(merged.models):
        label = "{}_{}".format(index, model.name())
        Potentials.save(model, label=label)

    dump_ls = merged.models[0].get_latent_space(inputs[0])
    dump(dump_ls, filename="checkme.latent")


if __name__ == "__main__":
    logger()
    cluster = LocalCluster(n_workers=5, threads_per_worker=2, dashboard_address=8798)
    client = Client(cluster)
    # Let's do this
    hybrid()
