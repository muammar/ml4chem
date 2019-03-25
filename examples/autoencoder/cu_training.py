from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
import sys
sys.path.append('../../')
from mlchem.data.handler import DataSet
from mlchem.fingerprints import Gaussian
from mlchem.models.autoencoders import AutoEncoder


def train():
    # Load the images with ASE
    images = Trajectory('cu_training.traj')
    purpose = 'training'

    # Arguments for fingerprinting the images
    normalized = True

    """
    Data Structure Preparation
    """
    data_handler = DataSet(images, purpose=purpose)
    training_set, targets = data_handler.get_images(purpose=purpose)

    """
    Let's create the outputs of the model
    """
    fingerprints = Gaussian(cutoff=6.5, normalized=normalized,
                            save_scaler='cu_training')

    outputs = fingerprints.calculate_features(training_set,
                                              data=data_handler,
                                              purpose=purpose,
                                              svm=False)
    """
    Building AutoEncoder
    """
    # Arguments for building the model
    hiddenlayers = {'encoder': (10, 5),
                    'decoder': (5, 10)}
    activation = 'relu'
    autoencoder = AutoEncoder(hiddenlayers=hiddenlayers,
                              activation=activation)

    output_dimension = len(list(outputs.values())[0][0][1])

    autoencoder.prepare_model(3, output_dimension, data=data_handler)
    ## Arguments for training the potential
    #convergence = {'energy': 5e-3}
    #epochs = 100
    #lr = 1e-4
    #weight_decay = 0.
    #regularization = 0.

    #calc.train(training_set=images, epochs=epochs, lr=lr,
    #           weight_decay=weight_decay, regularization=regularization,
    #           convergence=convergence)


if __name__ == '__main__':
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    train()
