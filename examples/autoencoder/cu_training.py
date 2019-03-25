from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
import sys
sys.path.append('../../')
from mlchem.data.handler import DataSet
from mlchem.fingerprints import Cartesian
from mlchem.models.autoencoders import AutoEncoder, train


def autoencode():
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

    #"""
    #Let's create the outputs of the model
    #"""
    #fingerprints = Gaussian(cutoff=6.5, normalized=normalized,
    #                        save_scaler='cu_training')

    #outputs = fingerprints.calculate_features(training_set,
    #                                          data=data_handler,
    #                                          purpose=purpose,
    #                                          svm=False)
    #output_dimension = len(list(outputs.values())[0][0][1])

    """
    Input
    """

    features = Cartesian()
    inputs = features.calculate_features(training_set, data=data_handler)

    """
    Building AutoEncoder
    """
    # Arguments for building the model
    hiddenlayers = {'encoder': (20, 10, 5),
                    'decoder': (5, 10, 20)}
    activation = 'tanh'
    autoencoder = AutoEncoder(hiddenlayers=hiddenlayers,
                              activation=activation)

    data_handler.get_unique_element_symbols(images, purpose=purpose)
    autoencoder.prepare_model(3, 3, data=data_handler)
    # Arguments for training the potential
    convergence = {'rmse': 5e-3}
    epochs = 2000
    lr = 1e-3
    weight_decay = 0
    regularization = None

    targets = [atom.position for atoms in images for atom in atoms]

    train(inputs, targets, model=autoencoder, data=data_handler,
            optimizer=None, lr=lr, weight_decay=weight_decay,
            regularization=regularization, epochs=epochs,
            convergence=convergence, lossfxn=None)

if __name__ == '__main__':
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    autoencode()
