from ase.io import Trajectory
from dask.distributed import Client, LocalCluster
import sys
sys.path.append('../../')
from mlchem.data.handler import DataSet
from mlchem.fingerprints import Cartesian, Gaussian
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
    training_set, energy_targets = data_handler.get_images(purpose=purpose)

    #"""
    #Let's create the outputs of the model
    #"""
    fingerprints = Gaussian(cutoff=6.5, normalized=normalized,
                            save_scaler='cu_training.scaler')

    outputs = fingerprints.calculate_features(training_set,
                                              data=data_handler,
                                              purpose=purpose,
                                              svm=False)
    output_dimension = len(list(outputs.values())[0][0][1])

    """
    Input
    """

    features = Cartesian()
    inputs = features.calculate_features(training_set, data=data_handler)

    """
    Building AutoEncoder
    """
    # Arguments for building the model
    hiddenlayers = {'encoder': (20, 10, 4),
                    'decoder': (4, 10, 20)}
    activation = 'tanh'
    autoencoder = AutoEncoder(hiddenlayers=hiddenlayers,
                              activation=activation)

    data_handler.get_unique_element_symbols(images, purpose=purpose)
    autoencoder.prepare_model(3, output_dimension, data=data_handler)
    # Arguments for training the potential
    convergence = {'rmse': 5e-2}
    epochs = 2000
    lr = 1e-3
    weight_decay = 0
    regularization = None

    #targets = [atom.position for atoms in images for atom in atoms]

    train(inputs, outputs, model=autoencoder, data=data_handler,
          optimizer=None, lr=lr, weight_decay=weight_decay,
          regularization=regularization, epochs=epochs,
          convergence=convergence, lossfxn=None)
    latent_space = autoencoder.get_latent_space(inputs, svm=True)
    print(latent_space)

    return latent_space, energy_targets, data_handler

def neural(inputs, targets, data_handler):
    from mlchem.models.neuralnetwork import NeuralNetwork, train
    model = NeuralNetwork(hiddenlayers=(10, 10), activation='relu')
    model.prepare_model(5, data=data_handler, purpose='training')

    lr = 1e-4
    convergence = {'energy': 5e-3}
    weight_decay = 0
    train(inputs, targets, model=model, data=data_handler, lr=lr,
          convergence=convergence, weight_decay=weight_decay)

if __name__ == '__main__':
    cluster = LocalCluster()
    client = Client(cluster, asyncronous=True)
    inputs, outputs, data_handler = autoencode()
    #neural(inputs, outputs, data_handler)
