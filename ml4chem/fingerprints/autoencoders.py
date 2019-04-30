import dask
import json
import logging
import time
import torch
import numpy as np
from collections import OrderedDict
from ml4chem.data.preprocessing import Preprocessing
from ml4chem.data.serialization import dump
from ml4chem.utils import convert_elapsed_time, dynamic_import

logger = logging.getLogger()


class LatentFeatures(object):
    """Extraction of features using AutoEncoder model class.

    The latent space represents a feature space from the inputs that an
    AutoEncoder model finds relevant about the underlying structure of the
    data. This class takes images in ASE format and returns them converted in
    a latent feature vector using the encoder layer of an AutoEncoder model
    already hashed to be used by ML4Chem. It also allows interoperability with
    the Potentials() class.


    Parameters
    ----------
    encoder : obj
        A ML4Chem AutoEncoder model to extract atomic features.
    scheduler : str
        The scheduler to be used with the dask backend.
    filename : str
        Name to save on disk of serialized database.
    preprocessor : tuple
        Use some scaling method to preprocess the data.
    save_preprocessor : str
        Save preprocessor to file.
    features : obj or tuple
        Users can set the features keyword argument to a tuple with the
        structure ('Name', {kwargs})
    """
    NAME = 'LatentFeatures'

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(self, encoder=None, scheduler='distributed',
                 filename='latent.db', preprocessor=None,
                 save_preprocessor='ml4chem', features=None):

        self.encoder = encoder
        self.filename = filename
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.save_preprocessor = save_preprocessor

        if features is None:
            # Add user-defined exception?
            # https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
            error = 'A fingerprint object or tuple has to be provided.'
            logger.error(error)
        else:
            self.features = features

        # Let's add parameters that are going to be stored in the .params json
        # file.
        self.params = OrderedDict()
        self.params['name'] = self.name()

    def calculate_features(self, images, purpose='training', data=None,
                           svm=False):
        """Return features per atom in an atoms object

        Parameters
        ----------
        image : dict
            Hashed images using the DataSet class.
        purpose : str
            The supported purposes are: 'training', 'inference'.
        data : obj
            data object
        svm : bool
            Whether or not these features are going to be used for kernel
            methods.

        Returns
        -------
        feature_space : dict
            A dictionary with key hash and value as a list with the following
            structure: {'hash': [('H', [vector]]}
        """
        # Now, we need to take the inputs and convert them to the right feature
        # space
        name, kwargs = self.features
        features = dynamic_import(name, 'ml4chem.fingerprints')
        features = features(**kwargs)
        feature_space = features.calculate_features(images, data=data,
                                                    purpose=purpose, svm=svm)

        encoder = self.load_encoder(self.encoder, data=data, purpose=purpose)

        latent_space = encoder.get_latent_space(feature_space, svm=svm)

        return latent_space

    def load_encoder(self, encoder, **kwargs):
        """Load an autoencoder in eval() mode"""

        params_path = encoder.get('params')
        model_path = encoder.get('model')

        model_params = json.load(open(params_path, 'r'))
        model_params = model_params.get('model')
        name = model_params.pop('name')
        del model_params['type']   # delete unneeded key, value

        input_dimension = model_params.pop('input_dimension')
        output_dimension = model_params.pop('output_dimension')

        autoencoder = dynamic_import(name, 'ml4chem.models',
                                     alt_name='autoencoders')
        autoencoder = autoencoder(**model_params)
        autoencoder.prepare_model(input_dimension, output_dimension, **kwargs)
        autoencoder.load_state_dict(torch.load(model_path), strict=True)

        return autoencoder.eval()
