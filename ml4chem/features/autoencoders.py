import json
import logging
import pandas as pd
import torch
from collections import OrderedDict
from ml4chem.data.preprocessing import Preprocessing
from ml4chem.features.base import AtomisticFeatures
from ml4chem.utils import dynamic_import

# Starting logger object
logger = logging.getLogger()


class LatentFeatures(AtomisticFeatures):
    """Extraction of features using AutoEncoder model class.

    The latent space represents a feature space from the inputs that an
    AutoEncoder model finds relevant about the underlying structure of the
    data. This class takes images in ASE format and returns them converted in
    a latent feature vector using the encoder layer of an AutoEncoder model
    already hashed to be used by ML4Chem. It also allows interoperability with
    the Potentials() class.


    Parameters
    ----------
    encoder : dict
        Dictionary with structure:
            >>> encoder = {'model': file.ml4c, 'params': file.params}

    scheduler : str
        The scheduler to be used with the dask backend.
    filename : str
        Name to save on disk of serialized database.
    preprocessor : tuple
        Use some scaling method to preprocess the data.
    features : tuple
        Users can set the features keyword argument to a tuple with the
        structure ('Name', {kwargs})
    save_preprocessor : str
        Save preprocessor to file.
    """

    NAME = "LatentFeatures"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(
        self,
        encoder=None,
        scheduler="distributed",
        filename="latent.db",
        preprocessor=None,
        features=None,
        save_preprocessor="latentfeatures.scaler",
    ):
        self.encoder = encoder
        self.filename = filename
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.save_preprocessor = save_preprocessor

        # TODO features could be passed as a dictionary, too?
        if features is None:
            # Add user-defined exception?
            # https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
            error = "A fingerprint object or tuple has to be provided."
            logger.error(error)
        else:
            self.features = features

        # Let's add parameters that are going to be stored in the .params json
        # file.
        self.params = OrderedDict()
        self.params["name"] = self.name()

    def calculate(self, images, purpose="training", data=None, svm=False):
        """Return features per atom in an atoms object

        Parameters
        ----------
        images : dict
            Hashed images using the Data class.
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
        features = dynamic_import(name, "ml4chem.features")
        features = features(**kwargs)

        feature_space = features.calculate(
            images, data=data, purpose=purpose, svm=False
        )

        preprocessor = Preprocessing(self.preprocessor, purpose=purpose)
        preprocessor.set(purpose=purpose)

        encoder = self.load_encoder(self.encoder, data=data, purpose=purpose)

        if self.preprocessor is not None and purpose == "training":
            hashes, symbols, _latent_space = encoder.get_latent_space(
                feature_space, svm=True, purpose="preprocessing"
            )
            _latent_space = preprocessor.fit(_latent_space, scheduler=self.scheduler)

            latent_space = OrderedDict()

            # TODO parallelize this.
            index = 0
            for i, hash in enumerate(hashes):
                pairs = []

                for symbol in symbols[i]:
                    feature_vector = _latent_space[index]

                    if svm is False:
                        feature_vector = torch.tensor(
                            feature_vector, requires_grad=False, dtype=torch.float
                        )

                    pairs.append((symbol, feature_vector))
                    index += 1

                latent_space[hash] = pairs

            del _latent_space

            # Save preprocessor.
            preprocessor.save_to_file(preprocessor, self.save_preprocessor)

        elif self.preprocessor is not None and purpose == "inference":
            hashes, symbols, _latent_space = encoder.get_latent_space(
                feature_space, svm=True, purpose="preprocessing"
            )
            scaled_latent_space = preprocessor.transform(_latent_space)

            latent_space = OrderedDict()
            # TODO parallelize this.
            index = 0
            for i, hash in enumerate(hashes):
                pairs = []

                for symbol in symbols[i]:
                    feature_vector = scaled_latent_space[index]

                    if svm is False:
                        feature_vector = torch.tensor(
                            feature_vector, requires_grad=False, dtype=torch.float
                        )

                    pairs.append((symbol, feature_vector))
                    index += 1

                latent_space[hash] = pairs

            del _latent_space

        else:
            if encoder.name() == "VAE":
                purpose = "inference"
            latent_space = encoder.get_latent_space(
                feature_space, svm=svm, purpose=purpose
            )

        self.feature_space = latent_space
        return latent_space

    def load_encoder(self, encoder, **kwargs):
        """Load an autoencoder in eval() mode

        Parameters
        ----------
        encoder : dict
            Dictionary with structure:

                >>> encoder = {'model': file.ml4c, 'params': file.params}

        data : obj
            data object
        svm : bool
            Whether or not these features are going to be used for kernel
            methods.

        Returns
        -------
        autoencoder.eval() : obj
            Autoencoder model object in eval mode to get the latent space.
        """

        params_path = encoder.get("params")
        model_path = encoder.get("model")

        model_params = json.load(open(params_path, "r"))
        model_params = model_params.get("model")
        name = model_params.pop("name")
        del model_params["type"]  # delete unneeded key, value

        input_dimension = model_params.pop("input_dimension")
        output_dimension = model_params.pop("output_dimension")

        autoencoder = dynamic_import(name, "ml4chem.models", alt_name="autoencoders")
        autoencoder = autoencoder(**model_params)
        autoencoder.prepare_model(input_dimension, output_dimension, **kwargs)
        autoencoder.load_state_dict(torch.load(model_path), strict=True)

        return autoencoder.eval()

    def to_pandas(self):
        """Convert features to pandas DataFrame"""
        return pd.DataFrame.from_dict(self.feature_space, orient="index")
