import dask
import datetime
import logging
import os
import time
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from ml4chem.atomistic.features.base import AtomisticFeatures
from ml4chem.data.preprocessing import Preprocessing
from ml4chem.data.serialization import dump, load
from ml4chem.utils import convert_elapsed_time

# Starting logger object
logger = logging.getLogger()


class Cartesian(AtomisticFeatures):
    """Cartesian Coordinates

    Cartesian coordinates are features, too (not very useful ones though). This
    class takes images in ASE format and return them hashed to be used by
    ML4Chem.


    Parameters
    ----------
    scheduler : str
        The scheduler to be used with the dask backend.
    filename : str
        Name to save on disk of serialized database.
    preprocessor : tuple
        Use some scaling method to preprocess the data. Default Normalizer.
    save_preprocessor : str
        Save preprocessor to file.
    overwrite : bool
        If overwrite is set to True, ml4chem will not try to load existing
        databases. Default is True.
    """

    NAME = "Cartesian"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(
        self,
        scheduler="distributed",
        filename="cartesians.db",
        preprocessor=("Normalizer", None),
        save_preprocessor="ml4chem",
        overwrite=True,
    ):
        super(Cartesian, self).__init__()

        self.filename = filename
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.save_preprocessor = save_preprocessor
        self.overwrite = overwrite

    def calculate(self, images=None, purpose="training", data=None, svm=False):
        """Return features per atom in an atoms objects

        Parameters
        ----------
        image : dict
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

        logger.info(" ")
        logger.info("Featurization")
        logger.info("=============")
        now = datetime.datetime.now()
        logger.info("Module accessed on {}.".format(now.strftime("%Y-%m-%d %H:%M:%S")))

        if os.path.isfile(self.filename) and self.overwrite is False:
            logger.warning("Loading features from {}.".format(self.filename))
            logger.info(" ")
            svm_keys = [b"feature_space", b"reference_space"]
            data = load(self.filename)

            if svm_keys == list(data.keys()):
                feature_space = data[svm_keys[0]]
                reference_space = data[svm_keys[1]]
                return feature_space, reference_space
            else:
                return data

        initial_time = time.time()

        # Verify that we know the unique element symbols
        if data.unique_element_symbols is None:
            logger.info("Getting unique element symbols for {}".format(purpose))

            unique_element_symbols = data.get_unique_element_symbols(
                images, purpose=purpose
            )

            unique_element_symbols = unique_element_symbols[purpose]

            logger.info("Unique chemical elements: {}".format(unique_element_symbols))

        preprocessor = Preprocessing(self.preprocessor, purpose=purpose)
        preprocessor.set(purpose=purpose)

        # We start populating computations with delayed functions to operate
        # with dask's scheduler. These computations get cartesian coordinates.
        computations = []

        for image in images.items():
            key, image = image

            feature_vectors = []

            computations.append(feature_vectors)

            for atom in image:
                if self.preprocessor is not None:
                    # In this case we will preprocess data and need numpy
                    # arrays to operate with sklearn.
                    afp = self.get_atomic_features(atom, svm=True)
                    feature_vectors.append(afp[1])
                else:
                    afp = self.get_atomic_features(atom, svm=svm)
                    feature_vectors.append(afp)

        # In this block we compute the delayed functions in computations.
        feature_space = dask.compute(*computations, scheduler=self.scheduler)

        hashes = list(images.keys())

        if self.preprocessor is not None and purpose == "training":
            feature_space = np.array(feature_space)
            dim = feature_space.shape

            if len(dim) > 1:
                d1, d2, d3 = dim
                feature_space = feature_space.reshape(d1 * d2, d3)
                feature_space = preprocessor.fit(
                    feature_space, scheduler=self.scheduler
                )
                feature_space = feature_space.reshape(d1, d2, d3)
            else:
                atoms_index_map = []
                stack = []

                d1 = ini = end = 0

                for i in feature_space:
                    end = ini + len(i)
                    atoms_map = list(range(ini, end))
                    atoms_index_map.append(atoms_map)
                    ini = end

                    for j in i:
                        stack.append(j)
                        d1 += 1

                feature_space = np.array(stack)

                d2 = len(stack[0])
                del stack

            # More data processing depending on the method used.
            computations = []

            if svm:
                reference_space = []

                for i, image in enumerate(images.items()):
                    computations.append(
                        self.restack_image(i, image, feature_space, svm=svm)
                    )

                    # image = (hash, ase_image) -> tuple
                    for atom in image[1]:
                        reference_space.append(
                            self.restack_atom(i, atom, feature_space)
                        )

                reference_space = dask.compute(
                    *reference_space, scheduler=self.scheduler
                )
            else:
                for i, image in enumerate(images.items()):
                    computations.append(
                        self.restack_image(i, image, feature_space, svm=svm)
                    )

            feature_space = dask.compute(*computations, scheduler=self.scheduler)

            feature_space = OrderedDict(feature_space)

            # Save preprocessor.
            preprocessor.save_to_file(preprocessor, self.save_preprocessor)

        elif self.preprocessor is not None and purpose == "inference":
            # We take stacked features and preprocess them
            stacked_features = np.array(feature_space)
            d1, d2, d3 = stacked_features.shape
            stacked_features = stacked_features.reshape(d1 * d2, d3)
            feature_space = OrderedDict()
            scaled_feature_space = preprocessor.transform(stacked_features)

            # Once preprocessed, they are wrapped as a dictionary.
            # TODO this has to be parallelized.
            for key, image in images.items():
                if key not in feature_space.keys():
                    feature_space[key] = []
                for index, atom in enumerate(image):
                    symbol = atom.symbol

                    if svm:
                        scaled = scaled_feature_space[index]
                        # TODO change this to something more elegant later
                        try:
                            self.reference_space
                        except AttributeError:
                            # If self.reference does not exist it means that
                            # reference_space is being loaded by Messagepack.
                            symbol = symbol.encode("utf-8")
                    else:
                        scaled = torch.tensor(
                            scaled_feature_space[index],
                            requires_grad=False,
                            dtype=torch.float,
                        )

                    feature_space[key].append((symbol, scaled))
        else:
            feature_space = OrderedDict(zip(hashes, feature_space))

        fp_time = time.time() - initial_time

        h, m, s = convert_elapsed_time(fp_time)

        logger.info(
            "Featurization finished in {} hours {} minutes {:.2f} "
            "seconds.\n".format(h, m, s)
        )

        if svm:
            data = {"feature_space": feature_space}
            dump(data, filename=self.filename)
        else:
            dump(feature_space, filename=self.filename)

        self.feature_space = feature_space
        return self.feature_space

    def to_pandas(self):
        """Convert features to pandas DataFrame"""
        return pd.DataFrame.from_dict(self.feature_space, orient="index")

    @dask.delayed
    def get_atomic_features(self, atom, svm=False):
        """Delayed class method to get atomic features


        Parameters
        ----------
        atom : object
            An ASE atom object.
        svm : bool
            Is this SVM?
        """

        symbol = atom.symbol
        position = atom.position

        if svm is False:
            position = torch.tensor(position, requires_grad=False, dtype=torch.float)

        return symbol, position

    @dask.delayed
    def restack_image(self, index, image, scaled_feature_space, svm=False):
        """Restack images to correct dictionary's structure to train

        Parameters
        ----------
        index : int
            Index of original hashed image.
        image : obj
            An ASE image object.
        scaled_feature_space : np.array
            A numpy array with the scaled features

        Returns
        -------
        key, features : tuple
            The hashed key image and its corresponding features.
        """
        key, image = image
        features = []
        for j, atom in enumerate(image):
            symbol = atom.symbol
            if svm:
                scaled = scaled_feature_space[index][j]
            else:
                scaled = torch.tensor(
                    scaled_feature_space[index][j],
                    requires_grad=False,
                    dtype=torch.float,
                )
            features.append((symbol, scaled))

        return key, features

    @dask.delayed
    def restack_atom(self, image_index, atom, scaled_feature_space):
        """Restack atoms to a raveled list to use with SVM

        Parameters
        ----------
        image_index : int
            Index of original hashed image.
        atom : object
            An atom object.
        scaled_feature_space : np.array
            A numpy array with the scaled features

        Returns
        -------
        symbol, features : tuple
            The hashed key image and its corresponding features.
        """

        symbol = atom.symbol
        features = scaled_feature_space[image_index][atom.index]

        return symbol, features
