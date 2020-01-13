import dask
import datetime
import logging
import os
import time
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from dscribe.descriptors import CoulombMatrix as CoulombMatrixDscribe
from ml4chem.data.preprocessing import Preprocessing
from ml4chem.data.serialization import dump, load
from ml4chem.features.base import AtomisticFeatures
from ml4chem.utils import get_chunks, convert_elapsed_time

logger = logging.getLogger()


class CoulombMatrix(AtomisticFeatures, CoulombMatrixDscribe):
    """Coulomb Matrix features

    
    Parameters
    ----------
    filename : str
        Path to save database. Note that if the filename exists, the features
        will be loaded without being recomputed.
    preprocessor : str
        Use some scaling method to preprocess the data. Default None.
    batch_size : int
        Number of data points per batch to use for training. Default is None.
    scheduler : str
        The scheduler to be used with the dask backend.

    Notes
    -----
    This class computes Coulomb matrix features using the dscribe module. As
    mentioned in ML4Chem's paper, we avoid duplication of efforts and this
    module serves as a demonstration.
    """

    NAME = "CoulombMatrix"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(self, preprocessor=None, batch_size=None, filename="features.db", scheduler="distributed", **kwargs):
        super(CoulombMatrix, self).__init__()

        CoulombMatrixDscribe.__init__(self, permutation='none', flatten=False, **kwargs)

        self.batch_size = batch_size
        self.filename = filename
        self.preprocessor = preprocessor
        self.scheduler = scheduler
    
    def calculate(self, images=None, purpose="training", data=None, svm=False):
        """Calculate the features per atom in an atoms objects

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
        reference_space : dict
            A reference space useful for SVM models.
        """

        client = dask.distributed.get_client()
        logger.info(" ")
        logger.info("Featurization")
        logger.info("=============")
        now = datetime.datetime.now()
        logger.info("Module accessed on {}.".format(now.strftime("%Y-%m-%d %H:%M:%S")))

        # FIXME the block below should become a function.
        if os.path.isfile(self.filename) and self.overwrite is False:
            logger.warning("Loading features from {}.".format(self.filename))
            logger.info(" ")
            svm_keys = [b"feature_space", b"reference_space"]
            data = load(self.filename)

            data_hashes = list(data.keys())
            image_hashes = list(images.keys())

            if image_hashes == data_hashes:
                # Check if both lists are the same.
                return data
            elif any(i in image_hashes for i in data_hashes):
                # Check if any of the elem
                _data = {}
                for hash in image_hashes:
                    _data[hash] = data[hash]
                return _data

            if svm_keys == list(data.keys()):
                feature_space = data[svm_keys[0]]
                reference_space = data[svm_keys[1]]
                return feature_space, reference_space

        initial_time = time.time()

        # Verify that we know the unique element symbols
        if data.unique_element_symbols is None:
            logger.info("Getting unique element symbols for {}".format(purpose))

            unique_element_symbols = data.get_unique_element_symbols(
                images, purpose=purpose
            )

            unique_element_symbols = unique_element_symbols[purpose]

            logger.info("Unique chemical elements: {}".format(unique_element_symbols))

        elif isinstance(data.unique_element_symbols, dict):
            unique_element_symbols = data.unique_element_symbols[purpose]

            logger.info("Unique chemical elements: {}".format(unique_element_symbols))

        # we make the features
        if self.preprocessor is not None:
            preprocessor = Preprocessing(self.preprocessor, purpose=purpose)
            preprocessor.set(purpose=purpose)

        # We start populating computations to get atomic features.
        logger.info("")
        logger.info("Embarrassingly parallel computation of atomic features...")

        stacked_features = []
        atoms_index_map = []  # This list is used to reconstruct images from atoms.

        if self.batch_size is None:
            self.batch_size = data.get_total_number_atoms()

        chunks = get_chunks(images, self.batch_size, svm=svm)

        ini = end = 0
        for chunk in chunks:
            images_ = OrderedDict(chunk)
            intermediate = []

            for image in images_.items():
                key, image = image
                end = ini + len(image)
                atoms_index_map.append(list(range(ini, end)))
                ini = end
                _features = dask.delayed(self.create)(image)
                intermediate.append(_features)

            intermediate = client.persist(intermediate, scheduler=self.scheduler)
            stacked_features += intermediate
            del intermediate

        scheduler_time = time.time() - initial_time

        dask.distributed.wait(stacked_features)

        h, m, s = convert_elapsed_time(scheduler_time)
        logger.info(
            "... finished in {} hours {} minutes {:.2f}" " seconds.".format(h, m, s)
        )

        logger.info("")

        if self.preprocessor is not None:
            logger.info("Converting features to dask array...")
            symbol = data.unique_element_symbols[purpose][0]
            sample = np.zeros(len(self.GP[symbol]))
            # dim = (len(stacked_features), len(sample))

            stacked_features = [
                dask.array.from_delayed(lazy, dtype=float, shape=sample.shape)
                for lazy in stacked_features
            ]

            layout = {0: tuple(len(i) for i in atoms_index_map), 1: -1}
            # stacked_features = dask.array.stack(stacked_features, axis=0).rechunk(layout)
            stacked_features = dask.array.stack(stacked_features, axis=0).rechunk(
                layout
            )

            logger.info(
                "Shape of array is {} and chunks {}.".format(
                    stacked_features.shape, stacked_features.chunks
                )
            )

            # To take advantage of dask_ml we need to convert our numpy array
            # into a dask array.

            scaled_feature_space = []

            # Note that dask_ml by default convert the output of .fit
            # in a concrete value.
            if purpose == "training":
                stacked_features = preprocessor.fit(
                    stacked_features, scheduler=self.scheduler
                )
            else:
                stacked_features = preprocessor.transform(stacked_features)

            atoms_index_map = [client.scatter(indices) for indices in atoms_index_map]
            stacked_features = client.scatter(stacked_features, broadcast=True)

            logger.info("Stacking features using atoms index map...")

            for indices in atoms_index_map:
                features = client.submit(
                    self.stack_features, *(indices, stacked_features)
                )

                # features = self.stack_features(indices, stacked_features)

                scaled_feature_space.append(features)

            # scaled_feature_space = client.gather(scaled_feature_space)

        else:
            feature_space = []
            atoms_index_map = [client.scatter(chunk) for chunk in atoms_index_map]

            for indices in atoms_index_map:
                features = client.submit(
                    self.stack_features, *(indices, stacked_features)
                )
                feature_space.append(features)
        # Clean
        del stacked_features

        # Restack images
        feature_space = []

        if svm and purpose == "training":
            logger.info("Building array with reference space.")
            reference_space = []

            for i, image in enumerate(images.items()):
                restacked = client.submit(
                    self.restack_image, *(i, image, None, scaled_feature_space, svm)
                )
                feature_space.append(restacked)

                # image = (hash, ase_image) -> tuple
                for atom in image[1]:
                    reference_space.append(
                        self.restack_atom(i, atom, scaled_feature_space)
                    )

            reference_space = dask.compute(*reference_space, scheduler=self.scheduler)
        else:
            try:
                for i, image in enumerate(images.items()):
                    restacked = client.submit(
                        self.restack_image, *(i, image, None, scaled_feature_space, svm)
                    )
                    feature_space.append(restacked)

            except UnboundLocalError:
                # scaled_feature_space does not exist.
                for i, image in enumerate(images.items()):
                    restacked = client.submit(
                        self.restack_image, *(i, image, feature_space, None, svm)
                    )
                    feature_space.append(restacked)

        feature_space = client.gather(feature_space)
        feature_space = OrderedDict(feature_space)

        fp_time = time.time() - initial_time

        h, m, s = convert_elapsed_time(fp_time)

        logger.info(
            "Featurization finished in {} hours {} minutes {:.2f}"
            " seconds.".format(h, m, s)
        )

        if svm and purpose == "training":
            client.restart()  # Reclaims memory aggressively
            preprocessor.save_to_file(preprocessor, self.save_preprocessor)

            if self.filename is not None:
                logger.info("features saved to {}.".format(self.filename))
                data = {"feature_space": feature_space}
                data.update({"reference_space": reference_space})
                dump(data, filename=self.filename)
                self.feature_space = feature_space
                self.reference_space = reference_space

            return self.feature_space, self.reference_space

        elif svm is False and purpose == "training":
            client.restart()  # Reclaims memory aggressively
            preprocessor.save_to_file(preprocessor, self.save_preprocessor)

            if self.filename is not None:
                logger.info("features saved to {}.".format(self.filename))
                dump(feature_space, filename=self.filename)
                self.feature_space = feature_space

            return self.feature_space
        else:
            self.feature_space = feature_space
            return self.feature_space

    def to_pandas(self):
        """Convert features to pandas DataFrame"""
        return pd.DataFrame.from_dict(self.feature_space, orient="index")