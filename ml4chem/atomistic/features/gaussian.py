import dask
import datetime
import logging
import os
import time
import torch
import dask.array as da
import numpy as np
import pandas as pd
from ase.data import atomic_numbers
from collections import OrderedDict
from ml4chem.atomistic.features.cutoff import Cosine
from ml4chem.atomistic.features.base import AtomisticFeatures
from ml4chem.data.serialization import dump, load
from ml4chem.data.preprocessing import Preprocessing
from ml4chem.utils import get_chunks, get_neighborlist, convert_elapsed_time

logger = logging.getLogger()


class Gaussian(AtomisticFeatures):
    """Behler-Parrinello symmetry functions

    This class builds local chemical environments for atoms based on the
    Behler-Parrinello Gaussian type symmetry functions. It is modular enough
    that can be used just for creating feature spaces.

    Parameters
    ----------
    cutoff : float
        Cutoff radius used for computing features.
    cutofffxn : object
        A Cutoff function object.
    normalized : bool
        Set it to true if the features are being normalized with respect to the
        cutoff radius.
    preprocessor : str
        Use some scaling method to preprocess the data. Default MinMaxScaler.
    custom : dict, opt
        Create custom symmetry functions, and override defaults. Default is
        None. The structure of the dictionary is as follows:

        >>> custom = {'G2': {'etas': etas},
                      'G3': {'etas': a_etas, 'zetas': zetas, 'gammas': gammas}}

    save_preprocessor : str
        Save preprocessor to file.
    scheduler : str
        The scheduler to be used with the dask backend.
    filename : str
        Path to save database. Note that if the filename exists, the features
        will be loaded without being recomputed.
    overwrite : bool
        If overwrite is set to True, ml4chem will not try to load existing
        databases. Default is True.
    angular_type : str
        Compute "G3" or "G4" angular symmetry functions.
    weighted : bool
        True if applying weighted feature of Gaussian function. See Ref. 2.
    batch_size : int
        Number of data points per batch to use for training. Default is None.

    References
    ----------
    1. Behler, J. Atom-centered symmetry functions for constructing
       high-dimensional neural network potentials. J. Chem. Phys. 134, 074106
       (2011).
    2. Gastegger, M., Schwiedrzik, L., Bittermann, M., Berzsenyi, F. &
       Marquetand, P. wACSF—Weighted atom-centered symmetry functions as
       descriptors in machine learning potentials. J. Chem. Phys. 148, 241709
       (2018).
    """

    NAME = "Gaussian"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(
        self,
        cutoff=6.5,
        cutofffxn=None,
        normalized=True,
        preprocessor=("MinMaxScaler", None),
        custom=None,
        save_preprocessor="ml4chem",
        scheduler="distributed",
        filename="features.db",
        overwrite=True,
        angular_type="G3",
        weighted=False,
        batch_size=None,
    ):
        super(Gaussian, self).__init__()

        cutoff_keys = ["radial", "angular"]
        if isinstance(cutoff, (int, float)):
            cutoff = {cutoff_key: cutoff for cutoff_key in cutoff_keys}

        self.normalized = normalized
        self.filename = filename
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.save_preprocessor = save_preprocessor
        self.overwrite = overwrite
        self.angular_type = angular_type.upper()
        self.weighted = weighted
        self.batch_size = batch_size

        # Let's add parameters that are going to be stored in the .params json
        # file.
        self.params = OrderedDict()
        self.params["name"] = self.name()

        # We verify that values of parameters are list otherwise they cannot be
        # serialized by json.

        # These keys are very likely to exist when doing inference
        keys = ["user_input", "GP"]

        if custom is None:
            custom = {key: custom for key in keys}
        elif (
            custom is not None and len(list(set(keys).intersection(custom.keys()))) == 0
        ):
            for value in custom.values():
                for k, v in value.items():
                    if isinstance(v, list) is False:
                        value[k] = v.tolist()
            custom = {"user_input": custom}

        self.custom = custom

        # This is a very general way of not forgetting to save variables
        _params = vars()

        # Delete useless variables
        delete = [
            "self",
            "scheduler",
            "overwrite",
            "k",
            "v",
            "value",
            "keys",
            "batch_size",
            "__class__",
            "cutoff_keys",
        ]

        for param in delete:
            try:
                del _params[param]
            except KeyError:
                # In case the variable does not exist we just pass.
                pass

        for k, v in _params.items():
            if v is not None:
                self.params[k] = v

        self.cutoff = cutoff
        self.cutofffxn = {}

        if cutofffxn is None:
            for cutoff_key in cutoff_keys:
                self.cutofffxn[cutoff_key] = Cosine(cutoff=self.cutoff[cutoff_key])
        else:
            raise RuntimeError("This case is not implemented yet...")

    def __call__(self, images, **kwargs):
        """Callable aspect of the Gaussian class

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
        return self.calculate(images=images, **kwargs)

    def calculate(self, images=None, purpose="training", data=None, svm=False, GP=None):
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

        self.forcetraining = data.forcetraining
        self.svm = svm

        try:
            client = dask.distributed.get_client()
        except (AttributeError, ValueError):
            # No dask operation
            client = None
        logger.info(" ")
        logger.info("Featurization")
        logger.info("=============")
        now = datetime.datetime.now()
        logger.info("Module accessed on {}.".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        logger.info(f"Module name: {self.name()}.")

        # FIXME the block below should become a function.
        if os.path.isfile(self.filename) and self.overwrite is False:
            logger.warning(f"Loading features from {self.filename}.")
            logger.info(" ")
            svm_keys = [b"feature_space", b"reference_space"]
            data = load(self.filename)

            data_hashes = list(data.keys())
            image_hashes = list(images.keys())

            if image_hashes == data_hashes:
                # Check if both lists are the same.
                self.dimension = len(data[list(data.keys())[0]][0][1])
                return data
            elif any(i in image_hashes for i in data_hashes):
                _data = {}
                for hash in image_hashes:
                    _data[hash] = data[hash]
                self.dimension = len(_data[list(_data.keys())[0]][0][1])
                return _data

            if svm_keys == list(data.keys()):
                feature_space = data[svm_keys[0]]
                reference_space = data[svm_keys[1]]
                return feature_space, reference_space

        initial_time = time.time()

        # Verify that we know the unique element symbols
        if data.unique_element_symbols is None:
            logger.info(f"Getting unique element symbols for {purpose}")

            unique_element_symbols = data.get_unique_element_symbols(
                images, purpose=purpose
            )

            unique_element_symbols = unique_element_symbols[purpose]

            logger.info(f"Unique chemical elements: {unique_element_symbols}")

        elif isinstance(data.unique_element_symbols, dict):
            unique_element_symbols = data.unique_element_symbols[purpose]

            logger.info(f"Unique chemical elements: {unique_element_symbols}")

        # we make the features
        if GP is None:
            self.GP = self.custom.get("GP", None)
        else:
            self.GP = GP

        if self.GP is None:
            custom = self.custom.get("user_input", None)
            self.GP = self.make_symmetry_functions(
                unique_element_symbols, custom=custom, angular_type=self.angular_type
            )
            self.custom.update({"GP": self.GP})
        else:
            logger.info("Using parameters from file to create symmetry functions...\n")

        self.print_features_params(self.GP)

        symbol = data.unique_element_symbols[purpose][0]
        sample = np.zeros(len(self.GP[symbol]))

        self.dimension = len(sample)

        preprocessor = Preprocessing(self.preprocessor, purpose=purpose)
        preprocessor.set(purpose=purpose, numpy=svm)

        # We start populating computations to get atomic features.
        logger.info("")
        logger.info("Embarrassingly parallel computation of atomic features...")

        stacked_features = []
        atoms_index_map = []  # This list is used to reconstruct images from atoms.

        if self.batch_size is None:
            self.batch_size = data.get_total_number_atoms()

        chunks = get_chunks(images, self.batch_size, svm=svm)

        ini = end = 0
        self.coordinates = []

        for chunk in chunks:
            images_ = OrderedDict(chunk)
            coordinates_ = OrderedDict()
            intermediate = []

            for image in images_.items():
                hash, image = image
                coordinates_[hash] = []
                end = ini + len(image)
                atoms_index_map.append(list(range(ini, end)))
                ini = end
                for atom in image:
                    index = atom.index
                    symbol = atom.symbol

                    cutoff_keys = ["radial", "angular"]
                    n_symbols, neighborpositions = {}, {}

                    if isinstance(self.cutoff, dict):
                        for cutoff_key in cutoff_keys:
                            nl = get_neighborlist(image, cutoff=self.cutoff[cutoff_key])
                            # n_indices: neighbor indices for central atom_i.
                            # n_offsets: neighbor offsets for central atom_i.
                            n_indices, n_offsets = nl[atom.index]

                            n_symbols_ = np.array(image.get_chemical_symbols())[
                                n_indices
                            ]
                            n_symbols[cutoff_key] = n_symbols_

                            neighborpositions_ = image.positions[n_indices] + np.dot(
                                n_offsets, image.get_cell()
                            )
                            neighborpositions[cutoff_key] = neighborpositions_
                    else:
                        for cutoff_key in cutoff_keys:
                            nl = get_neighborlist(image, cutoff=self.cutoff)
                            # n_indices: neighbor indices for central atom_i.
                            # n_offsets: neighbor offsets for central atom_i.
                            n_indices, n_offsets = nl[atom.index]

                            n_symbols_ = np.array(image.get_chemical_symbols())[
                                n_indices
                            ]
                            n_symbols[cutoff_key] = n_symbols_

                            neighborpositions_ = image.positions[n_indices] + np.dot(
                                n_offsets, image.get_cell()
                            )
                            neighborpositions[cutoff_key] = neighborpositions_

                    if client == None:
                        afp, Ri = self.get_atomic_features(
                            atom,
                            index,
                            symbol,
                            n_symbols,
                            neighborpositions,
                            image_molecule=image,
                            weighted=self.weighted,
                            n_indices=n_indices,
                        )
                    else:
                        afp, Ri = dask.delayed(self.get_atomic_features)(
                            atom,
                            index,
                            symbol,
                            n_symbols,
                            neighborpositions,
                            image_molecule=image,
                            weighted=self.weighted,
                            n_indices=n_indices,
                        )

                    coordinates_[hash].append((symbol, Ri))
                    intermediate.append(afp)

            if client == None:
                pass
            else:
                intermediate = client.persist(intermediate, scheduler=self.scheduler)
            stacked_features += intermediate
            self.coordinates.append(coordinates_)
            del intermediate

        scheduler_time = time.time() - initial_time

        if client != None:
            dask.distributed.wait(stacked_features)

        h, m, s = convert_elapsed_time(scheduler_time)
        logger.info(
            "... finished in {} hours {} minutes {:.2f}" " seconds.".format(h, m, s)
        )

        logger.info("")

        if self.preprocessor != None and svm:

            scaled_feature_space = []

            # To take advantage of dask_ml we need to convert our numpy array
            # into a dask array.
            logger.info("Converting features to dask array...")
            stacked_features = [
                da.from_delayed(lazy, dtype=float, shape=sample.shape)
                for lazy in stacked_features
            ]
            layout = {0: tuple(len(i) for i in atoms_index_map), 1: -1}
            # stacked_features = dask.array.stack(stacked_features, axis=0).rechunk(layout)
            stacked_features = da.stack(stacked_features, axis=0).rechunk(layout)

            logger.info(
                "Shape of array is {} and chunks {}.".format(
                    stacked_features.shape, stacked_features.chunks
                )
            )

            # Note that dask_ml by default convert the output of .fit
            # in a concrete value.
            if purpose == "training":
                stacked_features = preprocessor.fit(
                    stacked_features, scheduler=self.scheduler
                )
            else:
                stacked_features = preprocessor.transform(stacked_features)

            atoms_index_map = [client.scatter(indices) for indices in atoms_index_map]
            # stacked_features = [client.scatter(features) for features in stacked_features]
            stacked_features = client.scatter(stacked_features, broadcast=True)

            logger.info("Stacking features using atoms index map...")

            for indices in atoms_index_map:
                features = client.submit(
                    self.stack_features, *(indices, stacked_features)
                )

                # features = self.stack_features(indices, stacked_features)

                scaled_feature_space.append(features)

        elif self.preprocessor != None and svm == False:
            if purpose == "training":
                scaled_feature_space = preprocessor.fit(stacked_features)
            else:
                stacked_features = preprocessor.transform(stacked_features)

        else:
            # In this section, using the atom_index_map we gather the data to
            # reconstruct a list of lists with the right number of features.
            scaled_feature_space = []
            if client != None:
                atoms_index_map = [client.scatter(chunk) for chunk in atoms_index_map]
                stacked_features = client.scatter(stacked_features, broadcast=True)

            for indices in atoms_index_map:
                if client == None:
                    features = self.stack_features(indices, stacked_features)
                else:
                    features = client.submit(
                        self.stack_features, *(indices, stacked_features)
                    )
                scaled_feature_space.append(features)
            if client != None:
                scaled_feature_space = client.gather(scaled_feature_space)

        # Clean
        del stacked_features

        # Restack images
        feature_space = []

        if svm and purpose == "training":
            logger.info("Building array with reference space.")
            reference_space = []

            for i, image in enumerate(images.items()):
                if client == None:
                    restacked = self.restack_image(i, image, scaled_feature_space, svm)

                    # image = (hash, ase_image) -> tuple
                    for atom in image[1]:
                        restacked_atom = self.restack_atom(
                            i, atom, scaled_feature_space
                        )
                        reference_space.append(restacked_atom)

                    feature_space.append(restacked)
                else:
                    restacked = client.submit(
                        self.restack_image, *(i, image, scaled_feature_space, svm)
                    )

                    # image = (hash, ase_image) -> tuple
                    for atom in image[1]:
                        restacked_atom = client.submit(
                            self.restack_atom, *(i, atom, scaled_feature_space)
                        )
                        reference_space.append(restacked_atom)

                    feature_space.append(restacked)

            if client != None:
                reference_space = client.gather(reference_space)

        elif svm is False and purpose == "training":
            for i, image in enumerate(images.items()):
                if client == None:
                    restacked = self.restack_image(i, image, scaled_feature_space, svm)
                    feature_space.append(restacked)

                else:
                    restacked = client.submit(
                        self.restack_image, *(i, image, scaled_feature_space, svm)
                    )
                    feature_space.append(restacked)

        else:
            try:
                for i, image in enumerate(images.items()):
                    restacked = client.submit(
                        self.restack_image, *(i, image, scaled_feature_space, svm)
                    )
                    feature_space.append(restacked)

            except UnboundLocalError:
                # scaled_feature_space does not exist.
                for i, image in enumerate(images.items()):
                    restacked = client.submit(
                        self.restack_image, *(i, image, feature_space, svm)
                    )
                    feature_space.append(restacked)

        if client != None:
            feature_space = client.gather(feature_space)

        feature_space = OrderedDict(feature_space)

        fp_time = time.time() - initial_time

        h, m, s = convert_elapsed_time(fp_time)

        logger.info(
            "Featurization finished in {} hours {} minutes {:.2f}"
            " seconds.".format(h, m, s)
        )

        if svm and purpose == "training":
            if client != None:
                client.restart()  # Reclaims memory aggressively
            preprocessor.save_to_file(preprocessor, self.save_preprocessor)

            if self.filename is not None:
                logger.info(f"features saved to {self.filename}.")
                data = {"feature_space": feature_space}
                data.update({"reference_space": reference_space})
                dump(data, filename=self.filename)
                self.feature_space = feature_space
                self.reference_space = reference_space

            return self.feature_space, self.reference_space

        elif svm is False and purpose == "training":
            if client != None:
                client.restart()  # Reclaims memory aggressively
            preprocessor.save_to_file(preprocessor, self.save_preprocessor)

            if self.filename is not None:
                logger.info(f"features saved to {self.filename}.")
                dump(feature_space, filename=self.filename)
                self.feature_space = feature_space

            return self.feature_space
        else:
            self.feature_space = feature_space
            return self.feature_space

    def to_pandas(self):
        """Convert features to pandas DataFrame"""
        return pd.DataFrame.from_dict(self.feature_space, orient="index")

    def stack_features(self, indices, stacked_features):
        """Stack features """

        features = []
        for index in indices:
            features.append(stacked_features[index])

        return features

    def get_atomic_features(
        self,
        atom,
        index,
        symbol,
        n_symbols,
        neighborpositions,
        n_indices=None,
        image_molecule=None,
        weighted=False,
    ):
        """Delayed class method to compute atomic features

        Parameters
        ----------
        atom : object
            An ASE atom object.
        image : ase object, list
            List of atoms in an image.
        index : int
            Index of atom in atoms object.
        symbol : str
            Chemical symbol of atom in atoms object.
        n_symbols : ndarray of str
            Array of neighbors' symbols.
        neighborpositions : ndarray of float
            Array of Cartesian atomic positions.
        image_molecule : ase object, list
            List of atoms in an image.
        weighted : bool
            True if applying weighted feature of Gaussian function. See Ref.
            2.
        """
        cutoff_keys = ["radial", "angular"]
        num_symmetries = len(self.GP[symbol])
        # The central atom
        Ri = atom.position

        if self.forcetraining and isinstance(Ri, (np.ndarray, np.generic)) == False:
            Ri.requires_grad_(
                True
            )  # At this point the coordinates are a leaf and require gradients.

        features = [None] * num_symmetries

        # See https://listserv.brown.edu/cgi-bin/wa?A2=ind1904&L=AMP-USERS&P=19048
        # n_numbers = [atomic_numbers[symbol] for symbol in n_symbols]
        n_numbers = [
            atomic_numbers[item]
            for cutoff_key in cutoff_keys
            for item in n_symbols[cutoff_key]
        ]

        for count in range(num_symmetries):
            GP = self.GP[symbol][count]

            if GP["type"] == "G2":
                feature = calculate_G2(
                    n_numbers,
                    n_symbols["radial"],
                    neighborpositions["radial"],
                    GP["symbol"],
                    GP["eta"],
                    self.cutoff["radial"],
                    self.cutofffxn["radial"],
                    Ri,
                    image_molecule=image_molecule,
                    n_indices=n_indices,
                    normalized=self.normalized,
                    weighted=weighted,
                )
            elif GP["type"] == "G3":
                feature = calculate_G3(
                    n_numbers,
                    n_symbols["angular"],
                    neighborpositions["angular"],
                    GP["symbols"],
                    GP["gamma"],
                    GP["zeta"],
                    GP["eta"],
                    self.cutoff["angular"],
                    self.cutofffxn["angular"],
                    Ri,
                    normalized=self.normalized,
                    image_molecule=image_molecule,
                    n_indices=n_indices,
                    weighted=weighted,
                )
            elif GP["type"] == "G4":
                feature = calculate_G4(
                    n_numbers,
                    n_symbols,
                    neighborpositions,
                    GP["symbols"],
                    GP["gamma"],
                    GP["zeta"],
                    GP["eta"],
                    self.cutoff["angular"],
                    self.cutofffxn["angular"],
                    Ri,
                    normalized=self.normalized,
                    image_molecule=image_molecule,
                    n_indices=n_indices,
                    weighted=weighted,
                )
            else:
                raise NotImplementedError(
                    "The requested symmetry function is not implemented yet..."
                )
            features[count] = feature

        if self.svm:
            return np.array(features), Ri
        else:
            return torch.stack(features).float(), Ri

    def make_symmetry_functions(self, symbols, custom=None, angular_type="G3"):
        """Function to make symmetry functions

        This method needs at least unique symbols and defaults set to true.
        Parameters

        ----------
        symbols : list
            List of strings with chemical symbols to create symmetry functions.
            >>> symbols = ['H', 'O']

        custom : dict, opt
            Create custom symmetry functions, and override defaults. Default is
            None. The structure of the dictionary is as follows:

            >>> custom = {'G2': {'etas': etas},
                          'G3': {'etas': a_etas, 'zetas': zetas, 'gammas': gammas}}

        angular_type : str
            Compute "G3" or "G4" angular symmetry functions.

        Return
        ------
        GP : dict
            Symmetry function parameters.
        """

        GP = {}

        if custom is None:
            logger.warning("Making default symmetry functions...")

            for symbol in symbols:
                # Radial
                etas = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
                _GP = self.get_symmetry_functions(type="G2", etas=etas, symbols=symbols)

                # Angular
                etas = [0.005]
                zetas = [1.0, 4.0]
                gammas = [1.0, -1.0]
                _GP += self.get_symmetry_functions(
                    type=angular_type,
                    symbols=symbols,
                    etas=etas,
                    zetas=zetas,
                    gammas=gammas,
                )

                GP[symbol] = _GP
        else:
            logger.warning("Making custom symmetry functions...")
            types = sorted(custom.keys())

            for symbol in symbols:
                _GP = []
                for type_ in types:
                    if type_.upper() == "G2":
                        keys = ["Rs"]

                        kwargs = {}
                        for key in keys:
                            val = custom[type_].get(key, None)
                            if val is not None:
                                kwargs[key] = val

                        _GP += self.get_symmetry_functions(
                            type=type_,
                            etas=custom[type_]["etas"],
                            symbols=symbols,
                            **kwargs,
                        )
                    else:
                        keys = ["gammas", "Rs_a", "thetas"]

                        kwargs = {}
                        for key in keys:
                            val = custom[type_].get(key, None)
                            if val is not None:
                                kwargs[key] = val

                        _GP += self.get_symmetry_functions(
                            type=type_,
                            symbols=symbols,
                            etas=custom[type_]["etas"],
                            zetas=custom[type_]["zetas"],
                            **kwargs,
                        )
                GP[symbol] = _GP

        return GP

    def get_symmetry_functions(self, type, symbols, etas=None, zetas=None, gammas=None):
        """Get requested symmetry functions

        Parameters
        ----------
        type : str
            The desired symmetry function: 'G2', 'G3', or 'G4'.
        symbols : list
            List of chemical symbols.
        etas : list
            List of etas to build the Gaussian function.
        zetas : list
            List of zetas to build the Gaussian function.
        gammas : list
            List of gammas to build the Gaussian function.
        """

        supported_angular_symmetry_functions = ["G3", "G4"]

        if type == "G2":
            GP = [
                {"type": "G2", "symbol": symbol, "eta": eta}
                for eta in etas
                for symbol in symbols
            ]
            return GP

        elif type in supported_angular_symmetry_functions:
            GP = []
            for eta in etas:
                for zeta in zetas:
                    for gamma in gammas:
                        for idx1, sym1 in enumerate(symbols):
                            for sym2 in symbols[idx1:]:
                                pairs = sorted([sym1, sym2])
                                GP.append(
                                    {
                                        "type": type,
                                        "symbols": pairs,
                                        "eta": eta,
                                        "gamma": gamma,
                                        "zeta": zeta,
                                    }
                                )
            return GP
        else:
            raise RuntimeError(
                "The requested type of symmetry function is not supported."
            )

    def print_features_params(self, GP):
        """Print features parameters"""

        logger.info("Number of features per chemical element:")
        for symbol, v in GP.items():
            logger.info("    - {}: {}.".format(symbol, len(v)))

        _symbols = []
        for symbol, value in GP.items():
            logger.info(" ")
            logger.info(f"Symmetry function parameters for {symbol} atom:")
            underline = "---------------------------------------"

            for char in range(len(symbol)):
                underline += "-"

            logger.info(underline)
            logging.info(
                "{:^5} {:^12} {:4.4} {}".format("#", "Symbol", "Type", "Parameters")
            )

            if symbol not in _symbols:
                _symbols.append(symbol)
                for i, v in enumerate(value):
                    type_ = v["type"]
                    eta = v["eta"]
                    if type_ == "G2":
                        symbol = v["symbol"]
                        params = "{:^5} {:12.2} {:^4.4} eta: {:.4f}".format(
                            i, symbol, type_, eta
                        )
                    else:
                        symbol = str(v["symbols"])[1:-1].replace("'", "")
                        gamma = v["gamma"]
                        zeta = v["zeta"]
                        params = (
                            "{:^5} {:12} {:^4.5} eta: {:.4f} "
                            "gamma: {:7.4f} zeta: {:.4f}".format(
                                i, symbol, type_, eta, gamma, zeta
                            )
                        )

                    logging.info(params)


def calculate_G2(
    n_numbers,
    neighborsymbols,
    neighborpositions,
    center_symbol,
    eta,
    cutoff,
    cutofffxn,
    Ri,
    image_molecule=None,
    n_indices=None,
    normalized=True,
    weighted=False,
):
    """Calculate G2 symmetry function.

    These correspond to 2 body, or radial interactions.

    Parameters
    ----------
    n_symbols : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : list of str
        List of symbols of all neighbor atoms.
    neighborpositions : list of list of floats
        List of Cartesian atomic positions.
    center_symbol : str
        Chemical symbol of the center atom.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : float
        Cutoff radius.
    cutofffxn : object
        Cutoff function.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    normalized : bool
        Whether or not the symmetry function is normalized.
    image_molecule : ase object, list
        List of atoms in an image.
    n_indices : list
        List of indices of neighboring atoms from the image object.
    weighted : bool
        True if applying weighted feature of Gaussian function. See Ref. 2.


    Returns
    -------
    feature : float
        Radial feature.
    """
    feature = 0.0
    num_neighbors = len(neighborpositions)

    # Are we normalizing the feature?
    if normalized:
        Rc = cutoff
    else:
        Rc = 1.0

    Ris, Rjs, weights = [], [], []

    for count in range(num_neighbors):
        symbol = neighborsymbols[count]
        Rj = neighborpositions[count]

        if symbol == center_symbol:
            Ris.append(Ri)
            Rjs.append(Rj)

            if weighted:
                weights.append(image_molecule[n_indices[count]].number)

    if isinstance(Ri, np.ndarray):
        Ris = np.array(Ris)
        Rjs = np.array(Rjs)
        Rij = np.linalg.norm(Rjs - Ris, axis=1)
        feature = np.exp(-eta * (Rij ** 2.0) / (Rc ** 2.0)) * cutofffxn(Rij)
    else:
        Ris = torch.stack(Ris)
        Rjs = torch.stack(Rjs)
        Rij = torch.norm(Rjs - Ris, dim=1)
        feature = torch.exp(-eta * (Rij ** 2.0) / (Rc ** 2.0)) * cutofffxn(Rij)

    if weighted:
        feature *= weights

    return feature.sum()


def calculate_G3(
    n_numbers,
    neighborsymbols,
    neighborpositions,
    G_elements,
    gamma,
    zeta,
    eta,
    cutoff,
    cutofffxn,
    Ri,
    normalized=True,
    image_molecule=None,
    n_indices=None,
    weighted=False,
):
    """Calculate G3 symmetry function.

    These are 3 body or angular interactions.

    Parameters
    ----------
    n_symbols : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : list of str
        List of symbols of neighboring atoms.
    neighborpositions : list of list of floats
        List of Cartesian atomic positions of neighboring atoms.
    G_elements : list of str
        A list of two members, each member is the chemical species of one of
        the neighboring atoms forming the triangle with the center atom.
    gamma : float
        Parameter of Gaussian symmetry functions.
    zeta : float
        Parameter of Gaussian symmetry functions.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : float
        Cutoff radius.
    cutofffxn : object
        Cutoff function.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    normalized : bool
        Whether or not the symmetry function is normalized.
    image_molecule : ase object, list
        List of atoms in an image.
    n_indices : list
        List of indices of neighboring atoms from the image object.
    weighted : bool
        True if applying weighted feature of Gaussian function. See Ref. 2.

    Returns
    -------
    feature : float
        G3 feature value.
    """
    feature = 0.0
    counts = range(len(neighborpositions))

    # Are we normalizing the feature?
    if normalized:
        Rc = cutoff
    else:
        Rc = 1.0

    neighborpositions_j = []
    neighborpositions_k = []

    for j in counts:
        for k in counts[(j + 1) :]:
            els = sorted([neighborsymbols[j], neighborsymbols[k]])
            if els != G_elements:
                continue
            neighborpositions_j.append(neighborpositions[j])
            neighborpositions_k.append(neighborpositions[k])

    if isinstance(Ri, np.ndarray):
        neighborpositions_j = np.array(neighborpositions_j)
        Rij_vector = neighborpositions_j - Ri
        Rij = np.sqrt(np.einsum("ij,ij->i", Rij_vector, Rij_vector))

        neighborpositions_k = np.array(neighborpositions_k)
        Rik_vector = neighborpositions_k - Ri
        Rik = np.sqrt(np.einsum("ij,ij->i", Rik_vector, Rik_vector))

        Rjk_vector = neighborpositions_k - neighborpositions_j
        Rjk = np.sqrt(np.einsum("ij,ij->i", Rjk_vector, Rjk_vector))

        cos_theta_ijk = angles_row_wise(Rij_vector, Rik_vector, numpy=True)
        term = (1.0 + gamma * cos_theta_ijk) ** zeta
        term *= np.exp(-eta * (Rij ** 2.0 + Rik ** 2.0 + Rjk ** 2.0) / (Rc ** 2.0))
    else:
        neighborpositions_j = torch.stack(neighborpositions_j)
        Rij_vector = neighborpositions_j - Ri
        Rij = torch.sqrt(torch.einsum("ij,ij->i", Rij_vector, Rij_vector))

        neighborpositions_k = torch.stack(neighborpositions_k)
        Rik_vector = neighborpositions_k - Ri
        Rik = torch.sqrt(torch.einsum("ij,ij->i", Rik_vector, Rik_vector))

        Rjk_vector = neighborpositions_k - neighborpositions_j
        Rjk = torch.sqrt(torch.einsum("ij,ij->i", Rjk_vector, Rjk_vector))

        cos_theta_ijk = angles_row_wise(Rij_vector, Rik_vector, numpy=False)
        term = (1.0 + gamma * cos_theta_ijk) ** zeta
        term *= torch.exp(-eta * (Rij ** 2.0 + Rik ** 2.0 + Rjk ** 2.0) / (Rc ** 2.0))

    if weighted:
        term *= weighted_h(image_molecule, n_indices)

    term *= cutofffxn(Rij)
    term *= cutofffxn(Rik)
    term *= cutofffxn(Rjk)
    feature += term
    feature *= 2.0 ** (1.0 - zeta)
    return feature.sum()


def angles_row_wise(a, b, numpy):
    """Compute cosine angles row wise

    Parameters
    ----------
    a : tensor
        Tensor a. 
    b : tensor
        Tensor b.
    numpy : bool
        Are we operating a numpy or torch object?

    Returns
    -------
    angles
        The cosine angles row wise. 
    """
    if numpy:
        p1 = np.einsum("ij,ij->i", a, b)
        p2 = np.einsum("ij,ij->i", a, a)
        p3 = np.einsum("ij,ij->i", b, b)
        return p1 / np.sqrt(p2 * p3)
    else:
        p1 = torch.einsum("ij,ij->i", a, b)
        p2 = torch.einsum("ij,ij->i", a, a)
        p3 = torch.einsum("ij,ij->i", b, b)
        return p1 / torch.sqrt(p2 * p3)


def calculate_G4(
    n_numbers,
    neighborsymbols,
    neighborpositions,
    G_elements,
    gamma,
    zeta,
    eta,
    cutoff,
    cutofffxn,
    Ri,
    normalized=True,
    image_molecule=None,
    n_indices=None,
    weighted=False,
):
    """Calculate G4 symmetry function.

    These are 3 body or angular interactions.

    Parameters
    ----------
    n_symbols : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : list of str
        List of symbols of neighboring atoms.
    neighborpositions : list of list of floats
        List of Cartesian atomic positions of neighboring atoms.
    G_elements : list of str
        A list of two members, each member is the chemical species of one of
        the neighboring atoms forming the triangle with the center atom.
    gamma : float
        Parameter of Gaussian symmetry functions.
    zeta : float
        Parameter of Gaussian symmetry functions.
    eta : float
        Parameter of Gaussian symmetry functions.
    cutoff : float
        Cutoff radius.
    cutofffxn : object
        Cutoff function.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    normalized : bool
        Whether or not the symmetry function is normalized.
    image_molecule : ase object, list
        List of atoms in an image.
    n_indices : list
        List of indices of neighboring atoms from the image object.
    weighted : bool
        True if applying weighted feature of Gaussian function. See Ref. 2.

    Returns
    -------
    feature : float
        G4 feature value.

    Notes
    -----
    The difference between the calculate_G3 and the calculate_G4 function is
    that calculate_G4 accounts for bond angles of 180 degrees.
    """
    feature = 0.0
    counts = range(len(neighborpositions))
    # Are we normalizing the feature?
    if normalized:
        Rc = cutoff
    else:
        Rc = 1.0
    for j in counts:
        for k in counts[(j + 1) :]:
            els = sorted([neighborsymbols[j], neighborsymbols[k]])
            if els != G_elements:
                continue

            Rij_vector = neighborpositions[j] - Ri
            Rij = np.linalg.norm(Rij_vector)
            Rik_vector = neighborpositions[k] - Ri
            Rik = np.linalg.norm(Rik_vector)
            cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
            term = (1.0 + gamma * cos_theta_ijk) ** zeta
            term *= np.exp(-eta * (Rij ** 2.0 + Rik ** 2.0) / (Rc ** 2.0))

            if weighted:
                term *= weighted_h(image_molecule, n_indices)

            term *= cutofffxn(Rij)
            term *= cutofffxn(Rik)
            feature += term
    feature *= 2.0 ** (1.0 - zeta)
    return feature


def weighted_h(image_molecule, n_indices):
    """ Calculate the atomic numbers of neighboring atoms for a molecule,
    then multiplies each neighor atomic number by each other.

    Parameters
    ----------
    image_molecule : ase object, list
        List of atoms in an image.
    n_indices : list
        List of indices of neighboring atoms from the image object.
    """
    atomic_numbers = 1.0

    for i in n_indices:
        atomic_numbers *= image_molecule[i].number

    return atomic_numbers
