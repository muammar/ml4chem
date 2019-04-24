import dask
import logging
import time
import torch
import numpy as np
from ase.data import atomic_numbers
from collections import OrderedDict
from .cutoff import Cosine
from ml4chem.data.serialization import dump
from ml4chem.data.preprocessing import Preprocessing
from ml4chem.utils import get_neighborlist, convert_elapsed_time

logger = logging.getLogger()


class Gaussian(object):
    """Behler-Parrinello symmetry functions

    This class builds local chemical environments for atoms based on the
    Behler-Parrinello Gaussian type symmetry functions. It is modular enough
    that can be used just for creating feature spaces.


    Parameters
    ----------
    cutoff : float
        Cutoff radius used for computing fingerprints.
    cutofffxn : object
        A Cutoff function object.
    normalized : bool
        Set it to true if the features are being normalized with respect to the
        cutoff radius.
    preprocessor : str
        Use some scaling method to preprocess the data. Default MinMaxScaler.
    defaults : bool
        Are we creating default symmetry functions?
    save_preprocessor : str
        Save preprocessor to file.
    scheduler : str
        The scheduler to be used with the dask backend.
    """
    NAME = 'Gaussian'

    @classmethod
    def name(cls):
        """Returns name of class"""

        return cls.NAME

    def __init__(self, cutoff=6.5, cutofffxn=None, normalized=True,
                 preprocessor=('MinMaxScaler', None), defaults=True,
                 save_preprocessor='ml4chem', scheduler='distributed',
                 filename='fingerprints.db'):

        self.cutoff = cutoff
        self.normalized = normalized
        self.filename = filename
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.save_preprocessor = save_preprocessor

        # Let's add parameters that are going to be stored in the .params json
        # file.
        self.params = OrderedDict()
        self.params['name'] = self.name()

        # This is a very general way of not forgetting to save variables
        _params = vars()

        # Delete useless variables
        del _params['self']
        del _params['scheduler']

        for k, v in _params.items():
            if v is not None:
                self.params[k] = v

        self.defaults = defaults

        if cutofffxn is None:
            self.cutofffxn = Cosine(cutoff=cutoff)
        else:
            self.cutofffxn = cutofffxn

    def calculate_features(self, images, purpose='training', data=None,
                           svm=False):
        """Calculate the features per atom in an atoms objects

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

        logger.info(' ')
        logger.info('Fingerprinting')
        logger.info('==============')

        initial_time = time.time()

        # Verify that we know the unique element symbols
        if data.unique_element_symbols is None:
            logger.info('Getting unique element symbols for {}'
                        .format(purpose))

            unique_element_symbols = \
                data.get_unique_element_symbols(images, purpose=purpose)

            unique_element_symbols = unique_element_symbols[purpose]

            logger.info('Unique chemical elements: {}'
                        .format(unique_element_symbols))

        # If self.defaults is True we create default symmetry functions.
        if self.defaults:
            self.GP = self.make_symmetry_functions(unique_element_symbols,
                                                   defaults=self.defaults)

        self.print_fingerprint_params(self.GP)

        preprocessor = Preprocessing(self.preprocessor, purpose=purpose)
        preprocessor.set(purpose=purpose)

        # We start populating computations to get atomic fingerprints.
        logger.info('')
        logger.info('Adding atomic fingerprint calculations to scheduler...')

        computations = []
        for image in images.items():
            key, image = image
            feature_vectors = []
            computations.append(feature_vectors)

            for atom in image:
                index = atom.index
                symbol = atom.symbol
                nl = get_neighborlist(image, cutoff=self.cutoff)
                n_indices, n_offsets = nl[atom.index]

                n_symbols = np.array(image.get_chemical_symbols())[n_indices]
                neighborpositions = image.positions[n_indices] + \
                    np.dot(n_offsets, image.get_cell())

                afp = self.get_atomic_fingerprint(atom, index,
                                                  symbol, n_symbols,
                                                  neighborpositions,
                                                  self.preprocessor)
                feature_vectors.append(afp)

        scheduler_time = time.time() - initial_time

        h, m, s = convert_elapsed_time(scheduler_time)
        logger.info('... finished in {} hours {} minutes {:.2f}'
                    ' seconds.' .format(h, m, s))

        # In this block we compute the fingerprints.
        logger.info('')
        logger.info('Computing fingerprints...')
        if self.preprocessor is None:
            feature_space = dask.compute(*computations,
                                         scheduler=self.scheduler)
        else:
            stacked_features = dask.compute(*computations,
                                            scheduler=self.scheduler)

            stacked_features = np.array(stacked_features)

            dim = stacked_features.shape

            if len(dim) > 1:
                d1, d2, d3 = dim
                stacked_features = stacked_features.reshape(d1 * d2, d3)
            else:
                atoms_index_map = []
                stack = []

                d1 = ini = end = 0

                for i in stacked_features:
                    end = ini + len(i)
                    atoms_index_map.append(list(range(ini, end)))
                    ini = end

                    for j in i:
                        stack.append(j)
                        d1 += 1

                stacked_features = np.array(stack)

                d2 = len(stack[0])
                del stack

        if purpose == 'training':
            # To take advantage of dask_ml we need to convert our numpy array
            # into a dask array.
            if self.preprocessor is not None:
                if len(dim) > 1:
                    stacked_features = dask.array.from_array(stacked_features,
                                                             chunks=(d1 * d2,
                                                                     d3))
                    stacked_features = preprocessor.fit(stacked_features,
                                                        scheduler=self.scheduler)

                    scaled_feature_space = stacked_features.reshape(d1, d2, d3)
                else:
                    stacked_features = dask.array.from_array(stacked_features,
                                                             chunks=(d1, d2))

                    stacked_features = preprocessor.fit(stacked_features,
                                                        scheduler=self.scheduler)
                    scaled_feature_space = []

                    client = dask.distributed.get_client()
                    atoms_index_map = [client.scatter(chunk) for chunk in
                                       atoms_index_map]

                    self.stacked_features = client.scatter(stacked_features)

                    for indices in atoms_index_map:
                        features = client.submit(self.stack_features,
                                                 *(indices, stacked_features))
                        scaled_feature_space.append(features)

            # More data processing depending on the method used.
            computations = []

            if svm:
                reference_space = []

                for i, image in enumerate(images.items()):
                    computations.append(self.restack_image(
                        i, image, scaled_feature_space=scaled_feature_space,
                        svm=svm))

                    # image = (hash, ase_image) -> tuple
                    for atom in image[1]:
                        reference_space.append(self.restack_atom(
                            i, atom, scaled_feature_space))

                reference_space = dask.compute(*reference_space,
                                               scheduler=self.scheduler)
            else:
                try:
                    for i, image in enumerate(images.items()):
                        computations.append(self.restack_image(i, image,
                            scaled_feature_space=scaled_feature_space,
                            svm=svm))

                except UnboundLocalError:
                # scaled_feature_space does not exist.
                    for i, image in enumerate(images.items()):
                        computations.append(self.restack_image(i, image,
                            feature_space=feature_space,
                            svm=svm))


            feature_space = dask.compute(*computations,
                                         scheduler=self.scheduler)
            feature_space = OrderedDict(feature_space)

            preprocessor.save_to_file(preprocessor, self.save_preprocessor)

            fp_time = time.time() - initial_time

            h, m, s = convert_elapsed_time(fp_time)
            logger.info('Fingerprinting finished in {} hours {} minutes {:.2f}'
                        ' seconds.' .format(h, m, s))

            data = {'feature_space': feature_space}

            if svm:
                data.update({'reference_space': reference_space})
                dump(data, filename=self.filename)
                return feature_space, reference_space
            else:
                return feature_space

        elif purpose == 'inference':
            feature_space = OrderedDict()
            scaled_feature_space = preprocessor.transform(stacked_features)
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
                            symbol = symbol.encode('utf-8')
                    else:
                        scaled = torch.tensor(scaled_feature_space[index],
                                              requires_grad=True,
                                              dtype=torch.float)

                    feature_space[key].append((symbol, scaled))

            fp_time = time.time() - initial_time

            h, m, s = convert_elapsed_time(fp_time)

            logger.info('Fingerprinting finished in {} hours {} minutes {:.2f}'
                        ' seconds.' .format(h, m, s))

            return feature_space

    def stack_features(self, indices, stacked_features):
        """Stack features """

        features = []
        for index in indices:
            features.append(stacked_features[index])

        return features

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

    @dask.delayed
    def restack_image(self, index, image, feature_space=None,
                      scaled_feature_space=None, svm=False):
        """Restack images to correct dictionary's structure to train

        Parameters
        ----------
        index : int
            Index of original hashed image.
        image : obj
            An ASE image object.
        feature_space : np.array
            A numpy array with raw features.
        scaled_feature_space : np.array
            A numpy array with scaled features.

        Returns
        -------
        hash, features : tuple
            Hash of image and its corresponding features.
        """
        hash, image = image

        if scaled_feature_space is not None:
            features = []
            for j, atom in enumerate(image):
                symbol = atom.symbol
                if svm:
                    scaled = scaled_feature_space[index][j]
                else:
                    scaled = torch.tensor(scaled_feature_space[index][j],
                                          requires_grad=True,
                                          dtype=torch.float)
                features.append((symbol, scaled))

            return hash, features

        elif feature_space is not None:
            features = feature_space[index]

            return hash, features


    @dask.delayed
    def fingerprints_per_image(self, image):
        """A delayed function to parallelize fingerprints per image

        Parameters
        ----------
        image : obj
            An ASE image object.

        Notes
        -----
            This function is not being currently used.
        """

        key, image = image
        image_positions = image.positions

        feature_space = []

        for atom in image:
            index = atom.index
            symbol = atom.symbol
            nl = get_neighborlist(image, cutoff=self.cutoff)
            n_indices, n_offsets = nl[atom.index]

            n_symbols = [image[i].symbol for i in n_indices]
            neighborpositions = [image_positions[neighbor] +
                                 np.dot(offset, image.cell)
                                 for (neighbor, offset) in
                                 zip(n_indices, n_offsets)]

            feature_vector = self.get_atomic_fingerprint(atom, index,
                                                         symbol, n_symbols,
                                                         neighborpositions,
                                                         self.preprocessor)

            if self.preprocessor is not None:
                feature_space.append(feature_vector[1])
            else:
                feature_space.append(feature_vector)

        if self.preprocessor is not None:
            return feature_space
        else:
            return key, feature_space

    @dask.delayed
    def get_atomic_fingerprint(self, atom, index, symbol, n_symbols,
                               neighborpositions, preprocessor):
        """Delayed class method to compute atomic fingerprints


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
        preprocessor : str
            Feature preprocessor.
        n_symbols : ndarray of str
            Array of neighbors' symbols.
        neighborpositions : ndarray of float
            Array of Cartesian atomic positions.
        """

        num_symmetries = len(self.GP[symbol])
        Ri = atom.position
        fingerprint = [None] * num_symmetries

        n_numbers = [atomic_numbers[symbol] for symbol in n_symbols]

        for count in range(num_symmetries):
            GP = self.GP[symbol][count]

            if GP['type'] == 'G2':
                feature = calculate_G2(n_numbers, n_symbols, neighborpositions,
                                       GP['symbol'], GP['eta'],
                                       self.cutoff, self.cutofffxn, Ri,
                                       normalized=self.normalized)
            elif GP['type'] == 'G3':
                feature = calculate_G3(n_numbers, n_symbols, neighborpositions,
                                       GP['symbols'], GP['gamma'],
                                       GP['zeta'], GP['eta'], self.cutoff,
                                       self.cutofffxn, Ri)
            elif GP['type'] == 'G4':
                feature = calculate_G4(n_numbers, n_symbols, neighborpositions,
                                       GP['symbols'], GP['gamma'],
                                       GP['zeta'], GP['eta'], self.cutoff,
                                       self.cutofffxn, Ri)
            else:
                logger.error('not implemented')
            fingerprint[count] = feature

        if preprocessor is None:
            fingerprint = torch.tensor(fingerprint, requires_grad=True,
                                       dtype=torch.float)

        if preprocessor is None:
            return symbol, fingerprint
        else:
            return fingerprint

    def make_symmetry_functions(self, symbols, defaults=True, type=None,
                                etas=None, zetas=None, gammas=None):
        """Function to make symmetry functions

        This method needs at least unique symbols and defaults set to true.

        Parameters
        ----------
        symbols : list
            List of strings with chemical symbols to create symmetry functions.

            >>> symbols = ['H', 'O']

        defaults : bool
            Are we building defaults symmetry functions or not?

        type : str
            The supported Gaussian type functions are 'G2', 'G3', and 'G4'.
        etas : list
            List of etas.
        zetas : list
            Lists of zeta values.
        gammas : list
            List of gammas.

        Return
        ------
        GP : dict
            Symmetry function parameters.

        """

        GP = {}

        if defaults:
            logger.warning('Making default symmetry functions')

            for symbol in symbols:
                # Radial
                etas = np.logspace(np.log10(0.05), np.log10(5.), num=4)
                _GP = self.get_symmetry_functions(type='G2', etas=etas,
                                                  symbols=symbols)

                # Angular
                etas = [0.005]
                zetas = [1., 4.]
                gammas = [1., -1.]
                _GP += self.get_symmetry_functions(type='G3', symbols=symbols,
                                                   etas=etas, zetas=zetas,
                                                   gammas=gammas)

                GP[symbol] = _GP
        else:
            pass

        return GP

    def get_symmetry_functions(self, type, symbols, etas=None, zetas=None,
                               gammas=None):
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

        supported_angular_symmetry_functions = ['G3', 'G4']

        if type == 'G2':
            GP = [{'type': 'G2', 'symbol': symbol, 'eta': eta}
                  for eta in etas for symbol in symbols]
            return GP

        elif type in supported_angular_symmetry_functions:
            GP = []
            for eta in etas:
                for zeta in zetas:
                    for gamma in gammas:
                        for idx1, sym1 in enumerate(symbols):
                            for sym2 in symbols[idx1:]:
                                pairs = sorted([sym1, sym2])
                                GP.append({'type': type,
                                           'symbols': pairs,
                                           'eta': eta,
                                           'gamma': gamma,
                                           'zeta': zeta})
            return GP
        else:
            logger.error('The requested type of angular symmetry function is '
                         'not supported.')

    def print_fingerprint_params(self, GP):
        """Print fingerprint parameters"""

        logger.info('Number of features per chemical element:')
        for symbol, v in GP.items():
            logger.info('    - {}: {}.' .format(symbol, len(v)))

        logger.info(' ')
        logger.info('Symmetry function parameters:')
        logger.info('-----------------------------')
        logging.info('{:^5} {:^12} {:4.4} {}' .format('#', 'Symbol', 'Type',
                                                      'Parameters'))

        _symbols = []
        for symbol, value in GP.items():
            if symbol not in _symbols:
                _symbols.append(symbol)
                for i, v in enumerate(value):
                    type_ = v['type']
                    eta = v['eta']
                    if type_ == 'G2':
                        symbol = v['symbol']
                        params = '{:^5} {:12.2} {:^4.4} eta: {:.4f}' \
                            .format(i, symbol, type_, eta)
                    else:
                        symbol = str(v['symbols'])[1:-1].replace("'", "")
                        gamma = v['gamma']
                        zeta = v['zeta']
                        params = '{:^5} {:12} {:^4.5} eta: {:.4f} ' \
                                 'gamma: {:7.4f} zeta: {:.4f}' \
                                 .format(i, symbol, type_, eta, gamma, zeta)

                    logging.info(params)


def calculate_G2(n_numbers, neighborsymbols, neighborpositions, center_symbol,
                 eta, cutoff, cutofffxn, Ri, normalized=True):
    """Calculate G2 symmetry function.

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

    Returns
    -------
    feature : float
        Radial feature.
    """
    feature = 0.

    num_neighbors = len(neighborpositions)

    Rc = cutoff
    feature = 0.
    num_neighbors = len(neighborpositions)

    # Are we normalzing the feature?
    if normalized:
        Rc = cutoff
    else:
        Rc = 1.

    for count in range(num_neighbors):
        symbol = neighborsymbols[count]
        Rj = neighborpositions[count]

        if symbol == center_symbol:
            Rij = np.linalg.norm(Rj - Ri)

            feature += (np.exp(-eta * (Rij ** 2.) / (Rc ** 2.)) *
                        cutofffxn(Rij))

    return feature


def calculate_G3(n_numbers, neighborsymbols, neighborpositions, G_elements,
                 gamma, zeta, eta, cutoff, cutofffxn, Ri):
    """Calculate G3 symmetry function.

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

    Returns
    -------
    feature : float
        G3 feature value.
    """
    Rc = cutoff
    feature = 0.
    counts = range(len(neighborpositions))
    for j in counts:
        for k in counts[(j + 1):]:
            els = sorted([neighborsymbols[j], neighborsymbols[k]])
            if els != G_elements:
                continue

            Rij_vector = neighborpositions[j] - Ri
            Rij = np.linalg.norm(Rij_vector)
            Rik_vector = neighborpositions[k] - Ri
            Rik = np.linalg.norm(Rik_vector)
            Rjk_vector = neighborpositions[k] - neighborpositions[j]
            Rjk = np.linalg.norm(Rjk_vector)
            cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
            term = (1. + gamma * cos_theta_ijk) ** zeta
            term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                           (Rc ** 2.))
            term *= cutofffxn(Rij)
            term *= cutofffxn(Rik)
            term *= cutofffxn(Rjk)
            feature += term
    feature *= 2. ** (1. - zeta)
    return feature
