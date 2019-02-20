from mlchem.utils import get_neighborlist
from mlchem.backends.operations import BackendOperations
from .cutoff import Cosine
from collections import OrderedDict
import numpy
import torch


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
    backend : object
        A backend object.
    scaler : str
        Use some scaling method to preprocess the data.
    """
    def __init__(self, cutoff=6.5, cutofffxn=None, normalized=True,
                 backend=None, scaler=None):

        self.cutoff = cutoff
        self.backend = backend
        self.normalized = normalized
        self.scaler = scaler

        if cutofffxn is None:
            self.cutofffxn = Cosine(cutoff=cutoff)
        else:
            self.cutofffxn = cutofffxn

    def calculate_features(self, images, defaults=True,
                           category='trainingset', data=None):
        """Calculate the features per atom in an atoms objects

        Parameters
        ----------
        image : ase object, list
            A list of atoms.
        defaults : bool
            Are we creating default symmetry functions?
        category : str
            The supported categories are: 'trainingset', 'testset'.
        data : obj
            data object

        Returns
        -------
        feature_space : dict
            A dictionary with key hash and value as a list with the following
            structure: {'hash': [('H', [vector]]}
        """

        feature_space = OrderedDict()

        print()
        print('Fingerprinting')
        print('==============')

        if self.backend is None:
            print('No backend provided')
            self.backend = BackendOperations(numpy)
        else:
            self.backend = BackendOperations(backend)

        if data.unique_element_symbols is None:
            print('Getting unique element symbols for {}' .format(category))
            unique_element_symbols = \
                data.get_unique_element_symbols(images, category=category)
            unique_element_symbols = unique_element_symbols[category]

            print('Unique elements: {}' .format(unique_element_symbols))

        if defaults:
            self.GP = self.make_symmetry_functions(unique_element_symbols,
                                                   defaults=True)

        if self.scaler is not None:
            from sklearn.preprocessing import MinMaxScaler
            stack_features = []
            scaler = MinMaxScaler(feature_range=(-1, 1))

        for key, image in images.items():
            feature_space[key] = []

            image_positions = image.positions

            for atom in image:
                index = atom.index
                symbol = atom.symbol
                nl = get_neighborlist(image, cutoff=self.cutoff)
                n_indices, n_offsets = nl[atom.index]

                n_symbols = [image[i].symbol for i in n_indices]
                neighborpositions = [image_positions[neighbor] +
                                     self.backend.dot(offset, image.cell)
                                     for (neighbor, offset) in
                                     zip(n_indices, n_offsets)]

                feature_vector = self.get_atomic_fingerprint(atom, index,
                                                             symbol, n_symbols,
                                                             neighborpositions,
                                                             self.scaler)
                if self.scaler is not None:
                    stack_features.append(feature_vector[1])
                else:
                    feature_space[key].append(feature_vector)

        if self.scaler is not None:
            scaler.fit(stack_features)
            scaled_feature_space = scaler.transform(stack_features)
            index = 0
            for key, image in images.items():
                for atom in image:
                    symbol = atom.symbol
                    scaled = torch.tensor(scaled_feature_space[index],
                                          requires_grad=True,
                                          dtype=torch.float)
                    feature_space[key].append((symbol, scaled))
                    index += 1

        return feature_space

    def get_atomic_fingerprint(self, atom, index, symbol, n_symbols,
                               neighborpositions, scaler):
        """Class method to compute atomic fingerprints


        Parameters
        ----------
        image : ase object, list
            List of atoms in an image.
        index : int
            Index of atom in atoms object.
        symbol : str
            Chemical symbol of atom in atoms object.
        scaler : str
            Feature scaler.
        """

        num_symmetries = len(self.GP[symbol])
        Ri = atom.position
        fingerprint = [None] * num_symmetries

        for count in range(num_symmetries):
            GP = self.GP[symbol][count]

            if GP['type'] == 'G2':
                feature = calculate_G2(n_symbols, neighborpositions,
                                       GP['symbol'], GP['eta'],
                                       self.cutoff, self.cutofffxn, Ri,
                                       normalized=self.normalized,
                                       backend=self.backend)
            elif GP['type'] == 'G3':
                feature = calculate_G3(n_symbols, neighborpositions,
                                       GP['symbols'], GP['gamma'],
                                       GP['zeta'], GP['eta'], self.cutoff,
                                       self.cutofffxn, Ri,
                                       backend=self.backend)
            elif GP['type'] == 'G4':
                feature = calculate_G4(n_symbols, neighborpositions,
                                       GP['symbols'], GP['gamma'],
                                       GP['zeta'], GP['eta'], self.cutoff,
                                       self.cutofffxn, Ri,
                                       backend=self.backend)
            else:
                print('not implemented')
            fingerprint[count] = feature

        if scaler is None:
            fingerprint = torch.tensor(fingerprint, requires_grad=True,
                                       dtype=torch.float)

        return symbol, fingerprint

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
            print('Making default symmetry functions')
            for symbol in symbols:
                # Radial
                etas = self.backend.logspace(self.backend.log10(0.05),
                                             self.backend.log10(5.), num=4)
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
            print('The requested type of angular symmetry function is not'
                  ' supported.')


def calculate_G2(neighborsymbols, neighborpositions, center_symbol, eta,
                 cutoff, cutofffxn, Ri, normalized=True, backend=None):
    """Calculate G2 symmetry function.

    Parameters
    ----------
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
    backed : object
        A backend.

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

        # Backend checks
        if backend.name == 'torch':
            Ri = backend.from_numpy(Ri)
            Rc = backend.from_numpy(Rc)

        if symbol == center_symbol:
            Rij = backend.norm(Rj - Ri)

            feature += (backend.exp(-eta * (Rij ** 2.) / (Rc ** 2.)) *
                        cutofffxn(Rij))

    return feature


def calculate_G3(neighborsymbols, neighborpositions, G_elements, gamma, zeta,
                 eta, cutoff, cutofffxn, Ri, backend=None):
    """Calculate G3 symmetry function.

    Parameters
    ----------
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
    backend : object
        A backend object.

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

            if backend.name == 'torch':
                Ri = backend.from_numpy(Ri)
            Rij_vector = neighborpositions[j] - Ri
            Rij = backend.norm(Rij_vector)
            Rik_vector = neighborpositions[k] - Ri
            Rik = backend.norm(Rik_vector)
            Rjk_vector = neighborpositions[k] - neighborpositions[j]
            Rjk = backend.norm(Rjk_vector)
            cos_theta_ijk = backend.dot(Rij_vector, Rik_vector) / Rij / Rik
            term = (1. + gamma * cos_theta_ijk) ** zeta
            term *= backend.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                                (Rc ** 2.))
            term *= cutofffxn(Rij)
            term *= cutofffxn(Rik)
            term *= cutofffxn(Rjk)
            feature += term
    feature *= 2. ** (1. - zeta)
    return feature
