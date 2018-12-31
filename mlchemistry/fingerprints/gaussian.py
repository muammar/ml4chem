from mlchemistry.utils import get_neighborlist
from mlchemistry.data.handler import Data
from .cutoff import Cosine
import numpy as np


class Gaussian(object):
    """Behler-Parrinello symmetry functions

    This class builds local chemical environments for atoms based on the
    Behler-Parrinello Gaussian type symmetry functions. It is modular enough
    that can be used just for creating feature spaces.


    Parameters
    ----------
    cutoff : float
        Cutoff radius used for computing fingerprints.
    """
    def __init__(self, cutoff=6.5):
        self.cutoff = cutoff
        self.cutofffxn = Cosine(cutoff=cutoff)
        self.data = Data()

    def calculate_features(self, images, defaults=True,
                           category='trainingset'):
        """Calculate the features per atom in an atoms objects

        Parameters
        ----------
        image : ase object, list
            A list of atoms.
        defaults : bool
            Are we creating default symmetry functions?
        category : str
            The supported categories are: 'trainingset', 'testset'.
        """

        if self.data.unique_element_symbols is None:
            print('Getting unique element symbols for {}' .format(category))
            unique_element_symbols = \
                self.data.get_unique_element_symbols(images, category=category)
            unique_element_symbols = unique_element_symbols[category]

        for image in images:
            for atom in image:
                index = atom.index
                symbol = atom.symbol

                nl = get_neighborlist(image, cutoff=self.cutoff)
                n_indices, n_offsets = nl[atom.index]
                n_symbols = [image[i].symbol for i in n_indices]
                neighborpositions = [image.positions[neighbor] +
                                     np.dot(offset, image.cell)
                                     for (neighbor, offset) in
                                     zip(n_indices, n_offsets)]

                self.get_atomic_fingerprint(index, symbol, n_symbols,
                                            neighborpositions)

    def get_atomic_fingerprint(self, index, symbol, n_symbols,
                               neighborpositions):
        """Class method to compute atomic fingerprints


        Parameters
        ----------
        image : ase object, list
            List of atoms in an image.
        index : int
            Index of atom in atoms object.
        symbol : str
            Chemical symbol of atom in atoms object.
        """

        num_symmetries = len(self.GP[symbol])
        # print("I will do something I promise")
        # print(index, symbol, n_symbols, neighborpositions)
        pass


def make_symmetry_functions(symbols, defaults=True, type=None, etas=None,
                            zetas=None, gammas=None):
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
            etas = np.logspace(np.log10(0.05), np.log10(5.), num=4)
            _GP = get_symmetry_functions(type='G2', etas=etas, symbols=symbols)

            # Angular
            etas = [0.005]
            zetas = [1., 4.]
            gammas = [1., -1.]
            _GP += get_symmetry_functions(type='G3', symbols=symbols,
                                          etas=etas, zetas=zetas,
                                          gammas=gammas)

            GP[symbol] = _GP
        print(GP)
    else:
        pass

    return GP
