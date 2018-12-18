from mlchemistry.utils import get_neighborlist
from .cutoff import CutoffFunction
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

    def calculate_features(self, atoms):
        """Calculate the features per atom in an atoms objects

        Parameters
        ----------
        atoms : ase object, list
            A list of atoms in a molecule or solid.
        """
        for atom in atoms:
            index = atom.index
            symbol = atom.symbol

            nl = get_neighborlist(atoms, cutoff=self.cutoff)
            n_indices, n_offsets = nl[atom.index]
            n_symbols = [atoms[i].symbol for i in n_indices]
            neighborpositions = \
                    [atoms.positions[neighbor] + np.dot(offset, atoms.cell)
                     for (neighbor, offset) in zip(n_indices, n_offsets)]

            self.get_atomic_fingerprint(index, symbol, n_symbols,
                                        neighborpositions)

    def get_atomic_fingerprint(self, index, symbol, n_symbols,
                               neighborpositions):
        """Class method to compute atomic fingerprints


        Parameters
        ----------
        index : int
            Index of atom in atoms object.
        symbol : str
            Chemical symbol of atom in atoms object.
        """
        print("I will do something I promise")
        print(index, symbol, n_symbols, neighborpositions)


def make_symmetry_functions(symbols, defaults=True):
    """Function to make symmetry functions

    Parameters
    ----------
    symbols : list
        List of strings with chemical symbols to create symmetry functions.

        >>> symbols = ['H', 'O']

    defaults : bool
        Are we building defaults symmetry functions or not?

    Return
    ------
    GP : dict
        Symmetry function parameters.

    """

    GP = {}

    if defaults:
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
    else:
        pass

    return GP
