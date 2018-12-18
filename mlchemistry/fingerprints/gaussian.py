from mlchemistry.utils import get_neighborlist
from .cutoff import CutoffFunction
import numpy as np


class Gaussian(object):
    """Behler-Parrinello symmetry functions

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
        """Class method to compute atomic fingerprint


        Parameters
        ----------
        index : int
            Index of atom in atoms object.
        symbol : str
            Chemical symbol of atom in atoms object.
        """
        print("I will do something I promise")
        print(index, symbol, n_symbols, neighborpositions)


def make_symmetry_functions(symbols):
    """Function to make symmetry functions

    Parameters
    ----------
    symbols : list
        List of strings with chemical symbols to create symmetry functions.

        >>> symbols = ['H', 'O']
    """
    print(symbols)
    pass
