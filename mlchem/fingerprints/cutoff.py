import numpy as np


class Cosine(object):
    """Cosine cutoff function

    Parameters
    ----------
    cutoff : float
        The cutoff radius.
    """
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def __call__(self, rij):
        """Function to calculate Cosine cutoff function value

        Parameters
        ----------
        rij : float
            Distance between two atoms.

        Returns
        -------
        cutofffxn : float
            Value of the cutoff function.
        """
        if rij > self.cutoff:
            cutofffxn = 0.
        else:
            cutofffxn = .5 * (np.cos(np.pi * rij / self.cutoff) + 1.)

        return cutofffxn
