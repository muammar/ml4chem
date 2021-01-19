import torch
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
        try:
            # Serial
            if rij > self.cutoff:
                cutofffxn = 0.0
            else:
                cutofffxn = 0.5 * (np.cos(np.pi * rij / self.cutoff) + 1.0)
        except (ValueError, RuntimeError):
            # Vectorized
            if isinstance(rij, np.ndarray):
                cutofffxn = 0.5 * (np.cos(np.pi * rij / self.cutoff) + 1.0)
                cutofffxn[rij > self.cutoff] = 0
            else:
                cutofffxn = 0.5 * (torch.cos(np.pi * rij / self.cutoff) + 1.0)
                cutofffxn[rij > self.cutoff] = 0

        return cutofffxn
