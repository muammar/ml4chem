from mlchemistry.utils import get_neighborlist
from .cutoff import CutoffFunction
import numpy as np


class Gaussian(object):
    """Behler-Parrinello symmetry functions"""
    def __init__(self, images, cutoff=6.5):
        self.images = images
        self.cutoff = cutoff

    def calculate_features(self, images):
        """Calculate the features"""
        for image in images:
            for atom in image:
                nl = get_neighborlist(image, cutoff=self.cutoff)
                n_indices, n_offsets = nl[atom.index]
                n_symbols = [image[i].symbol for i in n_indices]
                neighborpositions = \
                        [image.positions[neighbor] + np.dot(offset, image.cell)
                         for (neighbor, offset) in zip(n_indices, n_offsets)]
