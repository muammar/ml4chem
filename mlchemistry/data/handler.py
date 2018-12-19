from collections import OrderedDict
from mlchemistry.utils import get_hash


class Data(object):
    """A Data class

    An adequate data structure is very important to develop machine-learning
    models. In general a model receives a data set (X) and a target vector (y).
    This class should in principle arrange this in a format that can be
    vectorized and operate not only with neural networks but also with support
    vector machines.

    The central object here is the data set.

    Parameters
    ----------
    images : object or list
        A list of atoms in molecules or solids.
    """

    def __init__(self, images):

        if self.is_valid_structure(images):
            print('Nothing to do')
        else:
            self.prepare_images(images)

    def prepare_images(self, images):
        """Function to prepare images to operate with mlchemistry

        Parameters
        ----------
        images : object
            Images to prepare.

        Returns
        -------
        _images : dict
            Dictionary of images.
        """
        _images = OrderedDict()

        duplicates = 0

        for image in images:
            key = get_hash(image)
            if key in _images.keys():
                duplicates += 1
            else:
                _images[key] = image

        return _images

    def is_valid_structure(self):
        """Check if the data has a valid structure

        """
        print('Do something')
