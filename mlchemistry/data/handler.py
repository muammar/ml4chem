from collections import OrderedDict
from mlchemistry.utils import get_hash

class Data(object):
    """A Data class

    An adequate data structure is very important to develop machine-learning
    models. In general a model receives a data set (X) and a target vector (y).
    This class should in principle arrange this in a format that can be
    vectorized and operate not only with neural networks but also with support
    vector machines.
    """

    def prepare_images(self, images):
        """Function to prepare a dictionary of images to operate with mlchemistry

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
