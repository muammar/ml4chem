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

    """
    def __init__(self):
        self.unique_element_symbols = None

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
        print('Preparing images...')
        _images = OrderedDict()
        targets = []

        duplicates = 0

        for image in images:
            key = get_hash(image)
            if key in _images.keys():
                duplicates += 1
            else:
                _images[key] = image
                targets.append(image.get_potential_energy())

        print('Images hashed...')
        return _images, targets

    def is_valid_structure(self, images):
        """Check if the data has a valid structure

        Parameters
        ----------
        images : list of atoms
            List of images.

        Returns
        -------
        valid : bool
            Whether or not the structure is valid.
        """
        if isinstance(images, dict):
            valid = True
        else:
            valid = False

        return valid

    def get_unique_element_symbols(self, images, category=None):
        """Unique element symbol in data set


        Parameters
        ----------
        images : list of images.
            ASE object.
        category : str
            The supported categories are: 'trainingset', 'testset'.
        """

        supported_categories = ['trainingset', 'testset']

        symbols = {}

        if category in supported_categories:
            if category not in symbols.keys():
                symbols[category] = {}
                try:
                    symbols[category] = sorted(list(set([atom.symbol for image in
                                           images for atom in image])))
                except AttributeError:
                    symbols[category] = sorted(list(set([atom.symbol for key, image in
                                           images.items() for atom in image])))

            else:
                print('what happens in the following case?')    # FIXME
        else:
            print('The requested category is not supported...')
            symbols = None

        self.unique_element_symbols = symbols

        return self.unique_element_symbols
