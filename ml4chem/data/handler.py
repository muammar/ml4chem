from collections import OrderedDict
from ml4chem.utils import get_hash
import dask
import logging

logger = logging.getLogger()


class DataSet(object):
    """A DataSet class

    An adequate data structure is very important to develop machine-learning
    models. In general a model receives a data set (X) and a target vector (y).
    This class should in principle arrange this in a format that can be
    vectorized and operate not only with neural networks but also with support
    vector machines.

    The central object here is the data set.

    Parameters
    ----------
    images : list or object
        List of images.
    purpose : str
        Are we needing the data for training or inferring?
    """
    def __init__(self, images, purpose=None):

        self.images = None
        self.targets = None
        self.unique_element_symbols = None

        if self.is_valid_structure(images) is False:
            logger.warning('Data structure is not compatible with ML4Chem')
            self.prepare_images(images, purpose=purpose)

    def prepare_images(self, images, purpose=None):
        """Function to prepare images to operate with ML4Chem

        Parameters
        ----------
        images : list or object
            List of images.
        purpose : str
            The purpose of the data so that structure is prepared accordingly.
            Supported are: 'training', 'inference'

        Returns
        -------
        self.images : dict
            Ordered dictionary of images corresponding to order of self.targets
            list.
        self.targets : list
            Targets used for training the model.
        """
        logger.info('Preparing images...')
        self.images = OrderedDict()

        if purpose == 'training':
            self.targets = []
            self.atoms_per_image = []

        duplicates = 0

        for image in images:
            key = get_hash(image)
            if key in self.images.keys():
                duplicates += 1
            else:
                self.images[key] = image
                if purpose == 'training':
                    # When purpose is training then you also need targets and
                    # number of atoms in each image
                    self.targets.append(image.get_potential_energy())
                    self.atoms_per_image.append(len(image))

        if purpose == 'training':
            max_energy = max(self.targets)
            max_index = self.targets.index(max_energy)
            min_energy = min(self.targets)
            min_index = self.targets.index(min_energy)

            max_energy = max_energy / len(images[max_index])
            min_energy = min_energy / len(images[min_index])

            self.max_energy, self.min_energy = max_energy, min_energy
        logger.info('Images hashed and processed...')

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

    def get_unique_element_symbols(self, images=None, purpose=None):
        """Unique element symbol in data set


        Parameters
        ----------
        images : list of images.
            ASE object.
        purpose : str
            The supported categories are: 'training', 'inference'.
        """

        if images is None:
            images = self.images

        supported_categories = ['training', 'inference']

        symbols = {}

        # TODO make this parallel.
        if purpose in supported_categories:
            if purpose not in symbols.keys():
                symbols[purpose] = {}
                try:
                    symbols[purpose] = sorted(list(set([atom.symbol for image
                                                         in images for atom in
                                                         image])))
                except AttributeError:
                    symbols[purpose] = sorted(list(set([atom.symbol for
                                                         key, image in
                                                         images.items() for
                                                         atom in image])))

            else:
                logger.warning('what happens in the following case?')    # FIXME
        else:
            logger.warning('The requested purpose is not supported...')
            symbols = None

        self.unique_element_symbols = symbols

        return self.unique_element_symbols

    def get_images(self, purpose=None):
        """
        Parameters
        ----------
        purpose : str
            The purpose of the data so that structure is prepared accordingly.
            Supported are: 'training', 'inference'
        """

        if purpose == 'training':
            return self.images, self.targets
        else:
            return self.images
