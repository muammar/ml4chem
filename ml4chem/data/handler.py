from collections import OrderedDict
from ml4chem.utils import get_hash
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
        List of images. Supported format is from ASE.
    purpose : str
        Is this data for training or inference purpose?. Supported strings are:
        "training", and "inference".
    """

    def __init__(self, images, purpose=None):

        self.images = None
        self.targets = None
        self.unique_element_symbols = None
        self.sorted_molecules = None
        self.forces = []
        logger.info("Data")
        logger.info("====")

        if self.is_valid_structure(images) is False:
            logger.warning("Data structure is not compatible with ML4Chem.")
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
        self.sorted_molecules : dict
            Ordereddict of images of the same molecule type
        self.max_energies_hashed : dict
            Ordereddict of the max energies of the particular molecule type
        self.max_energies : list
            List of the max value of each molecule type

        """
        logger.info("Preparing images for {}...".format(purpose))
        self.images = OrderedDict()

        if purpose == "training":
            self.targets = []
            self.atoms_per_image = []

        duplicates = 0

        self.sorted_molecules = OrderedDict()
        string_hash = str(images[0].symbols)
        if images[0].get_tags()[0] != 0:
            string_hash += "_" + str(images[0].get_tags()[0])
        self.sorted_molecules[string_hash] = []
        self.sorted_molecules[string_hash].append(images[0])
        for image in images:
            self.forces.append(image.get_forces())
            key = get_hash(image)
            if key in self.images.keys():
                duplicates += 1
            else:
                self.images[key] = image
                if purpose == "training":
                    # When purpose is training then you also need targets and
                    # number of atoms in each image

                    #Deals with molecules with the same symbol but different structures
                    string_hash = str(image.symbols)
                    if image.get_tags()[0] != 0:
                        string_hash += "_" + str(image.get_tags()[0])

                    if (str(image.symbols) not in self.sorted_molecules.keys()):
                        self.sorted_molecules[string_hash] = []
                        self.sorted_molecules[string_hash].append(image)
                    else:
                        self.sorted_molecules[string_hash].append(image)

                    self.targets.append(image.get_potential_energy())
                    self.atoms_per_image.append(len(image))

        if purpose == "training":
            max_energies_hashed = OrderedDict()
            min_energies_hashed = OrderedDict()
            max_energies = []
            min_energies = []
            for hash in self.sorted_molecules:
                energies = []
                for energy in range(len(self.sorted_molecules[hash])):
                    energies.append(self.sorted_molecules[hash][energy].get_potential_energy())
                max_energy = max(energies)
                max_index = energies.index(max_energy)
                min_energy = min(energies)
                min_index = energies.index(min_energy)

                max_energy = max_energy / len(images[max_index])
                min_energy = min_energy / len(images[min_index])

                max_energies_hashed[hash] = max_energy
                min_energies_hashed[hash] = min_energy
                max_energies.append(max_energy)
                min_energies.append(min_energy)

                self.max_energies, self.min_energies = max_energies, min_energies
                self.max_energies_hashed = max_energies_hashed
                self.min_energies_hashed = min_energies_hashed

        logger.info("Images hashed and processed...\n")

        if purpose == "training":
            logger.info("There are {} atoms in your data set.".format(sum(self.atoms_per_image)))

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

        supported_categories = ["training", "inference"]

        symbols = {}

        # FIXME make this parallel.
        if purpose in supported_categories:
            if purpose not in symbols.keys():
                symbols[purpose] = {}
                try:
                    symbols[purpose] = sorted(
                        list(set([atom.symbol for image in images for atom in image]))
                    )
                except AttributeError:
                    symbols[purpose] = sorted(
                        list(
                            set(
                                [
                                    atom.symbol
                                    for key, image in images.items()
                                    for atom in image
                                ]
                            )
                        )
                    )

            else:
                # FIXME
                logger.warning("what happens in the following case?")
        else:
            logger.warning("The requested purpose is not supported...")
            symbols = None

        self.unique_element_symbols = symbols

        return self.unique_element_symbols


    def get_data(self, purpose=None):
        """
        Parameters
        ----------
        purpose : str
            The purpose of the data so that structure is prepared accordingly.
            Supported are: 'training', 'inference'
        """

        if purpose == "training":
            return self.images, self.targets
        else:
            return self.images
