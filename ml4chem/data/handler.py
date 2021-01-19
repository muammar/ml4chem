from ase.geometry import wrap_positions
from collections import OrderedDict, Counter
from functools import partial
from ml4chem.data.utils import ase_to_xyz
from ml4chem.utils import get_hash
import pandas as pd
import datetime
import logging
import torch


logger = logging.getLogger()


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
    images : list or object
        List of images. Supported format is from ASE.
    purpose : str
        Is this data for training or inference purpose?. Supported strings are:
        "training", and "inference". Default is "inference".
    forcetraining : bool, optional
        Activate force training. Default is False.
    target_keys : list
        A list with the keys to build targets. For potentials the
        target_keys are ["energies", "forces"].
    svm : bool
        Whether or not these features are going to be used for kernel
        methods.
    """

    def __init__(
        self,
        images,
        purpose="inference",
        forcetraining=False,
        target_keys=None,
        svm=False,
    ):

        self.images = None
        self.targets = None
        self.target_keys = target_keys
        self.unique_element_symbols = None
        self.forcetraining = forcetraining
        self.svm = svm
        logger.info("\nData")
        logger.info("====")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Module accessed on {now}.")

        if self.is_valid_structure(images) is False:
            logger.warning(
                "Data structure is not compatible with ML4Chem but will be automatically prepared for you..."
            )
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
        """
        logger.info(f"Preparing images for {purpose}...")

        if self.target_keys == None:
            self.target_keys = ["energies"]

        if self.forcetraining:
            logger.info("Including forces in targets...")
            self.target_keys.append("forces")

        self.images, self.targets = OrderedDict(), OrderedDict()
        self.atoms_per_image = []

        if purpose == "training":
            for key in self.target_keys:
                self.targets[key] = []

        duplicates = 0

        for image in images:
            key = get_hash(image)
            if key in self.images.keys():
                duplicates += 1
            else:
                if self.svm == False:
                    image.get_positions = partial(get_positions, image)
                    image.arrays["positions"] = torch.tensor(
                        image.positions, requires_grad=False,
                    )

                self.images[key] = image
                if purpose == "training":
                    # When purpose is training then you also need targets and
                    # number of atoms in each image
		     try:
                        self.targets["energies"].append(image.get_potential_energy())
                    except RuntimeError: # Atoms object has no calculator
                        pass
                    self.atoms_per_image.append(len(image))
                    if self.forcetraining:
                        self.targets["forces"].append(image.get_forces())

        if purpose == "training":
            if "energies" in self.target_keys and len(self.targets["energies"]) > 0:
                max_energy = max(self.targets["energies"])
                max_index = self.targets["energies"].index(max_energy)
                min_energy = min(self.targets["energies"])
                min_index = self.targets["energies"].index(min_energy)

                max_energy /= len(images[max_index])
                min_energy /= len(images[min_index])

                self.max_energy, self.min_energy = max_energy, min_energy
        logger.info("Images hashed and processed...\n")

        self.total_number_atoms = self.get_total_number_atoms()
        self.total_number_molecules = len(self.atoms_per_image)

        logger.info(f"There are {self.total_number_atoms} atoms in your data set.")

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
        """A method to get data

        Parameters
        ----------
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

        if purpose == "training":
            return self.images, self.targets
        else:
            return self.images

    def get_total_number_atoms(self):
        """Get the total number of atoms"""
        return sum(self.atoms_per_image)

    def get_largest_number_atoms(self, purpose):
        """
        Parameters
        ----------
        purpose : str
            The purpose of the data so that structure is prepared accordingly.
            Supported are: 'training', 'inference'
        """
        unique_element_symbols = self.get_unique_element_symbols(purpose=purpose)
        self.largest_number_atoms = {
            symbol: 0 for symbol in unique_element_symbols[purpose]
        }

        for _, image in self.images.items():
            numbers = Counter(image.get_chemical_symbols())
            for symbol in unique_element_symbols[purpose]:
                if numbers[symbol] > self.largest_number_atoms[symbol]:
                    self.largest_number_atoms[symbol] = numbers[symbol]

        return self.largest_number_atoms

    def to_pandas(self):
        """Convert data to pandas DataFrame"""
        images = OrderedDict()
        columns = ["xyz"]

        for key, atoms in self.images.items():
            images[key] = ase_to_xyz(atoms, file=False)

        df = pd.DataFrame.from_dict(images, orient="index", columns=columns)
        df["energy"] = self.targets

        return df


"""
Auxiliary functions
"""


def get_positions(self, wrap=False, **wrap_kw):
    """Get array of positions.

    Parameters:

    wrap: bool
        wrap atoms back to the cell before returning positions
    wrap_kw: (keyword=value) pairs
        optional keywords `pbc`, `center`, `pretty_translation`, `eps`,
        see :func:`ase.geometry.wrap_positions`
    """
    if wrap:
        if "pbc" not in wrap_kw:
            wrap_kw["pbc"] = self.pbc
        return wrap_positions(self.positions, self.cell, **wrap_kw)
    else:
        return self.arrays["positions"]
