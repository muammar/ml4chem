import dask
import logging
import numpy as np
from ase.data import atomic_numbers
from collections import OrderedDict
from ml4chem.atomistic.features.gaussian import Gaussian, weighted_h
from ml4chem.atomistic.features.cutoff import Cosine

logger = logging.getLogger()


class AEV(Gaussian):
    """Atomic environment vector

    This class build atomic environment vectors as shown in the ANI-1
    potentials.

    Parameters
    ----------
    cutoff : float
        Cutoff radius used for computing features.
    cutofffxn : object
        A Cutoff function object.
    normalized : bool
        Set it to true if the features are being normalized with respect to the
        cutoff radius.
    preprocessor : str
        Use some scaling method to preprocess the data. Default MinMaxScaler.
    custom : dict, opt
        Create custom symmetry functions, and override defaults. Default is
        None. The structure of the dictionary is as follows:

        >>> custom = {'G2': {'etas': etas, 'Rs': rs},
                      'G3': {'etas': a_etas, 'zetas': zetas, 'thetas': thetas, 'Rs': rs}}

    save_preprocessor : str
        Save preprocessor to file.
    scheduler : str
        The scheduler to be used with the dask backend.
    filename : str
        Path to save database. Note that if the filename exists, the features
        will be loaded without being recomputed.
    overwrite : bool
        If overwrite is set to True, ml4chem will not try to load existing
        databases. Default is True.
    angular_type : str
        Compute "G3" or "G4" angular symmetry functions.
    weighted : bool
        True if applying weighted feature of Gaussian function. See Ref. 2.
    batch_size : int
        Number of data points per batch to use for training. Default is None.

    References
    ----------
    1. Smith, J. S., Isayev, O. & Roitberg, A. E. ANI-1: an extensible neural
    network potential with DFT accuracy at force field computational cost.
    Chem. Sci. 8, 3192–3203 (2017).
    """

    NAME = "AEV"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(
        self,
        cutoff=6.5,
        cutofffxn=None,
        normalized=True,
        preprocessor=("MinMaxScaler", None),
        custom=None,
        save_preprocessor="ml4chem",
        scheduler="distributed",
        filename="features.db",
        overwrite=True,
        angular_type="G3",
        weighted=False,
        batch_size=None,
    ):
        super(AEV, self).__init__()

        cutoff_keys = ["radial", "angular"]
        if isinstance(cutoff, (int, float)):
            cutoff = {cutoff_key: cutoff for cutoff_key in cutoff_keys}

        self.normalized = normalized
        self.filename = filename
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.save_preprocessor = save_preprocessor
        self.overwrite = overwrite
        self.angular_type = angular_type.upper()
        self.weighted = weighted
        self.batch_size = batch_size

        # Let's add parameters that are going to be stored in the .params json
        # file.
        self.params = OrderedDict()
        self.params["name"] = self.name()

        # We verify that values of parameters are list otherwise they cannot be
        # serialized by json.

        # These keys are very likely to exist when doing inference
        keys = ["user_input", "GP"]

        if custom is None:
            # custom = {key: custom for key in keys}
            raise NotImplementedError(
                "Please provide a custom dictionary as explained on the docstrings..."
            )
        elif (
            custom is not None and len(list(set(keys).intersection(custom.keys()))) == 0
        ):
            for value in custom.values():
                for k, v in value.items():
                    if isinstance(v, list) is False:
                        value[k] = v.tolist()
            custom = {"user_input": custom}

        self.custom = custom

        # This is a very general way of not forgetting to save variables
        _params = vars()

        # Delete useless variables
        delete = [
            "self",
            "scheduler",
            "overwrite",
            "k",
            "v",
            "value",
            "keys",
            "batch_size",
            "__class__",
            "cutoff_keys",
        ]

        for param in delete:
            try:
                del _params[param]
            except KeyError:
                # In case the variable does not exist we just pass.
                pass

        for k, v in _params.items():
            if v is not None:
                self.params[k] = v

        self.cutoff = cutoff
        self.cutofffxn = {}

        if cutofffxn is None:
            for cutoff_key in cutoff_keys:
                self.cutofffxn[cutoff_key] = Cosine(cutoff=self.cutoff[cutoff_key])
        else:
            raise RuntimeError("This case is not implemented yet...")

    def get_symmetry_functions(
        self, type, symbols, etas=None, zetas=None, Rs=None, Rs_a=None, thetas=None
    ):
        """Get requested symmetry functions

        Parameters
        ----------
        type : str
            The desired symmetry function: 'G2', 'G3', or 'G4'.
        symbols : list
            List of chemical symbols.
        etas : list
            List of etas to build the Gaussian function.
        zetas : list
            List of zetas to build the Gaussian function.
        Rs : list
            List to shift the center of the gaussian distributions.
        Rs_a : list
            List to shift the center of the gaussian distributions of angular
            symmetry functions.
        thetas : list
            Number of shifts in the angular environment.
        """

        supported_angular_symmetry_functions = ["G3", "G4"]

        if type == "G2":
            GP = []
            for eta in etas:
                for symbol in symbols:
                    for rs_ in Rs:
                        GP.append(
                            {"type": "G2", "symbol": symbol, "eta": eta, "Rs": rs_}
                        )
            return GP

        elif type in supported_angular_symmetry_functions:
            GP = []
            for eta in etas:
                for zeta in zetas:
                    for rs_ in Rs_a:
                        for theta in thetas:
                            for idx1, sym1 in enumerate(symbols):
                                for sym2 in symbols[idx1:]:
                                    pairs = sorted([sym1, sym2])
                                    GP.append(
                                        {
                                            "type": type,
                                            "symbols": pairs,
                                            "eta": eta,
                                            "Rs": rs_,
                                            "zeta": zeta,
                                            "theta": theta,
                                        }
                                    )
            return GP
        else:
            raise RuntimeError(
                "The requested type of symmetry function is not supported."
            )

    def print_features_params(self, GP):
        """Print features parameters"""

        logger.info("Number of features per chemical element:")
        for symbol, v in GP.items():
            logger.info("    - {}: {}.".format(symbol, len(v)))

        _symbols = []
        for symbol, value in GP.items():
            logger.info(" ")
            logger.info("Symmetry function parameters for {} atom:".format(symbol))
            underline = "---------------------------------------"

            for _ in range(len(symbol)):
                underline += "-"

            logger.info(underline)
            logging.info(
                "{:^5} {:^12} {:4.4} {}".format("#", "Symbol", "Type", "Parameters")
            )

            if symbol not in _symbols:
                _symbols.append(symbol)
                for i, v in enumerate(value):
                    type_ = v["type"]
                    eta = v["eta"]
                    rs = v["Rs"]
                    if type_ == "G2":
                        symbol = v["symbol"]
                        params = "{:^5} {:12.2} {:^4.4} eta: {:.4f} Rs: {:.4f}".format(
                            i, symbol, type_, eta, rs
                        )
                    else:
                        symbol = str(v["symbols"])[1:-1].replace("'", "")
                        theta = v["theta"]
                        zeta = v["zeta"]
                        params = (
                            "{:^5} {:12} {:^4.5} eta: {:.4f} "
                            "Rs: {:.4f} zeta: {:.4f} theta: {:.4f}".format(
                                i, symbol, type_, eta, rs, zeta, theta
                            )
                        )

                    logging.info(params)

    @dask.delayed
    def get_atomic_features(
        self,
        atom,
        index,
        symbol,
        n_symbols,
        neighborpositions,
        n_indices=None,
        image_molecule=None,
        weighted=False,
    ):
        """Delayed class method to compute atomic features

        Parameters
        ----------
        atom : object
            An ASE atom object.
        image : ase object, list
            List of atoms in an image.
        index : int
            Index of atom in atoms object.
        symbol : str
            Chemical symbol of atom in atoms object.
        n_symbols : ndarray of str
            Array of neighbors' symbols.
        neighborpositions : ndarray of float
            Array of Cartesian atomic positions.
        image_molecule : ase object, list
            List of atoms in an image.
        weighted : bool
            True if applying weighted feature of Gaussian function. See Ref.
            2.
        """

        cutoff_keys = ["radial", "angular"]
        num_symmetries = len(self.GP[symbol])
        Ri = atom.position
        features = [None] * num_symmetries

        # See https://listserv.brown.edu/cgi-bin/wa?A2=ind1904&L=AMP-USERS&P=19048
        # n_numbers = [atomic_numbers[symbol] for symbol in n_symbols]
        n_numbers = [
            atomic_numbers[item]
            for cutoff_key in cutoff_keys
            for item in n_symbols[cutoff_key]
        ]

        for count in range(num_symmetries):
            GP = self.GP[symbol][count]

            if GP["type"] == "G2":
                feature = calculate_G2(
                    n_numbers,
                    n_symbols["radial"],
                    neighborpositions["radial"],
                    GP["symbol"],
                    GP["eta"],
                    GP["Rs"],
                    self.cutoff["radial"],
                    self.cutofffxn["radial"],
                    Ri,
                    image_molecule=image_molecule,
                    n_indices=n_indices,
                    normalized=self.normalized,
                    weighted=weighted,
                )
            elif GP["type"] == "G3":
                feature = calculate_G3(
                    n_numbers,
                    n_symbols["angular"],
                    neighborpositions["angular"],
                    GP["symbols"],
                    GP["theta"],
                    GP["zeta"],
                    GP["eta"],
                    self.cutoff["angular"],
                    self.cutofffxn["angular"],
                    Ri,
                    normalized=self.normalized,
                    image_molecule=image_molecule,
                    n_indices=n_indices,
                    weighted=weighted,
                )
            elif GP["type"] == "G4":
                feature = calculate_G4(
                    n_numbers,
                    n_symbols["angular"],
                    neighborpositions["angular"],
                    GP["symbols"],
                    GP["theta"],
                    GP["zeta"],
                    GP["eta"],
                    GP["Rs"],
                    self.cutoff["angular"],
                    self.cutofffxn["angular"],
                    Ri,
                    normalized=self.normalized,
                    image_molecule=image_molecule,
                    n_indices=n_indices,
                    weighted=weighted,
                )
            else:
                raise NotImplementedError(
                    "The requested symmetry function is not implemented yet..."
                )
            features[count] = feature

        return np.array(features)


def calculate_G2(
    n_numbers,
    neighborsymbols,
    neighborpositions,
    center_symbol,
    eta,
    Rs,
    cutoff,
    cutofffxn,
    Ri,
    image_molecule=None,
    n_indices=None,
    normalized=True,
    weighted=False,
):
    """Calculate G2 symmetry function.

    These correspond to 2 body, or radial interactions.

    Parameters
    ----------
    n_symbols : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : list of str
        List of symbols of all neighbor atoms.
    neighborpositions : list of list of floats
        List of Cartesian atomic positions.
    center_symbol : str
        Chemical symbol of the center atom.
    eta : float
        Parameter of Gaussian symmetry functions.
    Rs : float
        Parameter to shift the center of the peak.
    cutoff : float
        Cutoff radius.
    cutofffxn : object
        Cutoff function.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    normalized : bool
        Whether or not the symmetry function is normalized.
    image_molecule : ase object, list
        List of atoms in an image.
    n_indices : list
        List of indices of neighboring atoms from the image object.
    weighted : bool
        True if applying weighted feature of Gaussian function. See Ref. 2.


    Returns
    -------
    feature : float
        Radial feature.
    """
    feature = 0.0
    num_neighbors = len(neighborpositions)

    for count in range(num_neighbors):
        symbol = neighborsymbols[count]
        Rj = neighborpositions[count]

        if symbol == center_symbol:
            Rij = np.linalg.norm(Rj - Ri)

            feature += np.exp(-eta * ((Rij - Rs) ** 2.0) * cutofffxn(Rij))

            if weighted:
                weighted_atom = image_molecule[n_indices[count]].number
                feature *= weighted_atom

    return feature


def calculate_G4(
    n_numbers,
    neighborsymbols,
    neighborpositions,
    G_elements,
    theta,
    zeta,
    eta,
    Rs,
    cutoff,
    cutofffxn,
    Ri,
    normalized=True,
    image_molecule=None,
    n_indices=None,
    weighted=False,
):
    """Calculate G4 symmetry function.

    These are 3 body or angular interactions.

    Parameters
    ----------
    n_symbols : list of int
        List of neighbors' chemical numbers.
    neighborsymbols : list of str
        List of symbols of neighboring atoms.
    neighborpositions : list of list of floats
        List of Cartesian atomic positions of neighboring atoms.
    G_elements : list of str
        A list of two members, each member is the chemical species of one of
        the neighboring atoms forming the triangle with the center atom.
    theta : float
        Parameter of Gaussian symmetry functions.
    zeta : float
        Parameter of Gaussian symmetry functions.
    eta : float
        Parameter of Gaussian symmetry functions.
    Rs : float
        Parameter to shift the center of the peak.
    cutoff : float
        Cutoff radius.
    cutofffxn : object
        Cutoff function.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    normalized : bool
        Whether or not the symmetry function is normalized.
    image_molecule : ase object, list
        List of atoms in an image.
    n_indices : list
        List of indices of neighboring atoms from the image object.
    weighted : bool
        True if applying weighted feature of Gaussian function. See Ref. 2.

    Returns
    -------
    feature : float
        G4 feature value.

    Notes
    -----
    The difference between the calculate_G3 and the calculate_G4 function is
    that calculate_G4 accounts for bond angles of 180 degrees.
    """
    feature = 0.0
    counts = range(len(neighborpositions))

    for j in counts:
        for k in counts[(j + 1) :]:
            els = sorted([neighborsymbols[j], neighborsymbols[k]])
            if els != G_elements:
                continue

            Rij_vector = neighborpositions[j] - Ri
            Rij = np.linalg.norm(Rij_vector)
            Rik_vector = neighborpositions[k] - Ri
            Rik = np.linalg.norm(Rik_vector)
            cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
            theta_ijk = np.arccos(
                np.clip(cos_theta_ijk, -1.0, 1.0)
            )  # Avoids rounding issues
            cos_theta = np.cos(theta_ijk - theta)
            term = (1.0 + cos_theta) ** zeta
            term *= np.exp(-eta * ((Rij + Rik) / 2.0 - Rs) ** 2.0)

            if weighted:
                term *= weighted_h(image_molecule, n_indices)

            term *= cutofffxn(Rij)
            term *= cutofffxn(Rik)
            feature += term
    feature *= 2.0 ** (1.0 - zeta)
    return feature
