from ml4chem.atomistic.features.autoencoders import LatentFeatures
from ml4chem.atomistic.features.cartesian import Cartesian
from ml4chem.atomistic.features.gaussian import Gaussian


__all__ = ["LatentFeatures", "Cartesian", "Gaussian"]

try:
    from ml4chem.atomistic.features.coulombmatrix import CoulombMatrix

    __all__.append("CoulombMatrix")
except ModuleNotFoundError:
    pass
