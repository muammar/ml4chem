from mlchem.potentials import Potentials
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = ['Potentials']

header = """
-------------------------------------------------------------------------------
                _______        _______ _     _ _______ _______
                |  |  | |      |       |_____| |______ |  |  |
                |  |  | |_____ |_____  |     | |______ |  |  |\n


MLChem is Machine Learning for Chemistry. This package is written in Python 3,
and intends to offer modern and rich features to perform machine learning
workflows for chemical physics.

This software is developed by Muammar El Khatib.
-------------------------------------------------------------------------------
"""

logger.info(header)
