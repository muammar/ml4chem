import hashlib
import importlib
import logging
from ase.neighborlist import NeighborList


def get_hash(image):
    """Get the SHA1 hash of an image object

    Parameters
    ----------
    image : object
        An image to be hashed.

    Returns
    -------
    _hash : str
        Hash of image in string format
    """
    string = ""

    for atom in image:
        string += str(atom)

    sha1 = hashlib.sha1(string.encode("utf-8"))
    _hash = sha1.hexdigest()

    return _hash


def get_neighborlist(image, cutoff):
    """Get the list of neighbors

    Parameters
    ----------
    image : object
        ASE image.

    Returns
    -------
        A list of neighbors with offset distances.
    """
    cutoffs = [cutoff / 2.0] * len(image)
    nlist = NeighborList(
        cutoffs=cutoffs, self_interaction=False, bothways=True, skin=0.0
    )
    nlist.update(image)
    return [nlist.get_neighbors(index) for index in range(len(image))]


def convert_elapsed_time(seconds):
    """Convert elapsed time in seconds to HH:MM:SS format"""

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return int(hours), int(minutes), seconds


def get_chunks(sequence, chunk_size, svm=True):
    """A function that yields a list in chunks

    Parameters
    ----------
    sequence : list or dictionary
        A list or a dictionary to be split.
    chunk_size : int
        Number of elements in each group.
    svm : bool
        Whether or not these chunks are going to be used for kernel methods.
    """
    res = []

    if svm is False and isinstance(sequence, dict):
        sequence = sequence.items()

    for item in sequence:
        res.append(item)

        if len(res) >= chunk_size:
            yield res
            res = []
    if res:
        yield res  # yield the last, incomplete, portion


def dynamic_import(name, package, alt_name=None):
    """A dynamic module importer

    Parameters
    ----------
    name : str
        Name of the module to be imported.
    package : str
        Path to package. Example: ml4chem.fingerprints
    alt_name : str
        Alternative module_name.

    Returns
    -------
    _class : obj
        An class object.
    """

    if alt_name is None:
        module_name = ".{}".format(name.lower())
    else:
        module_name = ".{}".format(alt_name.lower())

    module = importlib.import_module(module_name, package=package)
    imported_class = getattr(module, name)

    return imported_class


def logger(filename=None, level=None, format=None):
    """A wrapper to the logging python module
    
    This module is useful for cases where we need to log in a for loop
    different files. It also will allow more flexibility later on how the
    logging format could evolve.

    Parameters
    ----------
    filename : str, optional
        Name of logfile. If no filename is provided, we output to stdout. 
    level : str, optional
        Level of logging messages, by default 'info'. Supported are: 'info'
        and 'debug'.
    format : str, optional
        Format of logging messages, by default '%(message)s'.
    
    Returns
    -------
    logger
        A logger object.
    """

    levels = {"info": logging.INFO, "debug": logging.DEBUG}

    if level is None:
        level = levels["info"]
    else:
        level = levels[level.lower()]

    if format is None:
        format = "%(message)s"

    # https://stackoverflow.com/a/12158233/1995261
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.basicConfig(filename=filename, level=level, format=format)

    return logger


def get_header_message():
    """Function that returns ML4Chem header"""

    header = """
-------------------------------------------------------------------------------

          ███╗   ███╗██╗██╗  ██╗ ██████╗██╗  ██╗███████╗███╗   ███╗
          ████╗ ████║██║██║  ██║██╔════╝██║  ██║██╔════╝████╗ ████║
          ██╔████╔██║██║███████║██║     ███████║█████╗  ██╔████╔██║
          ██║╚██╔╝██║██║╚════██║██║     ██╔══██║██╔══╝  ██║╚██╔╝██║
          ██║ ╚═╝ ██║███████╗██║╚██████╗██║  ██║███████╗██║ ╚═╝ ██║
          ╚═╝     ╚═╝╚══════╝╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝\n


ML4Chem is Machine Learning for Chemistry. This package is written in Python 3,
and intends to offer modern and rich features to perform machine learning
workflows for chemical physics.

This software is developed by Muammar El Khatib.
-------------------------------------------------------------------------------
"""
    return header
