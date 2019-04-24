import hashlib
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
    string = ''

    for atom in image:
        string += str(atom)

    sha1 = hashlib.sha1(string.encode('utf-8'))
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
    cutoffs = [cutoff / 2.] * len(image)
    nlist = NeighborList(cutoffs=cutoffs, self_interaction=False,
                         bothways=True, skin=0.)
    nlist.update(image)
    return [nlist.get_neighbors(index) for index in range(len(image))]


def convert_elapsed_time(seconds):
    """Convert elapsed time in seconds to HH:MM:SS format """

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return int(hours), int(minutes), seconds


def get_chunks(sequence, chunk_size, svm=True):
    """A function that yields a list in chunks"""
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
