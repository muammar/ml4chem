import hashlib
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
