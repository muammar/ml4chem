import hashlib
from collections import OrderedDict
from ase.neighborlist import NeighborList

def prepare_images(images):
    """Function to prepare a dictionary of images to operate with mlchem

    Parameters
    ----------
    images : object
        Images to prepare.

    Returns
    -------
    _images : dict
        Dictionary of images.
    """
    _images = OrderedDict()

    duplicates = 0

    for image in images:
        key = get_hash(image)
        if key in _images.keys():
            duplicates += 1
        else:
            _images[key] = image

    return _images

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
