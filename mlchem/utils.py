import hashlib
from collections import OrderedDict

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

