import msgpack
import msgpack_numpy as m


def dump(arr, filename='data.db'):
    """Dump array to file using msgpack

    Parameters
    ----------
    arr : dict or array
        A dictionary or array containting data to be saved to file using
        msgpack.
    binary : bool
        Whether or not we are storing in binary format.
    filename : str
        Name of file.
    """

    with open(filename, 'wb') as f:
        f.write(msgpack.packb(arr, default=m.encode))


def load(filename):
    """Load a msgpack file

    Parameters
    ----------
    filename : str
        Name of file.
    """

    with open(filename, 'rb') as f:
        content = msgpack.unpackb(f.read(), object_hook=m.decode)
    return content
