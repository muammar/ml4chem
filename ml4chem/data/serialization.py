import torch
import msgpack
import msgpack_numpy as m


def dump(data, filename="data.db"):
    """Serialize data

    This function allows to dump data and ML4Chem dictionaries serialized
    with msgpack, or torch (depending on the models).

    Parameters
    ----------
    data : dict or array
        A dictionary or array containting data to be saved to file using
        msgpack.
    filename : str
        Name of file to save in disk.
    """

    # Let's try to dump the data with msgpack, if that does not work we assume
    # torch.
    try:
        with open(filename, "wb") as f:
            f.write(msgpack.packb(data, default=m.encode, use_bin_type=True))
    except TypeError:
        torch.save(data, filename)


def load(filename):
    """Load a msgpack file

    Parameters
    ----------
    filename : str
        Path of file to load from disk.
    """

    # Let's try to open a serialized data file with msgpack, if that does not
    # work we again assume it is a torch serialized file.
    try:
        with open(filename, "rb") as f:
            content = msgpack.unpackb(f.read(), object_hook=m.decode)
    except msgpack.exceptions.ExtraData:
        content = torch.load(filename)

    return content
