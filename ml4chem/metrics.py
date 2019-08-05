import torch
import numpy as np


def compute_rmse(outputs, targets, atoms_per_image=None):
    """Compute RMSE

    Useful when using futures.

    Parameters
    ----------
    outputs : list
        List of outputs.
    targets : list
        List if targets.
    atoms_per_image : list
        List of atoms per image.

    Returns
    -------
    rmse : float
        Root-mean squared error.
    """

    # Concatenate outputs and targets if they come as list of tensors
    if isinstance(outputs, list):
        try:
            outputs = torch.cat(outputs)
            numpy = False
        except TypeError:
            outputs = np.array(outputs)
            numpy = True

    if isinstance(targets, list):
        try:
            targets = torch.cat(targets)
            numpy = False
        except TypeError:
            targets = np.array(targets)
            numpy = True

    # When doing atomistic models then atoms_per_image exists.
    if atoms_per_image is not None:
        # Dimensions do not match
        outputs = outputs / atoms_per_image
        targets = targets / atoms_per_image

    if numpy:
        rmse = np.sqrt((np.square(outputs - targets)).mean())
    else:
        rmse = torch.sqrt(torch.mean((outputs - targets).pow(2))).item()
    return rmse


def compute_mse(outputs, targets, atoms_per_image=None):
    """Compute MSE

    Useful when using futures.

    Parameters
    ----------
    outputs : list
        List of outputs.
    targets : list
        List if targets.
    atoms_per_image : list
        List of atoms per image.

    Returns
    -------
    mse : float
        Mean squared error.
    """

    # Concatenate outputs and targets if they come as list of tensors
    if isinstance(outputs, list):
        try:
            outputs = torch.cat(outputs)
            numpy = False
        except TypeError:
            outputs = np.array(outputs)
            numpy = True

    if isinstance(targets, list):
        try:
            targets = torch.cat(targets)
            numpy = False
        except TypeError:
            targets = np.array(targets)
            numpy = True

    # When doing atomistic models then atoms_per_image exists.
    if atoms_per_image is not None:
        # Dimensions do not match
        outputs = outputs / atoms_per_image
        targets = targets / atoms_per_image

    if numpy:
        mse = (np.square(outputs - targets)).mean()
    else:
        mse = torch.mean((outputs - targets).pow(2)).item()
    return mse
