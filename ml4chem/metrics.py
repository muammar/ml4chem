import torch


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
        outputs = torch.cat(outputs)

    if isinstance(targets, list):
        targets = torch.cat(targets)

    # When doing atomistic models then atoms_per_image exists.
    if atoms_per_image is not None:
        # Dimensions do not match
        outputs = outputs / atoms_per_image
        targets = targets / atoms_per_image

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
        outputs = torch.cat(outputs)

    if isinstance(targets, list):
        targets = torch.cat(targets)

    # When doing atomistic models then atoms_per_image exists.
    if atoms_per_image is not None:
        # Dimensions do not match
        outputs = outputs / atoms_per_image
        targets = targets / atoms_per_image

    mse = torch.mean((outputs - targets).pow(2)).item()
    return mse
