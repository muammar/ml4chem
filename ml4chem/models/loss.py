import torch


def AtomicMSELoss(outputs, targets, atoms_per_image):
    """Default loss function

    If user does not input loss function we provide mean-squared error loss
    function.

    Parameters
    ----------
    outputs : tensor
        Outputs of the model.
    targets : tensor
        Expected value of outputs.


    Returns
    -------
    loss : tensor
        The value of the loss function.
    """

    criterion = torch.nn.MSELoss(reduction='sum')
    outputs_atom = torch.div(outputs, atoms_per_image)
    targets_atom = torch.div(targets, atoms_per_image)

    loss = criterion(outputs_atom, targets_atom) * .5

    return loss


def SumSquaredDiff(outputs, targets):
    """Sum of squared differences loss function

    This is the default loss function for a real-valued autoencoder.

    Parameters
    ----------
    outputs : tensor
        Outputs of the model.
    targets : tensor
        Expected value of outputs.

    Returns
    -------
    loss : tensor
        The value of the loss function.

    Notes
    -----
    In the literature it is mentioned that for real-valued autoencoders the
    reconstruction loss function is the sum of squared differences.
    """
    loss = (outputs - targets).pow(2).sum() * .5
    return loss


def MSELoss(outputs, targets):
    """Default loss function

    If user does not input loss function we provide mean-squared error loss
    function.

    Parameters
    ----------
    outputs : tensor
        Outputs of the model.
    targets : tensor
        Expected value of outputs.
    device : str
        Calculation can be run in the cpu or cuda (gpu).

    Returns
    -------
    loss : tensor
        The value of the loss function.
    """

    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, targets) * .5
    return loss
