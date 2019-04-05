import torch


def MSELoss(outputs, targets, atoms_per_image, device='cpu'):
    """Default loss function

    If user does not input loss function we provide mean-squared error loss
    function.

    Parameters
    ----------
    outputs : tensor
        Outputs of the model.
    targets : tensor
        Expected value of outputs.
    data : obj
        A data object from mlchem.
    device : str
        Calculation can be run in the cpu or cuda (gpu).

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


def MSELossAE(outputs, targets):
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

    criterion = torch.nn.MSELoss(reduction='mean')
    # TODO verify this is the correct form
    # loss = criterion(outputs, targets) * .5
    loss = criterion(outputs, targets)

    return loss
