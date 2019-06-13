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

    criterion = torch.nn.MSELoss(reduction="sum")
    outputs_atom = torch.div(outputs, atoms_per_image)
    targets_atom = torch.div(targets, atoms_per_image)

    loss = criterion(outputs_atom, targets_atom) * 0.5

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
    loss = (outputs - targets).pow(2).sum() * 0.5
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

    Returns
    -------
    loss : tensor
        The value of the loss function.
    """

    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, targets) * 0.5
    return loss


def EncoderMapLoss(
    inputs,
    outputs,
    targets,
    latent,
    periodicity=float("inf"),
    k_c=1.0,
    k_auto=1.0,
    k_sketch=1.0,
    sigma_h=4.5,
    a_h=12.0,
    b_h=6.0,
    sigma_l=1.0,
    a_l=2.0,
    b_l=6.0,
):
    """Encodermap loss function


    Parameters
    ----------
    inputs : tensor
        Inputs of the model.
    outputs : tensor
        Outputs of the model.
    targets : tensor
        Expected value of outputs.
    latent : tensor
        The latent space tensor.
    periodicity : float
        Defines the distance between periodic walls for the inputs. For example
        2pi for angular values in radians. All periodic data processed by
        EncoderMap must be wrapped to one periodic window. E.g. data with 2pi
        periodicity may contain values from -pi to pi or from 0 to 2pi.
        Default is float("inf") -- non-periodic inputs.
    k_auto : float
        Contribution of distance loss function to total loss.
    k_sketch : float
        Contribution of sketch map loss function to total loss.

    Returns
    -------
    loss : tensor
        The value of the loss function.

    Notes
    -----
    This loss function combines a distance measure between outputs and targets
    plus a sketch-map loss plus a regularization. See Eq. (5) from paper
    referenced above.

    When passing it to the Autoencoder() class, the model basically becomes an
    atom-centered model with the encodermap variant.

    There is something to note about regularization for this loss function.
    Autors of EncoderMap penalize both the weights using L2 regularization,
    and the magnitude of activation in the latent space layer.  is added in
    the optimizer The L2 regularization is included using weight_decay in the
    optimizer of choice. The activation penalization is computed below.

    References
    ----------
    This is the implementation of the encodermap loss function as proposed by:

    1. Lemke, T., & Peter, C. (2019). EncoderMap: Dimensionality Reduction and
       Generation of Molecule Conformations. Journal of Chemical Theory and
       Computation, 15(2), 1209–1215. research-article.
        https://doi.org/10.1021/acs.jctc.8b00975
    """

    loss = 0.0

    # Computation of distance loss function
    distance = get_distance(inputs, outputs, periodicity=periodicity)
    loss_auto = torch.mean(torch.norm(distance, p=2, dim=1))
    loss += k_auto * loss_auto

    # Computation of sketch map loss function
    # FIXME only works with non-periodic systems.
    cdist_i = get_pairwise_distances(inputs, inputs)
    cdist_l = get_pairwise_distances(latent, latent)

    sigmoid_i = sigmoid(cdist_i, sigma_h, a_h, b_h)
    sigmoid_l = sigmoid(cdist_l, sigma_l, a_l, b_l)
    sketch_loss = torch.mean(torch.pow((sigmoid_i - sigmoid_l), 2))
    loss += k_sketch * sketch_loss

    # Computation of activation regularization
    activation_reg = torch.mean(torch.pow(latent, 2))
    loss += k_c * activation_reg

    return loss


"""
Extra functions that might probably move away from this module
"""


def sigmoid(r, sigma, a, b):
    """Sigmoid function

    Parameters
    ----------
    r : array
        Pairwise distances.
    sigma : float
        Location of the inflection point.
    a : float
        Rate in which sigmoid approaches 0 or 1.
    b : float
        Rate in which sigmoid approaches 0 or 1.

    Returns
    -------
    sigmoid : float
        Value of the sigmoid function.
    """
    sigmoid = 1 - (1 + (2 ** (a / b) - 1) * (r / sigma) ** a) ** (-b / a)
    return sigmoid


def get_distance(i, j, periodicity):
    """Get distance between two tensors

    Parameters
    ----------
    i : tensor
        A tensor.
    j : tensor
        A tensor.
    periodicity : float
        Defines the distance between periodic walls for the inputs.

    Returns
    -------
        tensor with distances.
    
    Notes
    -----
    Cases where periodicity is present are not yet supported.
    """
    d = torch.abs(i - j)

    return torch.min(d, periodicity - d)


def get_pairwise_distances(positions, squared=False):
    """Get pairwise distances of a matrix
    
    Parameters
    ----------
    positions : tensor
        Tensor with positions.
    squared : bool, optional
        Whether or not the squared of pairwise distances are computed, by
        default False.
    
    Returns
    -------
    distances
        Pairwise distances.
    """

    dot_product = torch.matmul(positions, positions.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.max(distances, torch.zeros_like(distances))

    if squared is True:
        distances = torch.sqrt(distances)

    return distances
