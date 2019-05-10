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

    Returns
    -------
    loss : tensor
        The value of the loss function.
    """

    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, targets) * .5
    return loss


def EncoderMapLoss(outputs, targets, latent, p=2, k_a=1., k_s=1., sig_h=4.5,
                   a_h=12., b_h=6., sig_l=1., a_l=2., b_l=6.):
    """Encodermap loss function

    This is the implementation of the encodermap loss function as proposed by:

    Lemke, T., & Peter, C. (2019). EncoderMap: Dimensionality Reduction and
    Generation of Molecule Conformations. Journal of Chemical Theory and
    Computation, 15(2), 1209–1215. research-article.
    https://doi.org/10.1021/acs.jctc.8b00975

    Parameters
    ----------
    outputs : tensor
        outputs of the model.
    targets : tensor
        Expected value of outputs.
    latent : tensor
        The latent space tensor.
    p : int
        The norm to be computed. Default is L2 norm.
    k_a : float
        Contribution of distance loss function to total loss.

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
    Autores of EncoderMap penalize both the weights using L2 regularization,
    and the magnitude of activations in the latent space layer.  is added in
    the optimizer The L2 regularization is included using weight_decay in the
    optimizer of choice. The activation penalization is computed below.

    """

    loss = 0.

    # Computation of distance loss function
    n = len(outputs)
    loss_auto = torch.dist(outputs, targets, p=p) / n
    loss += k_a * loss_auto

    # Computation of sketch map loss function
    cdist_h = torch.cdist(outputs, outputs)
    cdist_l = torch.cdist(latent, latent)
    sigmoid_h = sigmoid(cdist_h, sig_h, a_h, b_h)
    sigmoid_l = sigmoid(cdist_l, sig_l, a_l, b_l)

    mse = torch.nn.MSELoss()
    loss += mse(sigmoid_h, sigmoid_l) * k_s

    # Computation of activation regularization
    activation_reg = 0.

    for l in latent:
        activation_reg += torch.dot(l, l)

    loss += activation_reg / n

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
    sigmoid = 1 - (1 + (2**(a / b) - 1) * (r / sigma)**a)**(-b / a)
    return sigmoid
