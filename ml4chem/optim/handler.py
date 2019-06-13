import logging
import torch


logger = logging.getLogger()


def get_optimizer(optimizer, params):
    """Get optimizer to train pytorch models

    There are several optimizers available in pytorch, and all of them take
    different parameters. This function takes as arguments an optimizer tuple
    with the following structure:

        >>> optimizer = ('adam', {'lr': 1e-2, 'weight_decay': 1e-6})

    and returns an optimizer object.

    Parameters
    ----------
    optimizer : tuple
        Tuple with name of optimizer and keyword arguments of optimizer as
        shown above.
    params : list
        Parameters obtained from model.parameters() method.

    Returns
    -------
    optimizer : obj
        An optimizer object.

    Notes
    -----
    For a list of all supported optimizers please check:

    https://pytorch.org/docs/stable/optim.html
    """

    optimizer_name, kwargs = optimizer

    try:
        optimizer_name = optimizer_name.lower()
    except AttributeError:
        pass

    if optimizer_name is None:
        kwargs = {
            "lr": 1,
            "history_size": 10,
            "line_search": "Wolfe",
            "dtype": torch.float,
            "debug": False,
        }

        from ml4chem.optim.LBFGS import FullBatchLBFGS

        optimizer_name = "LBFGS"
        optimizer = FullBatchLBFGS(params, **kwargs)

    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, **kwargs)
        optimizer_name = "Adam"

    elif optimizer_name == "lbfgs":
        from ml4chem.optim.LBFGS import FullBatchLBFGS

        optimizer = FullBatchLBFGS(params, **kwargs)
        optimizer_name = "LBFGS"

    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(params, **kwargs)
        optimizer_name = "Adagrad"

    elif optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(params, **kwargs)
        optimizer_name = "Adadelta"

    elif optimizer_name == "sparseadam":
        optimizer = torch.optim.SparseAdam(params, **kwargs)
        optimizer_name = "SparseAdam"

    elif optimizer_name == "adamax":
        optimizer = torch.optim.Adamax(params, **kwargs)
        optimizer_name = "Adamax"

    elif optimizer_name == "asgd":
        optimizer = torch.optim.ASGD(params, **kwargs)
        optimizer_name = "ASGD"

    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(params, **kwargs)
        optimizer_name = "RMSprop"

    elif optimizer_name == "rprop":
        optimizer = torch.optim.Rprop(params, **kwargs)
        optimizer_name = "Rprop"

    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, **kwargs)
        optimizer_name = "SGD"

    logger.info("Optimizer")
    logger.info("---------")
    logger.info("Name: {}.".format(optimizer_name))
    logger.info("Options:")
    for k, v in kwargs.items():
        logger.info("    - {}: {}.".format(k, v))

    logger.info(" ")

    return optimizer_name, optimizer


def get_lr_scheduler(optimizer, lr_scheduler):
    """Get a learning rate scheduler


    With a learning rate scheduler it is possible to perform training with an
    adaptative learning rate.


    Parameters
    ----------
    optimizer : obj
        An optimizer object.
    lr_scheduler : tuple
        Tuple with structure: scheduler's name and a dictionary with keyword
        arguments.

        >>> scheduler = ('ReduceLROnPlateau', {'mode': 'min', 'patience': 10})

    Returns
    -------
    scheduler : obj
        A learning rate scheduler object that can be used to train models.

    Notes
    -----
    For a list of schedulers and respective keyword arguments, please refer to
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
    """

    scheduler_name, kwargs = lr_scheduler
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        name = "ReduceLROnPlateau"

    logger.info("Learning Rate Scheduler")
    logger.info("-----------------------")
    logger.info("    - Name: {}.".format(name))
    logger.info("    - Args: {}.".format(kwargs))
    logger.info("")

    return scheduler
