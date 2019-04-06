import logging
import torch


logger = logging.getLogger()


def get_optimizer(optimizer, params):
    """Get optimizer to train pytorch models

    There are several optimizers available in pytorch, and all of them take
    different parameters. This function takes as arguments an optimizer tuple
    with the following structure:

        >>> optimizer = ('adam', kwargs={'lr': 1e-2, 'weight_decay': 1e-6})

    and returns an optimizer object.

    Parameters
    ----------
    optimizer : tuple
        Tuple with name of optimizer and keyword arguments of optimizer as
        shown above.
    params : list
        Parameters obtained from .parameters().

    Returs
    ------
    optimizer : obj
        An optimizer object.
    """

    optimizer_name, kwargs = optimizer

    try:
        optimizer_name = optimizer_name.lower()
    except AttributeError:
        pass

    if optimizer_name is None:
        kwargs = {'lr': 1, 'history_size': 10, 'line_search': 'Wolfe', 'dtype':
                  torch.float, 'debug': False}
        from mlchem.optim.LBFGS import FullBatchLBFGS
        optimizer_name = 'LBFGS'
        optimizer = FullBatchLBFGS(params, **kwargs)

    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params, **kwargs)
        optimizer_name = 'Adam'

    logger.info('Optimizer')
    logger.info('---------')
    logger.info('Name: {}.' .format(optimizer_name))
    logger.info('Arguments:')
    for k, v in kwargs.items():
        logger.info('    - {}: {}.' .format(k, v))

    logger.info(' ')

    return optimizer_name, optimizer
