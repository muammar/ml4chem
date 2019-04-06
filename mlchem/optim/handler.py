def get_optimizer(optimizer, params):
    """docstring for get_optimizer"""

    optimizer_name, kwargs = optimizer

    try:
        optimizer_name = optimizer_name.lower()
    except AttributeError:
        pass

    if optimizer_name is None:
        kwargs = {}
        from mlchem.optim.LBFGS import FullBatchLBFGS
        optimizer = FullBatchLBFGS(params, **kwargs)

    return optimizer
