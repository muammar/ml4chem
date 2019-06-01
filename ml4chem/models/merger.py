import torch


class ModelMerger(torch.nn.Module):
    """Model Merger

    A class that can merge models and train them simultaneously.

    Parameters
    ----------
    merge : dict
        A dictionary with models, losses, and extrafunctions. The structure
        of the dictionary is the following:

        merge = {'models': [list of models],
                 'losses': [lists of losses for each model in models key]
                 'funcs': [list of extra functions]
                 }
    """

    NAME = "Merged"

    @classmethod
    def name(cls):
        """Returns name of class"""
        return cls.NAME

    def __init__(self, merge):
        super(ModelMerger, self).__init__()
        self.merge = merge

    def forward(self, X):
        for index, model in enumerate(self.merge["models"]):
            if index == 0:
                x = model(X)
            else:
                x = model(x)
        return x