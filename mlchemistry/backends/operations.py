import numpy as np


class BackendOperations(object):
    """A class for centralizing backend operations

    This class will be growing systematically. This is probably not the best
    solution but can be worked out later.


    Parameters
    ----------
    backend : object
        A backend object: numpy, tensorflow, or pytorch.
    """
    def __init__(self, backend):
        self.backend = backend
        self.name = self.backend.__name__

    def dot(self, a, b):
        """Dot product"""
        if self.name == 'torch':
            if isinstance(a, np.ndarray):
                a = self.backend.Tensor(a).float()
            if isinstance(b, np.ndarray):
                b = self.backend.Tensor(b).float()
            return self.backend.matmul(a, b)
        else:
            return self.backend.dot(a, b)

    def logspace(self, a, b, num):
        """Logspace"""
        if self.name == 'torch':
            return self.backend.logspace(start=float(a), end=float(b),
                                         steps=num)
        else:
            return self.backend.logspace(a, b, num)

    def log10(self, a):
        """Log base 10"""
        if self.name == 'torch':
            a = self.backend.Tensor([a])
        return self.backend.log10(a)

    def norm(self, a):
        """Norm between two vectors"""
        if self.name == 'torch':
            return self.backend.norm(a).float()
        else:
            return self.backend.linalg.norm(a)

    def exp(self, a):
        """Exponential of a number"""
        return self.backend.exp(a)

    def from_numpy(self, a):
        """Convert from numpy to right data type"""
        a = np.array(a)     # This is the safest way
        return self.backend.from_numpy(a).float()

    def to_numpy(self, a):
        """Convert from numpy to right data type"""
        if self.name == 'torch':
            return a.numpy()

    def divide(self, a, b):
        """Divide two vectors/tensors"""
        if self.name == 'torch':
            return self.backend.div(a, b)

    def sum(self, a):
        """Sum a list of values"""
        if self.name == 'torch':
            return self.backend.sum(a)
