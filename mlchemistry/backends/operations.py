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
        self.backend_name = self.backend.__name__

    def dot(self, a, b):
        """Dot product"""
        return self.backend.dot(a, b)

    def logspace(self, a, b, num):
        """Logspace"""
        return self.backend.logspace(a, b, num=num)

    def log10(self, a):
        """Log base 10"""
        return self.backend.log10(a)

    def norm(self, a):
        """Norm between two vectors"""
        return self.backend.linalg.norm(a)

    def exp(self, a):
        """Exponential of a number"""
        return self.backend.exp(a)
