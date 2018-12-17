class CutoffFunction(object):
    """Cutoff function class
    """
    def __init__(self, function='cosine', cutoff='6.5'):

        function_caller = {'cosine': cosine}

        function_caller[function](cutoff)

    def cosine(self, cutoff):
        """docstring for cosine"""
        print(cutoff)

