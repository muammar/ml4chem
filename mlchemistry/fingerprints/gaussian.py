from mlchemistry.utils import get_neighborlist


class Gaussian(object):
    """Behler-Parrinello symmetry functions"""
    def __init__(self, images, cutoff=6.5):
        self.images = images
        self.cutoff = cutoff

    def calculate_features(self, images):
        """Calculate the features"""
        nl = get_neighborlist(images, cutoff=self.cutoff)
        print(nl)
