import logging
import numpy as np
import torch
import joblib

logger = logging.getLogger()


class Preprocessing(object):
    """A wrap for preprocessing data with sklearn

    This intends to be a wrapper around sklearn. The idea is to make easier to
    preprocess data without too much burden to users.

    Parameters
    ----------
    preprocessor : tuple
        Tuple with structure: ('name', {kwargs}).
    purpose : str
        Supported purposes are : 'training', 'inference'.

    Notes
    -----
    The list of preprocessing modules available on sklearn and options can be
    found at:

    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

    If you need a preprocessor that is not implemented yet, just create a bug
    report or follow the structure shown below to implement it yourself (PR are
    very welcomed). In principle, all preprocessors can be implemented.
    """

    def __init__(self, preprocessor, purpose):

        # preprocessor has to be a tuple, but it might be the case that user
        # input is not that.
        if preprocessor is None:
            self.preprocessing = None
            self.kwargs = None
        elif preprocessor is not None and purpose == "training":
            self.preprocessing, self.kwargs = preprocessor
            self.preprocessing = self.preprocessing.lower()
        else:
            self.preprocessing = preprocessor

    def set(self, purpose, numpy=True):
        """Set a preprocessing method

        Parameters
        ----------
        purpose : str
            Supported purposes are : 'training', 'inference'.
        numpy : bool
            Whether we are preparing the preprocessor to work with tensors or
            numpy arrays.

        Returns
        -------
            Preprocessor object.
        """

        logger.info("")

        if self.preprocessing == "minmaxscaler" and purpose == "training":

            if self.kwargs is None:
                self.kwargs = {"feature_range": (-1, 1)}

            if numpy:
                from dask_ml.preprocessing import MinMaxScaler

                self.preprocessor = MinMaxScaler(**self.kwargs)
                preprocessor_name = "MinMaxScaler"
            else:
                self.preprocessor = MinMaxScalerVectorized(**self.kwargs)
                preprocessor_name = "MinMaxScaler"

        elif self.preprocessing == "standardscaler" and purpose == "training":
            from dask_ml.preprocessing import StandardScaler

            if self.kwargs is None:
                self.kwargs = {}
            self.preprocessor = StandardScaler(**self.kwargs)
            preprocessor_name = "StandardScaler"

        elif self.preprocessing == "normalizer" and purpose == "training":
            if self.kwargs is None:
                self.kwargs = {"norm": "l2"}
            from sklearn.preprocessing import Normalizer

            self.preprocessor = Normalizer()
            preprocessor_name = "Normalizer"

        elif self.preprocessing is not None and purpose == "inference":
            logger.info("\nData preprocessing")
            logger.info("------------------")
            logger.info(f"Preprocessor loaded from file : {self.preprocessing}.")
            self.preprocessor = joblib.load(self.preprocessing)

        else:
            logger.warning(
                f"Preprocessor ({self.preprocessing}, {self.kwargs}) is not supported."
            )
            self.preprocessor = preprocessor_name = None

        if purpose == "training" and preprocessor_name is not None:
            logger.info("\nData preprocessing")
            logger.info("------------------")
            logger.info(f"Preprocessor: {preprocessor_name}.")
            logger.info("Options:")
            for k, v in self.kwargs.items():
                logger.info(f"    - {k}: {v}.")

        logger.info(" ")

        return self.preprocessor

    def save_to_file(self, preprocessor, path):
        """Save the preprocessor object to file

        Parameters
        ----------
        preprocessor : obj
            Preprocessing object
        path : str
            Path to save .prep file.
        """
        joblib.dump(preprocessor, path)

    def fit(self, stacked_features, scheduler=None):
        """Fit features

        Parameters
        ----------
        stacked_features : list
            List of stacked features.
        scheduler : str
            What is the scheduler to be used in dask.

        Returns
        -------
        scaled_features : list
            Scaled features using requested preprocessor.
        """

        logger.info("Calling feature preprocessor...")
        if isinstance(stacked_features, np.ndarray):
            # The Normalizer() is not supported by dask_ml.
            self.preprocessor.fit(stacked_features)
            scaled_features = self.preprocessor.transform(stacked_features)
            return scaled_features
        else:
            return self.preprocessor.fit(stacked_features)
            # scaled_features = self.preprocessor.transform(stacked_features)
            # return scaled_features.compute(scheduler=scheduler)

    def transform(self, raw_features):
        """Transform features to scaled features

        Given a Preprocessor object, we return features.

        Parameters
        ----------
        raw_features : list
            Unscaled features.

        Returns
        -------
        scaled_features : list
            Scaled features using the scaler set in self.set().
        """

        scaled_features = self.preprocessor.transform(raw_features)

        try:
            return scaled_features.compute()
        except:
            return scaled_features


class MinMaxScalerVectorized(object):
    """MinMax Scaling

    Transforms each channel to the range [0, 1].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def fit(self, tensor):
        """Fit features

        Parameters
        ----------
        stacked_features : list
            List of stacked features.

        Returns
        -------
        scaled_features 
            A tensor with scaled features using requested preprocessor.
        """

        tensor = torch.stack(tensor)

        # Feature range
        a, b = self.feature_range

        dist = tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        tensor.mul_(b - a).add_(a)

        return tensor
