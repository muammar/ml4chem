import logging
from dask_ml.preprocessing import MinMaxScaler
from sklearn.externals import joblib

logger = logging.getLogger()


class Scaler(object):
    """A class to deal with scalers

    This intends to be a wrapper around sklearn with dask. The idea is to make
    easier to change scalers without too much burden to users.

    Parameters
    ----------
    name : str
        Name of scaler to be used. Supported are:
    """
    def __init__(self, name):
        self.scaler_name = name

    def set_scaler(self, purpose):
        """Set a scaler

        Parameters
        ----------
        purpose : str
            Supported purposes are : 'training', 'inference'.

        Returns
        -------
            Scaler object.
        """

        if self.scaler_name == 'minmaxscaler' and purpose == 'training':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif purpose == 'inference':
            self.scaler = joblib.load(self.scaler_name)
        else:
            logger.warning('{} is not supported.' .format(self.scaler))
            self.scaler = None

        return self.scaler

    def save_scaler_to_file(self, scaler, path):
        """Save the scaler object to file

        Parameter
        ---------
        scaler : obj
            Scaler object
        path : str
            Path to save .scaler file.
        """
        joblib.dump(scaler, path)

    def fit(self, stacked_features, scheduler):
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
            Scaled features using requested scaler.
        """

        if self.scaler_name == 'minmaxscaler':
            self.scaler.fit(stacked_features.compute(scheduler=scheduler))
            scaled_features = self.scaler.transform(stacked_features.compute(
                                                    scheduler=scheduler))
        return scaled_features

    def transform(self, raw_features):
        """Transform features to scaled features

        Given an scaler object, we return scaled features.

        Parameters
        ----------
        raw_features : list
            Unscaled features.

        Returns
        -------
        scaled_features : list
            Scaled features using the scaler set in self.set_scaler.
        """
        scaled_features = self.scaler.transform(raw_features)
        return scaled_features
