import logging
import numpy as np
from sklearn.externals import joblib

logger = logging.getLogger()


class Preprocessing(object):
    """A wrap for preprocessing data with sklearn

    This intends to be a wrapper around sklearn as implemented by dask. The
    idea is to make easier to preprocess data without too much burden to users.

    Parameters
    ----------
    name : str
        Name of preprocessor to be used. Supported are:
    """
    def __init__(self, name):
        self.preprocessing = name.lower()

    def set(self, purpose):
        """Set a preprocessing method

        Parameters
        ----------
        purpose : str
            Supported purposes are : 'training', 'inference'.

        Returns
        -------
            Preprocessor object.
        """

        if self.preprocessing == 'minmaxscaler' and purpose == 'training':
            from dask_ml.preprocessing import MinMaxScaler
            self.preprocessor = MinMaxScaler(feature_range=(-1, 1))
            preprocessor_name = 'MinMaxScaler'

        elif self.preprocessing == 'normalizer' and purpose == 'training':
            from sklearn.preprocessing import Normalizer
            self.preprocessor = Normalizer()
            preprocessor_name = 'Normalizer'

        elif purpose == 'inference':
            self.preprocessor = joblib.load(self.preprocessing)
        else:
            logger.warning('{} is not supported.' .format(self.preprocessor))
            self.preprocessor = None

        logger.info(' ')
        logger.info('Data preprocessing')
        logger.info('------------------')
        logger.info('Preprocessor: {}.' .format(preprocessor_name))
        logger.info(' ')

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
            Scaled features using requested preprocesor.
        """

        if isinstance(stacked_features, np.ndarray):
            # The Normalizer() is not supported by dask_ml.
            self.preprocessor.fit(stacked_features)
            scaled_features = self.preprocessor.transform(stacked_features)
        else:
            self.preprocessor.fit(stacked_features.compute(
                scheduler=scheduler))
            scaled_features = \
                self.preprocessor.transform(
                    stacked_features.compute(scheduler=scheduler))

        return scaled_features

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

        return scaled_features
