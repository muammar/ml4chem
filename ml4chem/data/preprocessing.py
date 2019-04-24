import logging
import numpy as np
from sklearn.externals import joblib

logger = logging.getLogger()


class Preprocessing(object):
    """A wrap for preprocessing data with sklearn

    This intends to be a wrapper around sklearn. The idea is to make easier to
    preprocess data without too much burden to users.

    Parameters
    ----------
    preprocesor : tuple
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
    def __init__(self, preprocesor, purpose):

        # preprocesor has to be a tuple, but it might be the case that user
        # input is not that.
        if preprocesor is None:
            self.preprocessing = None
            self.kwargs = None
        elif preprocesor is not None and purpose == 'training':
            self.preprocessing, self.kwargs = preprocesor
            self.preprocessing = self.preprocessing.lower()
        else:
            self.preprocessing = preprocesor

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

        logger.info(' ')

        if self.preprocessing == 'minmaxscaler' and purpose == 'training':
            from dask_ml.preprocessing import MinMaxScaler
            if self.kwargs is None:
                self.kwargs = {'feature_range': (-1, 1)}
            self.preprocessor = MinMaxScaler(**self.kwargs)
            preprocessor_name = 'MinMaxScaler'

        elif self.preprocessing == 'standardscaler' and purpose == 'training':
            from dask_ml.preprocessing import StandardScaler
            if self.kwargs is None:
                self.kwargs = {}
            self.preprocessor = StandardScaler(**self.kwargs)
            preprocessor_name = 'StandardScaler'

        elif self.preprocessing == 'normalizer' and purpose == 'training':
            if self.kwargs is None:
                self.kwargs = {'norm': 'l2'}
            from sklearn.preprocessing import Normalizer
            self.preprocessor = Normalizer()
            preprocessor_name = 'Normalizer'

        elif purpose == 'inference':
            self.preprocessor = joblib.load(self.preprocessing)
        else:
            logger.warning('Preprocessor is not supported.')
            self.preprocessor = preprocessor_name = None

        if purpose == 'training' and preprocessor_name is not None:
            logger.info('Data preprocessing')
            logger.info('------------------')
            logger.info('Preprocessor: {}.' .format(preprocessor_name))
            logger.info('Options:')
            for k, v in self.kwargs.items():
                logger.info('    - {}: {}.' .format(k, v))

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
