from __future__ import division
import re

import pandas as pd
import numpy as np


class DataIO:
    def __init__(self, df, target_label, norm_fn = None, clip_to = None,
                 encode_n_minus_1 = False):
        """Data class with functions for preprocessing and streaming

        Args: df (pandas DataFrame)
              target_label (string)
              norm_fn (function) - optional function for features normalization
              clip_to (iterable) - optional 2-tuple of max/min values for clipping
              encode_n_minus_1 (bool) - encode categorical features to (N-1)-dim,
                rather than N-dim, one-hots (to avoid collinearity)
        """
        self.regex = {'features': '^[^({})]'.format(target_label),
                      'targets': '^{}'.format(target_label)}

        self.df = self.processXs(df, norm_fn, encode_n_minus_1)
        self.df = self.encodeYs(target_label)
        if clip_to:
            # don't touch encoded one-hots
            assert sum(abs(x) > 1 for x in clip_to)
            self.df = self.df.clip(*clip_to)

    @property
    def len_(self):
        return len(self.df)

    @property
    def n_features(self):
        x, _ = self.splitXY()
        return x.shape[1]

    @property
    def n_targets(self):
        _, y = self.splitXY()
        return y.shape[1]

    @staticmethod
    def isBinary(col):
        #return np.product([x in {0, 1} for x in col.values])
        return sum(x not in {0,1} for x in col.values) == 0

    def splitXY(self, df = None):
        """Split given DataFrame into DataFrames representing features & targets"""
        df = df if isinstance(df, pd.DataFrame) else self.df # <-- default
        return tuple(df.filter(regex = self.regex[k])
                     for k in ('features', 'targets'))

    def processXs(self, df, norm_fn = None, encode_n_minus_1 = False):
        """Encode categorical features and, optionally, normalize quantitative
        features by given function"""
        xs, ys = self.splitXY(df)
        xs = self.encodeCategoricals(xs, encode_n_minus_1)
        # only normalize non-binary df columns
        bin_cols = xs.apply(DataIO.isBinary, axis=0)
        if norm_fn:
            xs.loc[:, ~bin_cols] = norm_fn(xs.loc[:, ~bin_cols])
        return pd.concat([xs, ys], axis=1)

    def encodeCategoricals(self, df, encode_n_minus_1 = False):
        """Encode all categorical columns in df (dtype `Object`) as one-hots
        of dimension N (or N-1, if encode_n_minus_1 is True)"""
        return pd.get_dummies(df, columns = df.loc[:, df.dtypes == 'O'],
                              drop_first = encode_n_minus_1)

    def encodeYs(self, target_str):
        """Encode categorical values (labeled with given target_str) as one-hots

        Returns: dataframe including binary {0, 1} columns for unique labels
        """
        try:
            encoded = pd.get_dummies(self.df, columns=[target_str])
        except(ValueError):
            # ignore unlabeled test data - but don't fail silently
            print """
Warning: categorical values not encoded...no targets labeled `{}`
""".format(target_str)
            encoded = self.df
        return encoded

    def kFoldCrossVal(self, k):
        """Generate chunks of data suitable for k-fold cross-validation; i.e.
        yields `k` 2-tuples of DataFrame uniquely chunked into `k-1`-fold
        train and `1`-fold validation sets"""
        # shuffle
        df = self.df.sample(frac=1)
        # chunk into k folds
        #fold_idxs = [range(i, self.len_, k) for i in xrange(k)] TODO: pointers ?
        df_arr = [df[i::k] for i in xrange(k)]

        for i, validate in enumerate(df_arr):
            train_list = df_arr[:i] + df_arr[i+1:]
            train = pd.concat(train_list, axis=0)
            assert len(train) + len(validate) ==  len(df)
            yield (train, validate)

    def stream(self, df = None, batchsize = None, max_iter = np.inf):
        """Generator of minibatches of given batch size, optionally
        limited by maximum numer of iterations over dataset (=epochs).

        Yields: (x,y) tuples of numpy arrays representing features & targets
        """
        df = df if isinstance(df, pd.DataFrame) else self.df # <-- default
        len_ = len(df)
        batchsize = len_ if not batchsize else batchsize
        reps = 0
        while True:
            if reps < max_iter:
                for i in xrange(0, len_, batchsize):
                    try:
                        batched = df[i:i+batchsize]
                    except(IndexError):
                        batched = df[i:]
                    finally:
                        x, y = self.splitXY(batched)
                        yield (x.values, y.values)
                reps += 1
                # shuffle
                df = self.df.sample(frac=1)
            else:
                break

    @staticmethod
    def gaussianNorm(df, mean = None, std = None):
        """Normalize dataframe columns by z-score (mean 0, stdev 1) if params unspecified.
        Else, center by the given mean & normalize by the given stddev."""
        mean = mean if isinstance(mean, pd.Series) else df.mean(axis=0)
        std = std if isinstance(std, pd.Series) else df.std(axis=0)
        return (df - mean) / std

    @staticmethod
    def minMax(df):
        """Center dataframe column ranges to [-1, 1]"""
        max_, min_ = df.max(axis=0), df.min(axis=0)
        midrange = (max_ + min_) / 2
        half_range = (max_ - min_) / 2
        return (df - midrange) / half_range

    @staticmethod
    def centerMeanAndNormalize(df):
        """Center dataframe column means to 0 and scale range to [-1,1]"""
        return minMax(df - df.mean(axis=0))

########################################################################################

def splitTrainValidate(df, perc_training = 0.8):
    """Split dataframe into training and validation sets based on given %"""
    train = df.sample(frac=perc_training)#, random_state=200)
    validate = df.drop(train.index)
    return (train, validate)
