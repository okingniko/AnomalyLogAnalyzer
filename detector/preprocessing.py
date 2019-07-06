"""
The interface for data preprocessing.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re
import sys
from collections import Counter
from scipy.special import expit
from itertools import compress


class FeatureExtractor(object):

    def __init__(self):
        self.events = None

    def fit_transform(self, X_seq):
        """ Fit and transform the data matrix

        Arguments
        ---------
            X_seq: ndarray, log sequences matrix

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed train data summary ======')
        # print("X_seq shape", X_seq.shape)
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        X_new = X_df.values
        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new, X_df.columns

    def get_event_repr(self, invariant_dict):
        for key, value in range(invariant_dict):
            print(key, value)

    def transform(self, X_seq):
        """ Transform the data matrix with trained parameters

        Arguments
        ---------
            X_seq: log sequences matrix

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        # 不在的event将其补0
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        # 按照events的顺序取出
        X = X_df[self.events].values

        print('Test data shape: {}-by-{}\n'.format(X.shape[0], X.shape[1]))
        return X
