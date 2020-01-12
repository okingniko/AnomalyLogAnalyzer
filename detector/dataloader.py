#!/usr/bin/env python
# -*-coding: utf-8 -*-

import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def load_syslog(log_file, window='session', train_ratio=0.5, save_csv=False):
    """ Load structured sys log.

    Arguments
    ---------
        log_file: str, the file path of structured log.
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.

    Returns
    -------
    """

    print('====== Input data summary ======')
    if log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported."
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            cid = row['ComponentAndPID']
            if '[' not in cid:
                continue
            if cid not in data_dict:
                data_dict[cid] = []
            data_dict[cid].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['ComponentAndPID', 'EventSequence'])

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        x_sequence = data_df['EventSequence'].values
        x_idx = data_df['ComponentAndPID'].values
        print('Total: {} instances.'.format(x_sequence.shape[0]))
        return x_sequence, x_idx
    else:
        raise NotImplementedError('load_syslog() only support csv files!')


def load_HDFS(log_file, window='session', train_ratio=0.5, save_csv=False):
    """ Load HDFS sys log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    print('====== Input data summary ======')

    if log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported."
        struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        # Split training and validation set sequentially
        x_sequence = data_df['EventSequence'].values
        x_idx = data_df['BlockId'].values
        print('Total: {} instances.'.format(x_sequence.shape[0]))
        return x_sequence, x_idx
    else:
        raise NotImplementedError('load_HDFS() only support csv files!')


def _split_data(x_data, train_ratio=0):
    num_train = int(train_ratio * x_data.shape[0])
    # print(num_train)
    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    # print(indexes)
    x_train = x_train[indexes]
    return x_train, x_test

def load_data_instances(log_file):
    data_df = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True, converters={"EventSequence": eval})
    # print(data_df)
    x_data = data_df['EventSequence'].values
    print('Total: {} instances'.format(x_data.shape[0]))
    # print(x_data)
    return x_data, data_df
