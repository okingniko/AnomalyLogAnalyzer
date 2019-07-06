#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' This is a demo file for the Invariants Mining model.
    API usage:
        dataloader.load_syslog(): load syslog dataset
        feature_extractor.fit_transform(): fit and transform features
        feature_extractor.transform(): feature transform after fitting
        model.fit(): fit the model
        model.predict(): predict anomalies on given data
        model.evaluate(): evaluate model accuracy with labeled data
'''

from detector import *
import time
import os

epsilon = 0.5  # threshold for estimating invariant space
longest_invarant = 4

train_struct_log = 'log_result/auth.log_structured.csv'
test_struct_log = 'log_result/auth.log_structured.csv'

if __name__ == '__main__':
    print("current pid", os.getpid())
    begin = time.time()
    print("begin parse file {}, time: {}".format(train_struct_log,
                                                 time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
    # 1. 训练阶段
    x_train, _ = load_syslog(train_struct_log,
                             window='session',
                             save_csv=True)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train, events = feature_extractor.fit_transform(x_train)

    model = InvariantsMiner(epsilon=epsilon, longest_invarant=longest_invarant)
    model.fit(x_train, events)
    print("Spent {} seconds".format(time.time() - begin))
    print("finish parse file {}, time: {}".format(train_struct_log,
                                                  time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))

    # 2. 在线检测阶段
    beginOnline = time.time()
    print("Online: begin parse file {}, time: {}".format(test_struct_log,
                                                         time.strftime("%Y/%m/%d %H:%M:%S",
                                                                       time.localtime(time.time()))))
    y_test, y_idx = load_syslog(test_struct_log, window='session', save_csv=True)
    y_test = feature_extractor.transform(y_test)
    model.predict(y_test, y_idx)
    print("Spend {} seconds".format(time.time() - beginOnline))
    print("Online: finish parse file {}, time: {}".format(test_struct_log, time.strftime("%Y/%m/%d %H:%M:%S",
                                                                                         time.localtime(time.time()))))