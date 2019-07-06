#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from detector import *

epsilon = 0.5  # threshold for estimating invariant space
longest_invariant = 4

train_struct_log = 'log_result/HDFS_100k.log_structured.csv'
test_struct_log = 'log_result/HDFS_100k.log_structured.csv'

if __name__ == '__main__':
    print("current pid", os.getpid())
    begin = time.time()
    print("begin parse file {}, time: {}".format(train_struct_log,
                                                 time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
    # 1. 训练阶段
    x_train, _ = load_HDFS(train_struct_log,
                           window='session',
                           save_csv=True)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train, events = feature_extractor.fit_transform(x_train)

    model = InvariantsMiner(epsilon=epsilon, longest_invarant=longest_invariant)
    model.fit(x_train, events)
    print("Spent {} seconds".format(time.time() - begin))
    print("finish parse file {}, time: {}".format(train_struct_log,
                                                  time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))

    # 2. 在线检测阶段
    beginOnline = time.time()
    print("Online: begin parse file {}, time: {}".format(test_struct_log,
                                                         time.strftime("%Y/%m/%d %H:%M:%S",
                                                                       time.localtime(time.time()))))
    y_test, y_idx = load_HDFS(test_struct_log, window='session', save_csv=True)
    y_test = feature_extractor.transform(y_test)
    model.predict(y_test, y_idx)
    print("Spend {} seconds".format(time.time() - beginOnline))
    print("Online: finish parse file {}, time: {}".format(test_struct_log, time.strftime("%Y/%m/%d %H:%M:%S",
                                                                                         time.localtime(time.time()))))
