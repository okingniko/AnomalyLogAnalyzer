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

label_file = ''
epsilon = 0.5  # threshold for estimating invariant space

struct_log_list = ['log_result/HDFS.log_structured.csv']

if __name__ == '__main__':
    print("current pid", os.getpid())
    # time.sleep(20)
    for struct_log in struct_log_list:
        begin = time.time()
        print("begin parse file {}, time: {}".format(struct_log,
                                                     time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
        # Load structured log without label info
        x_train, x_test = load_HDFS(struct_log,
                                    window='session',
                                    train_ratio=1.0,
                                    save_csv=True)
        # Feature extraction
        feature_extractor = preprocessing.FeatureExtractor()
        x_train, events = feature_extractor.fit_transform(x_train)

        # Model initialization and training
        model = InvariantsMiner(epsilon=epsilon)
        model.fit(x_train, events)
        print("Spent {} seconds".format(time.time() - begin))
        print("finish parse file {}, time: {}".format(struct_log,
                                                      time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
        #
        # Predict anomalies on the training set offline, and manually check for correctness
        # print(y_train)
        #
        # Predict anomalies on the test set to simulate the online mode
        # x_test may be loaded from another log file
        # beginOnline = time.time()
        # print("Online: begin parse file {}, time: {}".format(struct_log, time.strftime("%Y/%m/%d %H:%M:%S",
        #                                                                                time.localtime(time.time()))))
        # x_test = feature_extractor.transform(x_test)
        # y_test = model.predict(x_test)
        # print("Spend {} seconds".format(time.time() - beginOnline))
        # print("Online: finish parse file {}, time: {}".format(struct_log, time.strftime("%Y/%m/%d %H:%M:%S",
        #                                                                                 time.localtime(time.time()))))

        # print(y_test)
        #
