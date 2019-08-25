#!/usr/bin/env python
# -*- coding: utf-8 -*-

from detector import *
import time

log_file = 'ssh_data_instances.csv'
# log_file = 'ssh_data_instances.csv'
invariant_file = 'invariant_result/ssh.json'

if __name__ == '__main__':
    x_data, orig_data = load_data_instances(log_file)
    feature_extractor = preprocessing.FeatureExtractor()
    x_online, events = feature_extractor.fit_transform(x_data)
    print(events)
    #
    beginOnline = time.time()
    print("Online: begin parse file {}, time: {}".format(log_file, time.strftime("%Y/%m/%d %H:%M:%S",
                                                                                   time.localtime(time.time()))))
    model = InvariantsMiner()
    model.load_invariants(invariant_file, events)
    y_test = model.predict_new(x_online, orig_data)
    model.get_actual_event_repr()
    print("Spend {} seconds".format(time.time() - beginOnline))
    print("Online: finish parse file {}, time: {}".format(log_file, time.strftime("%Y/%m/%d %H:%M:%S",
                                                                                    time.localtime(time.time()))))
