#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import time

label_file = ''
epsilon = 0.5  # threshold for estimating invariant space

struct_log_list = ['log_result/HDFS_100k.log_structured.csv']
out_file = 'log_result/HDFS_100k.log_template.csv'

if __name__ == '__main__':
    print("current pid", os.getpid())
    # time.sleep(20)
    for struct_log in struct_log_list:
        begin = time.time()
        print("begin parse file {}, time: {}".format(struct_log,
                                                     time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
        event_count = {}
        event_content = {}
        # Load structured log without label info
        with open(struct_log, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                # print(line['EventId'], line['EventTemplate'])
                event_id = line['EventId']
                if event_id in event_count:
                    event_count[event_id] += 1
                else:
                    event_count[event_id] = 1
                if event_id not in event_content:
                    event_content[event_id] = line['EventTemplate']
        print(event_count, event_content)

        with open(out_file, "w") as csvfile:
            fieldnames = ['EventId', 'EventTemplate', 'Occurrences']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(50):
                event_id = 'E' + str(i)
                if event_id in event_count and event_id in event_content:
                    writer.writerow({'EventId': event_id, 'EventTemplate': event_content[event_id],
                                     'Occurrences': event_count[event_id]})
